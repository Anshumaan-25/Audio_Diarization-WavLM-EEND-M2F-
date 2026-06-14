"""
Forensic pipeline — layer2_tracker.py
======================================

Purpose:    Layer 2 runs the EEND-M2F diarizer (over WavLM features) across the
            session and produces the two maps Layer 3 needs:
              - target_active : when the enrolled TARGET is speaking, and
              - overlap       : when ≥2 speakers are simultaneously active
                                (the contamination map).
            Both are integer-ms intervals on the global clock.

Design choices (own judgment — see the project's "test new methods" goal):
            * Per-WINDOW target re-identification. EEND-M2F emits anonymous
              speaker masks per window; the global "which query == which person"
              permutation problem is HARD. We sidestep it: we only care about
              ONE person (the target), so in each fixed window we embed each
              model-speaker's active audio with WavLM and pick the one matching
              the Layer-1 seed (cosine ≥ threshold). No global clustering, fully
              deterministic, and robust to query-index churn across windows.
            * Overlap is IDENTITY-AGNOSTIC: a frame is contaminated iff ≥2 of
              the model's speaker masks are active there — no identity needed.
              This is what powers Layer 3's NaN-only policy.
            * FIXED-SHAPE windows (length/hop) for determinism; the adapter
              reports its own frame rate, which we map to integer ms.

Guarantees preserved: integer-ms time, deterministic (fixed windows, sorted
            iteration, rounded cosines in the log), offline (no I/O here), and
            every window/decision recorded to the hash-chained manifest.

Adapter seam:
            `EendM2fModel.infer(wav, start_ms, end_ms, sr) -> SpeakerActivity`
            (per-speaker 0/1 activity + frame_ms). Real impl wraps WavLM →
            EEND-M2F (deps injected, Ubuntu). WavLM speaker embeddings reuse the
            Layer-1 `WavLMEmbedder` seam. `Stub*` makes all the windowing /
            interval / re-ID logic testable here with no torch.

Run / test: python3 layer2_tracker.py --selftest      (stdlib only)
"""
from __future__ import annotations

import abc
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from forensics.pts import merge
from layer1_enrollment import WavLMEmbedder, StubWavLM, cosine, l2_normalize

# Fixed, manifest-logged tracking parameters (engineering choices).
WINDOW_MS = 10000              # fixed EEND-M2F window length
HOP_MS = 10000                 # non-overlapping (deterministic stitching)
WAVLM_MATCH_THRESHOLD = 0.50   # cosine(seed, speaker-embedding) ≥ ⇒ that speaker is the target
MIN_SPEAKER_ACTIVE_MS = 250    # ignore speakers with too little audio to re-identify


# =============================================================================
# Model output + adapter seam
# =============================================================================

@dataclass
class SpeakerActivity:
    """Per-speaker frame activity for one window. activity[speaker][frame] ∈ {0,1};
    frame_ms is the model's frame period in integer ms."""
    frame_ms: int
    activity: List[List[int]]


class EendM2fModel(abc.ABC):
    @abc.abstractmethod
    def infer(self, wav_path, start_ms, end_ms, sr) -> SpeakerActivity:
        ...


class StubEend(EendM2fModel):
    """`infer_fn(wav_path, start_ms, end_ms, sr) -> SpeakerActivity` for tests."""
    def __init__(self, infer_fn):
        self._fn = infer_fn

    def infer(self, wav_path, start_ms, end_ms, sr):
        return self._fn(wav_path, start_ms, end_ms, sr)


# =============================================================================
# Pure logic (no torch — fully testable here)
# =============================================================================

def windows(total_ms, window_ms, hop_ms):
    """Window schedule over [0, total_ms). Every window is exactly `window_ms`
    long (a FIXED shape for the model) — the final window is anchored to the end
    (`total_ms - window_ms .. total_ms`), overlapping its predecessor rather than
    emitting a short tail that would feed the model a degenerate sub-window
    (overlap is harmless: target/overlap intervals are merged). A session shorter
    than one window yields a single (0, total_ms) window."""
    if total_ms <= 0:
        return []
    if total_ms <= window_ms:
        return [(0, total_ms)]
    out, s = [], 0
    while s + window_ms < total_ms:
        out.append((s, s + window_ms))
        s += hop_ms
    out.append((total_ms - window_ms, total_ms))
    return out


def frames_to_intervals(active_bools, frame_ms, offset_ms=0):
    """Run-length-encode a per-frame boolean into integer-ms intervals; frame f
    spans [offset + f*frame_ms, offset + (f+1)*frame_ms)."""
    out, start = [], None
    for i, v in enumerate(active_bools):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((offset_ms + start * frame_ms, offset_ms + i * frame_ms))
            start = None
    if start is not None:
        out.append((offset_ms + start * frame_ms, offset_ms + len(active_bools) * frame_ms))
    return out


def overlap_bools(activity):
    """Per-frame: True where ≥2 speaker masks are active (identity-agnostic)."""
    if not activity:
        return []
    n_fr = len(activity[0])
    return [sum(spk[f] for spk in activity) >= 2 for f in range(n_fr)]


def pool_embeddings(vectors):
    """Mean of L2-normalised vectors, re-normalised (caller ensures non-empty)."""
    dim = len(vectors[0])
    norm = [l2_normalize(v) for v in vectors]
    mean = [sum(v[i] for v in norm) / len(norm) for i in range(dim)]
    return l2_normalize(mean)


def _infer_windows(model, wav, win, sr):
    """All window activities for one file. Uses the model's batched `infer_batch`
    (one GPU forward over the fixed-shape window batch) when available, else the
    scalar `infer` per window. Behaviour-identical either way."""
    fn = getattr(model, "infer_batch", None)
    if fn is not None:
        return fn(wav, win, sr)
    return [model.infer(wav, ws, we, sr) for ws, we in win]


def _embed_intervals(wavlm, wav, intervals):
    """Embed a flat list of (start_ms, end_ms) slices. Uses `embed_windows_batch`
    (one WavLM forward) when available, else scalar `embed_window`. Same order in,
    same vectors out."""
    if not intervals:
        return []
    fn = getattr(wavlm, "embed_windows_batch", None)
    if fn is not None:
        return fn(wav, intervals)
    return [wavlm.embed_window(wav, a, b) for a, b in intervals]


def identify_target_speaker(embeddings, seed, threshold):
    """The model-speaker index whose embedding best matches the seed at/above
    threshold, plus its cosine. Deterministic ties (lowest index). (None, 0.0)
    if none qualifies."""
    best, best_c = None, None
    for idx in sorted(embeddings):
        v = embeddings[idx]
        if v is None:
            continue
        c = cosine(v, seed)
        if c >= threshold and (best_c is None or c > best_c):
            best, best_c = idx, c
    return best, (best_c if best is not None else 0.0)


# =============================================================================
# Result
# =============================================================================

@dataclass
class Layer2Result:
    target_active: List[Tuple[int, int]]                       # global ms
    overlap: List[Tuple[int, int]]                             # global ms
    target_by_file: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    overlap_by_file: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)


# =============================================================================
# Orchestration
# =============================================================================

def track(layer0, seed, *, manifest_writer, model, wavlm,
          window_ms=WINDOW_MS, hop_ms=HOP_MS,
          match_threshold=WAVLM_MATCH_THRESHOLD,
          min_speaker_active_ms=MIN_SPEAKER_ACTIVE_MS):
    """Run EEND-M2F over each file in fixed windows; return target_active +
    overlap maps on the global clock. Records layer2_window / layer2_file_summary
    / layer2_summary."""
    target_by_file: Dict[int, List[Tuple[int, int]]] = {}
    overlap_by_file: Dict[int, List[Tuple[int, int]]] = {}

    for f in layer0.files:
        fi, wav, dur = f.file_index, f.wav_path, f.duration_ms
        target_local: List[Tuple[int, int]] = []
        overlap_local: List[Tuple[int, int]] = []
        win = windows(dur, window_ms, hop_ms)
        acts = _infer_windows(model, wav, win, layer0.sr)      # batched if supported

        for (ws, we), act in zip(win, acts):
            fm, n_spk = act.frame_ms, len(act.activity)

            ov = frames_to_intervals(overlap_bools(act.activity), fm, ws)
            overlap_local.extend(ov)

            spk_intervals: Dict[int, List[Tuple[int, int]]] = {
                s: frames_to_intervals(act.activity[s], fm, ws) for s in range(n_spk)
            }
            # Re-identify only speakers with enough audio; batch every qualifying
            # speaker's intervals into ONE WavLM call, then redistribute in order.
            qualifying = [s for s in range(n_spk)
                          if sum(b - a for a, b in spk_intervals[s]) >= min_speaker_active_ms]
            flat = [iv for s in qualifying for iv in spk_intervals[s]]
            vecs = _embed_intervals(wavlm, wav, flat)
            spk_embs: Dict[int, Optional[List[float]]] = {s: None for s in range(n_spk)}
            pos = 0
            for s in qualifying:
                k = len(spk_intervals[s])
                spk_embs[s] = pool_embeddings(vecs[pos:pos + k])
                pos += k

            tgt_idx, tgt_cos = identify_target_speaker(spk_embs, seed, match_threshold)
            tgt_ivs = spk_intervals[tgt_idx] if tgt_idx is not None else []
            target_local.extend(tgt_ivs)

            manifest_writer.append("layer2_window", {
                "window_ms": [ws, we], "frame_ms": fm, "n_speakers": n_spk,
                "target_speaker_idx": tgt_idx,
                "target_cosine": round(tgt_cos, 6),       # rounded -> deterministic JSON
                "target_ms": sum(b - a for a, b in tgt_ivs),
                "overlap_ms": sum(b - a for a, b in ov),
                "match_threshold": match_threshold,
                "min_speaker_active_ms": min_speaker_active_ms,
            }, file_index=fi, start_ms=ws)

        tl, ol = merge(target_local), merge(overlap_local)
        target_by_file[fi], overlap_by_file[fi] = tl, ol
        manifest_writer.append("layer2_file_summary", {
            "target_ms": sum(b - a for a, b in tl),
            "overlap_ms": sum(b - a for a, b in ol),
            "n_windows": len(win),
        }, file_index=fi, start_ms=0)

    gc = layer0.global_clock
    target_global = merge([gc.to_global_interval(fi, a, b)
                           for fi, ivs in target_by_file.items() for a, b in ivs])
    overlap_global = merge([gc.to_global_interval(fi, a, b)
                            for fi, ivs in overlap_by_file.items() for a, b in ivs])
    manifest_writer.append("layer2_summary", {
        "session_ms": gc.total_ms,
        "target_ms": sum(b - a for a, b in target_global),
        "overlap_ms": sum(b - a for a, b in overlap_global),
    })
    return Layer2Result(target_active=target_global, overlap=overlap_global,
                        target_by_file=target_by_file, overlap_by_file=overlap_by_file)


# =============================================================================
# Real adapter (heavy deps INJECTED; runs only on the Ubuntu GPU box)
# =============================================================================

class WavLMEendM2f(EendM2fModel):
    """Real EEND-M2F over WavLM features, GPU-resident and batched.

    WavLM is SHARED with Layer 1's seed embedder — the same `wavlm_model` is
    injected here, so the 300M-parameter encoder lives in VRAM exactly once. This
    adapter computes WavLM features (on `device`), then hands them to the injected
    diarization head:

        `forward_fn(features) -> [B, n_speakers, n_frames] of {0,1}`

    where `features` is the WavLM `last_hidden_state` batch `[B, F, D]`. Returning
    a per-window batch lets the Mask2Former head run once over all of a file's
    fixed-shape windows. (Operators whose head wants raw audio instead can pass
    `feature_input=False` and a `forward_fn(wav_batch)`; the contract is otherwise
    identical.) Slices come from the RAM `audio_cache`. `frame_ms` is the head's
    fixed output frame period."""
    def __init__(self, forward_fn, *, frame_ms, sr=16000, device="cuda",
                 wavlm_model=None, audio_cache=None, feature_input=True):
        self._forward = forward_fn
        self._frame_ms = frame_ms
        self._sr = sr
        self._device = device
        self._wavlm = wavlm_model
        self._cache = audio_cache
        self._feature_input = feature_input and wavlm_model is not None

    def infer(self, wav_path, start_ms, end_ms, sr):
        return self.infer_batch(wav_path, [(start_ms, end_ms)], sr)[0]

    def infer_batch(self, wav_path, windows_list, sr):
        """All windows of one file in a single GPU pass. The windows are a fixed
        shape (every entry is `window_ms`), so they stack into one clean batch."""
        import torch
        if not windows_list:
            return []
        if self._cache is not None:
            batch, _lens = self._cache.batch(wav_path, windows_list, device=self._device)
        else:
            batch = self._read_batch(wav_path, windows_list)
        with torch.no_grad():
            model_in = self._wavlm(batch).last_hidden_state if self._feature_input else batch
            out = self._forward(model_in)               # [B, n_speakers, n_frames] 0/1
        results = []
        for b in range(len(windows_list)):
            activity = [[int(v) for v in row] for row in out[b]]
            results.append(SpeakerActivity(frame_ms=self._frame_ms, activity=activity))
        return results

    def _read_batch(self, wav_path, windows_list):
        import array
        import wave
        import torch
        from forensics.pts import ms_to_samples
        segs = []
        with wave.open(str(wav_path), "rb") as w:
            for s, e in windows_list:
                w.setpos(ms_to_samples(s, self._sr))
                n = ms_to_samples(e, self._sr) - ms_to_samples(s, self._sr)
                frames = w.readframes(max(0, n))
                pcm = array.array("h")
                pcm.frombytes(frames)
                segs.append(torch.tensor(pcm, dtype=torch.float32) / 32768.0)
        tmax = max(t.numel() for t in segs)
        segs = [torch.nn.functional.pad(t, (0, tmax - t.numel())) if t.numel() < tmax
                else t for t in segs]
        return torch.stack(segs, dim=0).to(self._device)


# =============================================================================
# Self-test (stdlib only)
# =============================================================================

def _selftest():
    import tempfile
    from pathlib import Path
    from forensics.manifest import ManifestWriter, load_manifest, verify_chain
    from forensics.pts import GlobalClock as GC
    from layer0_preprocessor import Layer0File, Layer0Result

    # --- pure logic ---
    assert windows(10000, 10000, 10000) == [(0, 10000)]            # single full window
    assert windows(20000, 10000, 10000) == [(0, 10000), (10000, 20000)]
    # short tail anchored to the end so every window is exactly window_ms
    assert windows(2500, 1000, 1000) == [(0, 1000), (1000, 2000), (1500, 2500)]
    assert all(e - s == 1000 for s, e in windows(2500, 1000, 1000))
    assert windows(800, 1000, 1000) == [(0, 800)]                  # shorter than one window
    assert frames_to_intervals([0, 1, 1, 0, 1], 20, 100) == [(120, 160), (180, 200)]
    assert overlap_bools([[1, 1, 0], [0, 1, 1]]) == [False, True, False]
    assert identify_target_speaker({0: [1, 0, 0], 1: [0, 1, 0]}, [1, 0, 0], 0.5) == (0, 1.0)
    assert identify_target_speaker({0: None, 1: [0, 1, 0]}, [1, 0, 0], 0.5) == (None, 0.0)
    assert pool_embeddings([[2, 0, 0], [3, 0, 0]]) == [1.0, 0.0, 0.0]

    # --- end-to-end over one 10s window, two model speakers ---
    n = 500
    s0 = [1 if (0 <= f < 150 or 300 <= f < 400) else 0 for f in range(n)]   # target
    s1 = [1 if 350 <= f < 450 else 0 for f in range(n)]                      # other
    act = SpeakerActivity(frame_ms=20, activity=[s0, s1])

    def infer_fn(wav, ws, we, sr):
        return act

    def wavlm_fn(wav, a, b):
        # speaker0's regions start at 0 and 6000 → seed-like; speaker1's at 7000 → other
        return [1.0, 0.0, 0.0] if a in (0, 6000) else [0.0, 1.0, 0.0]

    files = [Layer0File(0, "a.mp4", "a.wav", "sha", 16000, 1, 10000, 0)]
    layer0 = Layer0Result(sr=16000, files=files, global_clock=GC([10000]),
                          speech_by_file={0: []}, speech_global=[])
    seed = [1.0, 0.0, 0.0]

    with tempfile.TemporaryDirectory() as d:
        mp = Path(d) / "session.manifest.jsonl"
        with ManifestWriter(mp, fsync=False) as w:
            w.append("session_start", {"seed": 0})
            res = track(layer0, seed, manifest_writer=w, model=StubEend(infer_fn),
                        wavlm=StubWavLM(wavlm_fn))

        # target = speaker0 ⇒ its two active regions; overlap where s0 & s1 coincide
        assert res.target_active == [(0, 3000), (6000, 8000)], res.target_active
        assert res.overlap == [(7000, 8000)], res.overlap

        # batch-dispatch equivalence: a model/embedder exposing the batched
        # methods must yield byte-identical maps to the scalar fallback (this is
        # the contract the GPU adapters rely on — verified here without torch).
        class BatchEend(StubEend):
            def infer_batch(self, wav, win, sr):
                return [self._fn(wav, ws, we, sr) for ws, we in win]

        class BatchWavLM(StubWavLM):
            def embed_windows_batch(self, wav, intervals):
                return [self._fn(wav, a, b) for a, b in intervals]

        mp_b = Path(d) / "batch.manifest.jsonl"
        with ManifestWriter(mp_b, fsync=False) as w:
            w.append("session_start", {"seed": 0})
            res_b = track(layer0, seed, manifest_writer=w, model=BatchEend(infer_fn),
                          wavlm=BatchWavLM(wavlm_fn))
        assert res_b.target_active == res.target_active
        assert res_b.overlap == res.overlap
        assert res_b.target_by_file == res.target_by_file
        assert _embed_intervals(BatchWavLM(wavlm_fn), "a", []) == []

        entries = load_manifest(mp)
        assert verify_chain(entries) is True
        ops = [e["operation"] for e in entries]
        assert ops == ["session_start", "layer2_window", "layer2_file_summary",
                       "layer2_summary"], ops
        win = entries[1]["payload"]["payload"]
        assert win["target_speaker_idx"] == 0 and win["overlap_ms"] == 1000
        assert entries[-1]["payload"]["overlap_ms"] == 1000

    print("layer2_tracker.py self-test: OK (fixed windows, overlap detection, "
          "per-window seed re-ID, global stitching, manifest chain)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
