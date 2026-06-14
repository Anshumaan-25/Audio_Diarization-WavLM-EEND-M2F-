"""
Forensic pipeline — layer0_preprocessor.py
===========================================

Purpose:    Layer 0 turns raw clips into a clean, time-anchored acoustic base
            for the rest of the pipeline:
              1. ffmpeg → 16 kHz mono PCM s16le per clip (deterministic flags);
                 SHA-256 each extracted wav into the manifest (provenance).
              2. Build the global session clock (clips laid end-to-end in
                 canonical order, integer-ms offsets).
              3. Silero VAD → speech regions per clip, converted to INTEGER-ms
                 intervals on the global clock and logged.

            Output (`Layer0Result`) feeds Layer 1 (enrollment) and Layer 2
            (tracking): per-file wavs + checksums + durations + offsets, the
            GlobalClock, and merged speech intervals (local + global).

Adapter seam:
            Silero VAD is reached through `VadAdapter` (abstract). The real
            `SileroVad` (vendored at a pinned commit, pre-staged, loaded via
            torch) takes its heavy deps by INJECTION so this file imports with
            no torch. `StubVad` lets the whole layer self-test on plain Python
            with synthetic segments and no ffmpeg (pass-through wavs).

Forensic:   All time is integer ms (forensics.pts). VAD params are fixed
            constants, manifest-logged. A pass-through wav must already be
            16 kHz mono or the layer halts. Manifest is written as we go.

Run / test: python3 layer0_preprocessor.py --selftest    (stdlib only)
"""
from __future__ import annotations

import abc
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from forensics.pts import GlobalClock, merge, samples_to_ms
from forensics.manifest import sha256_file

TARGET_SR = 16000
# Fixed, manifest-logged VAD parameters (match the legacy Layer 0 exactly:
# kept speech segments are widened by 30 ms — speech_pad_ms = 30).
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 100
VAD_SPEECH_PAD_MS = 30


# =============================================================================
# Pure helpers
# =============================================================================

def build_ffmpeg_cmd(src, dst, sr=TARGET_SR):
    """Deterministic 16 kHz mono PCM extraction (audio only, lossless PCM)."""
    return [
        "ffmpeg", "-nostdin", "-y",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sr),
        "-c:a", "pcm_s16le", "-f", "wav",
        str(dst),
    ]


def wav_meta(path):
    """(sample_rate, channels, n_frames) via the stdlib `wave` module."""
    import wave
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getnframes()


def vad_samples_to_intervals_ms(segments, sr):
    """Silero speech segments (sample indices; tuples or {'start','end'} dicts)
    -> merged integer-ms intervals."""
    out = []
    for s in segments:
        if isinstance(s, dict):
            a, b = s["start"], s["end"]
        else:
            a, b = s[0], s[1]
        out.append((samples_to_ms(a, sr), samples_to_ms(b, sr)))
    return merge(out)


# =============================================================================
# VAD adapter seam
# =============================================================================

class VadAdapter(abc.ABC):
    """Returns speech segments as integer sample-index pairs [(start, end), …]."""

    @abc.abstractmethod
    def detect(self, wav_path, sr) -> List[Tuple[int, int]]:
        ...


class StubVad(VadAdapter):
    """Deterministic stub for self-tests: returns a pre-set segment list per
    call, in file order."""

    def __init__(self, per_call):
        self._per_call = [list(x) for x in per_call]
        self._i = 0

    def detect(self, wav_path, sr):
        seg = self._per_call[self._i]
        self._i += 1
        return seg


class SileroVad(VadAdapter):
    """Real Silero VAD. Heavy deps are INJECTED (model + get_speech_timestamps,
    both from the vendored, pinned, pre-staged silero-vad) so this module stays
    torch-free at import. Runs on the Ubuntu GPU box under the determinism flags
    set by the environment gate. Fixed params for reproducibility."""

    def __init__(self, model, get_speech_timestamps, *,
                 threshold=VAD_THRESHOLD, min_speech_ms=VAD_MIN_SPEECH_MS,
                 min_silence_ms=VAD_MIN_SILENCE_MS, speech_pad_ms=VAD_SPEECH_PAD_MS):
        self.model = model
        self._gst = get_speech_timestamps
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms

    def detect(self, wav_path, sr):
        import array
        import wave
        import torch
        with wave.open(str(wav_path), "rb") as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                raise ValueError(f"{wav_path}: expected 16-bit mono PCM")
            frames = w.readframes(w.getnframes())
        pcm = array.array("h")
        pcm.frombytes(frames)
        wav = torch.tensor(pcm, dtype=torch.float32) / 32768.0
        ts = self._gst(
            wav, self.model, threshold=self.threshold, sampling_rate=sr,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        return [(int(t["start"]), int(t["end"])) for t in ts]


# =============================================================================
# Result containers
# =============================================================================

@dataclass
class Layer0File:
    file_index: int
    source: str
    wav_path: str
    wav_sha256: str
    sample_rate: int
    channels: int
    duration_ms: int
    offset_ms: int


@dataclass
class Layer0Result:
    sr: int
    files: List[Layer0File]
    global_clock: GlobalClock
    speech_by_file: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    speech_global: List[Tuple[int, int]] = field(default_factory=list)


# =============================================================================
# Heavy path: extraction (lazy ffmpeg)
# =============================================================================

def ensure_wav(src, work_dir, sr=TARGET_SR):
    """Return a 16 kHz mono wav path. Pass-through wavs must already be 16 kHz
    mono (else halt); other inputs are extracted with ffmpeg into work_dir."""
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"input not found: {src}")
    if src.suffix.lower() == ".wav":
        msr, mch, _ = wav_meta(src)
        if msr != sr or mch != 1:
            raise ValueError(f"{src}: pass-through wav must be {sr} Hz mono, "
                             f"got {msr} Hz / {mch}ch")
        return src
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dst = work_dir / (src.stem + f".{sr // 1000}k.wav")
    cmd = build_ffmpeg_cmd(src, dst, sr)
    import subprocess
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH — install it (apt-get "
                           "install ffmpeg) on the Ubuntu box.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("ffmpeg failed:\n" +
                           exc.stderr.decode("utf-8", "replace")[-2000:])
    return dst


def preprocess(inputs, *, manifest_writer, vad, work_dir, sr=TARGET_SR,
               max_workers=1):
    """Run Layer 0 over the batch (canonical file order). Records layer0_file +
    layer0_vad per clip and a layer0_summary, then returns a Layer0Result.

    ffmpeg extraction is the only heavy CPU step and is per-file independent, so
    with `max_workers > 1` the clips are decoded in parallel (a thread pool —
    ffmpeg is an external process and releases the GIL — saturating the 44-core
    box). The manifest is still written strictly in canonical file order, so the
    hash chain stays deterministic regardless of worker count."""
    files: List[Layer0File] = []
    speech_by_file: Dict[int, List[Tuple[int, int]]] = {}
    offset = 0

    # Stage 1 (parallelisable, no manifest side effects): produce the wavs.
    if max_workers > 1 and len(inputs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(max_workers, len(inputs))) as ex:
            wavs = list(ex.map(lambda s: ensure_wav(s, work_dir, sr), inputs))
    else:
        wavs = [ensure_wav(s, work_dir, sr) for s in inputs]

    # Stage 2 (strictly ordered): meta, checksum, VAD, manifest, clock.
    for idx, src in enumerate(inputs):
        wav = wavs[idx]
        msr, mch, nframes = wav_meta(wav)
        if msr != sr or mch != 1:
            raise ValueError(f"{wav}: extracted wav is not {sr} Hz mono")
        duration_ms = samples_to_ms(nframes, msr)
        wsha = sha256_file(wav)
        manifest_writer.append("layer0_file", {
            "source": str(src), "wav_path": str(wav), "wav_sha256": wsha,
            "sample_rate": msr, "channels": mch,
            "duration_ms": duration_ms, "offset_ms": offset,
        }, file_index=idx, start_ms=0)

        segs = vad.detect(wav, msr)
        local = vad_samples_to_intervals_ms(segs, msr)
        speech_by_file[idx] = local
        manifest_writer.append("layer0_vad", {
            "params": {"threshold": VAD_THRESHOLD,
                       "min_speech_ms": VAD_MIN_SPEECH_MS,
                       "min_silence_ms": VAD_MIN_SILENCE_MS,
                       "speech_pad_ms": VAD_SPEECH_PAD_MS},
            "n_segments": len(local),
            "speech_ms": sum(b - a for a, b in local),
            "segments_ms": [[a, b] for a, b in local],
        }, file_index=idx, start_ms=0)

        files.append(Layer0File(idx, str(src), str(wav), wsha, msr, mch,
                                duration_ms, offset))
        offset += duration_ms

    clock = GlobalClock([f.duration_ms for f in files])
    speech_global = merge([clock.to_global_interval(i, a, b)
                           for i, ivs in speech_by_file.items() for a, b in ivs])
    manifest_writer.append("layer0_summary", {
        "n_files": len(files),
        "session_ms": clock.total_ms,
        "speech_ms": sum(b - a for a, b in speech_global),
    })
    return Layer0Result(sr=sr, files=files, global_clock=clock,
                        speech_by_file=speech_by_file, speech_global=speech_global)


# =============================================================================
# Self-test (stdlib only — no torch, no ffmpeg; pass-through wavs + StubVad)
# =============================================================================

def _selftest():
    import tempfile
    import wave
    from forensics.manifest import ManifestWriter, load_manifest, verify_chain

    # pure conversions
    assert vad_samples_to_intervals_ms([(8000, 24000)], 16000) == [(500, 1500)]
    assert vad_samples_to_intervals_ms([{"start": 0, "end": 16000}], 16000) == [(0, 1000)]
    cmd = build_ffmpeg_cmd("a.mp4", "b.wav", 16000)
    assert cmd[0] == "ffmpeg" and "16000" in cmd and "-vn" in cmd

    with tempfile.TemporaryDirectory() as d:
        def mkwav(p, n_frames):
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\x00\x00" * n_frames)

        f1, f2 = Path(d) / "a.wav", Path(d) / "b.wav"
        mkwav(f1, 16000)   # 1000 ms
        mkwav(f2, 8000)    # 500 ms
        # file0 speech 100–500 ms; file1 speech 0–250 ms (in sample indices)
        vad = StubVad([[(1600, 8000)], [(0, 4000)]])

        mpath = Path(d) / "session.manifest.jsonl"
        with ManifestWriter(mpath, fsync=False) as w:
            w.append("session_start", {"seed": 0})
            res = preprocess([f1, f2], manifest_writer=w, vad=vad,
                             work_dir=Path(d) / "work")

        assert res.global_clock.total_ms == 1500
        assert res.files[0].duration_ms == 1000 and res.files[1].offset_ms == 1000
        assert res.files[0].wav_sha256 == sha256_file(f1)
        assert res.speech_by_file[0] == [(100, 500)]
        assert res.speech_by_file[1] == [(0, 250)]
        assert res.speech_global == [(100, 500), (1000, 1250)]

        entries = load_manifest(mpath)
        assert verify_chain(entries) is True
        ops = [e["operation"] for e in entries]
        assert ops == ["session_start", "layer0_file", "layer0_vad",
                       "layer0_file", "layer0_vad", "layer0_summary"], ops
        # provenance is double-nested per file
        assert entries[1]["payload"]["file_index"] == 0
        assert entries[1]["payload"]["payload"]["wav_sha256"] == sha256_file(f1)

    print("layer0_preprocessor.py self-test: OK (ffmpeg cmd, VAD→int-ms→global "
          "clock, per-file checksums, manifest chain)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
