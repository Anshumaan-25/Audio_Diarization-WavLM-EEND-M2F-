"""
Forensic pipeline — layer3_contamination.py
============================================

Purpose:    Layer 3 turns Layer 2's maps into the FINAL deliverable: verified,
            uncontaminated target-speech `.wav` clips + their checksums.

NaN-only overlap policy (the hard guarantee):
            EEND-M2F can separate overlapping speech — we are FORBIDDEN from
            emitting separated/reconstructed audio. So any target segment that
            TOUCHES an overlap region (≥2 speakers, from Layer 2) is discarded
            ENTIRELY and logged as a `layer3_nan_block`. We never trim around
            the overlap and keep the rest; the whole segment is contaminated.

Edge-trim refinement:
            Each surviving clean segment is trimmed INWARD by a fixed margin
            (`EDGE_TRIM_MS`) on both ends — strictly conservative (removes
            boundary frames, never adds audio), so it can only strengthen the
            uncontamination guarantee. Segments shorter than `MIN_CLEAN_MS`
            after trimming are dropped (`layer3_discard_short`).

Output + custody:
            Clean clips are written as byte-exact slices of the Layer-0 16 kHz
            wav (stdlib `wave`, deterministic). Each is SHA-256'd into a
            `layer3_segment` record (with start_global_ms/end_global_ms — the
            format the audit/comparator read). A final `output_hash` record
            carries the per-file checksums and a single session output
            fingerprint (sha of the sorted output checksums).

Determinism: integer-ms slicing, fixed params, sorted fingerprint. Re-runs
            produce byte-identical clips and an identical fingerprint.

Run / test: python3 layer3_contamination.py --selftest    (stdlib only — uses a
            real synthetic wav written via the `wave` module)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from forensics.pts import intersect, merge, ms_to_samples
from forensics.manifest import canonical, sha256_file, sha256_hex

# Fixed, manifest-logged parameters (engineering choices).
EDGE_TRIM_MS = 50              # trim inward each end (conservative)
MIN_CLEAN_MS = 500             # drop clean segments shorter than this after trim


# =============================================================================
# Pure logic
# =============================================================================

def nan_partition(target_segs, overlap):
    """Split target segments into (clean, nan): a segment goes to `nan` iff it
    intersects ANY overlap region — discarded whole, per the NaN-only policy."""
    clean, nan = [], []
    for seg in merge(target_segs):
        if intersect([seg], overlap):
            nan.append(seg)
        else:
            clean.append(seg)
    return clean, nan


def edge_trim(seg, trim_ms):
    """Trim inward by trim_ms on both ends. Returns (start, end) which may be
    empty/negative-length (caller enforces the minimum)."""
    return (seg[0] + trim_ms, seg[1] - trim_ms)


# =============================================================================
# Audio (stdlib wave — deterministic byte-exact slice)
# =============================================================================

def write_segment_wav(src_wav, start_ms, end_ms, out_path):
    """Write [start_ms, end_ms) of src_wav to out_path as a byte-exact slice."""
    import wave
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(src_wav), "rb") as r:
        sr = r.getframerate()
        sw, ch = r.getsampwidth(), r.getnchannels()
        r.setpos(ms_to_samples(start_ms, sr))
        n = ms_to_samples(end_ms, sr) - ms_to_samples(start_ms, sr)
        frames = r.readframes(max(0, n))
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(frames)
    return out_path


# =============================================================================
# Result
# =============================================================================

@dataclass
class Layer3Result:
    clean_segments: List[dict]
    output_fingerprint: str
    clean_ms: int = 0
    contaminated_ms: int = 0
    discarded_short: int = 0


# =============================================================================
# Orchestration
# =============================================================================

def contaminate(layer0, layer2, *, manifest_writer, out_dir,
                edge_trim_ms=EDGE_TRIM_MS, min_clean_ms=MIN_CLEAN_MS):
    """Apply the NaN-only policy + edge-trim per file, write clean clips, and
    record everything. Returns a Layer3Result with the session fingerprint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_records: List[dict] = []
    output_shas: List[str] = []
    clean_ms_total = contaminated_ms_total = short_total = 0

    for f in layer0.files:
        fi, src = f.file_index, f.wav_path
        target_local = layer2.target_by_file.get(fi, [])
        overlap_local = layer2.overlap_by_file.get(fi, [])
        clean, nan = nan_partition(target_local, overlap_local)
        n_clean = n_nan = n_short = file_clean = file_cont = 0

        for s, e in nan:                                    # discarded whole
            g0, g1 = layer0.global_clock.to_global_interval(fi, s, e)
            manifest_writer.append("layer3_nan_block", {
                "decision": "NaN", "reason": "overlap",
                "start_local_ms": s, "end_local_ms": e,
                "start_global_ms": g0, "end_global_ms": g1, "duration_ms": e - s,
            }, file_index=fi, start_ms=s)
            n_nan += 1
            file_cont += e - s

        for s, e in clean:                                  # edge-trim then write
            ts, te = edge_trim((s, e), edge_trim_ms)
            if te - ts < min_clean_ms:
                g0, g1 = layer0.global_clock.to_global_interval(fi, s, e)
                manifest_writer.append("layer3_discard_short", {
                    "decision": "DISCARD", "reason": "below_min_clean_ms_after_edge_trim",
                    "start_global_ms": g0, "end_global_ms": g1, "duration_ms": e - s,
                    "trimmed_ms": max(0, te - ts),
                    "edge_trim_ms": edge_trim_ms, "min_clean_ms": min_clean_ms,
                }, file_index=fi, start_ms=s)
                n_short += 1
                continue
            g0, g1 = layer0.global_clock.to_global_interval(fi, ts, te)
            out_path = out_dir / f"target_f{fi:03d}_{g0:09d}-{g1:09d}.wav"
            write_segment_wav(src, ts, te, out_path)
            sha = sha256_file(out_path)
            output_shas.append(sha)
            rec = {
                "decision": "CLEAN",
                "start_local_ms": ts, "end_local_ms": te,
                "start_global_ms": g0, "end_global_ms": g1, "duration_ms": te - ts,
                "wav_path": str(out_path), "wav_sha256": sha,
                "edge_trim_ms": edge_trim_ms,
            }
            manifest_writer.append("layer3_segment", rec, file_index=fi, start_ms=ts)
            clean_records.append(rec)
            n_clean += 1
            file_clean += te - ts

        manifest_writer.append("layer3_file_summary", {
            "n_clean": n_clean, "n_nan": n_nan, "n_short": n_short,
            "clean_ms": file_clean, "contaminated_ms": file_cont,
        }, file_index=fi, start_ms=0)
        clean_ms_total += file_clean
        contaminated_ms_total += file_cont
        short_total += n_short

    fingerprint = sha256_hex(canonical(sorted(output_shas)))
    manifest_writer.append("output_hash", {
        "n_outputs": len(output_shas),
        "output_sha256s": sorted(output_shas),
        "output_fingerprint": fingerprint,
        "clean_ms_total": clean_ms_total,
        "contaminated_ms_total": contaminated_ms_total,
    })
    return Layer3Result(clean_segments=clean_records, output_fingerprint=fingerprint,
                        clean_ms=clean_ms_total, contaminated_ms=contaminated_ms_total,
                        discarded_short=short_total)


# =============================================================================
# Self-test (stdlib only — real synthetic wav)
# =============================================================================

def _selftest():
    import tempfile
    import wave
    from forensics.manifest import ManifestWriter, load_manifest, verify_chain
    from forensics.pts import GlobalClock as GC
    from layer0_preprocessor import Layer0File, Layer0Result
    from layer2_tracker import Layer2Result

    # pure logic
    clean, nan = nan_partition([(0, 2000), (3000, 5000)], [(3500, 4500)])
    assert clean == [(0, 2000)] and nan == [(3000, 5000)]
    assert edge_trim((0, 1000), 50) == (50, 950)

    with tempfile.TemporaryDirectory() as d:
        # a real 6 s, 16 kHz mono wav
        src = Path(d) / "a.wav"
        with wave.open(str(src), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x01\x00" * (16000 * 6))      # 6000 ms

        files = [Layer0File(0, "a.mp4", str(src), "sha", 16000, 1, 6000, 0)]
        layer0 = Layer0Result(sr=16000, files=files, global_clock=GC([6000]),
                              speech_by_file={}, speech_global=[])
        layer2 = Layer2Result(
            target_active=[], overlap=[],
            target_by_file={0: [(0, 2000), (3000, 5000), (5500, 5800)]},
            overlap_by_file={0: [(3500, 4500)]})

        out = Path(d) / "clean"
        mpath = Path(d) / "session.manifest.jsonl"
        with ManifestWriter(mpath, fsync=False) as mw:
            mw.append("session_start", {"seed": 0})
            res = contaminate(layer0, layer2, manifest_writer=mw, out_dir=out)

        # (0,2000) clean -> trim (50,1950) kept; (3000,5000) NaN; (5500,5800) too short
        assert len(res.clean_segments) == 1, res.clean_segments
        seg = res.clean_segments[0]
        assert (seg["start_global_ms"], seg["end_global_ms"]) == (50, 1950)
        assert res.contaminated_ms == 2000 and res.discarded_short == 1

        # the clip exists, is byte-exact length, and its sha matches the record
        clip = Path(seg["wav_path"])
        assert clip.is_file() and sha256_file(clip) == seg["wav_sha256"]
        with wave.open(str(clip), "rb") as r:
            assert r.getnframes() == ms_to_samples(1950, 16000) - ms_to_samples(50, 16000)

        # re-run -> identical fingerprint (bit-exact output)
        out2 = Path(d) / "clean2"
        m2 = Path(d) / "again.manifest.jsonl"
        with ManifestWriter(m2, fsync=False) as mw:
            mw.append("session_start", {"seed": 0})
            res2 = contaminate(layer0, layer2, manifest_writer=mw, out_dir=out2)
        assert res2.output_fingerprint == res.output_fingerprint, "non-deterministic output"

        entries = load_manifest(mpath)
        assert verify_chain(entries) is True
        ops = [e["operation"] for e in entries]
        assert ops == ["session_start", "layer3_nan_block", "layer3_segment",
                       "layer3_discard_short", "layer3_file_summary", "output_hash"], ops
        assert entries[-1]["payload"]["n_outputs"] == 1

    print("layer3_contamination.py self-test: OK (NaN-only whole-segment discard, "
          "edge-trim, short-drop, byte-exact clips + checksums, deterministic "
          "fingerprint, manifest chain)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
