"""
Forensic pipeline — run_ubuntu.py  (the real-run launcher)
==========================================================

This is the ONE file you launch on the Ubuntu RTX 6000 Ada box. It supplies the
three deployment-specific callables the core pipeline cannot ship generically,
then calls `build_real_adapters(...)` + `run_session(...)` on a single
hash-chained manifest. Everything else (the four forensic guarantees, the layers,
the determinism gate) lives in the audited modules and is untouched here.

    PYTHONHASHSEED=0 python3 run_ubuntu.py \
        clip01.mp4 clip02.mp4 ... \
        --clicks operator_clicks.json \
        --models-dir ./models \
        --out ./session_out \
        --device cuda --workers 44

Three seams you wire (search for "WIRE THIS"):
  1. decode_fn(video_path, pts_ms) -> full RGB/BGR frame  (Layer 1 visual anchor)
  2. get_speech_timestamps                                 (Silero VAD, vendored)
  3. eend_forward_fn(features) -> [B, n_speakers, n_frames] of {0,1}
       — your EEND-M2F (Mask2Former) head over the SHARED WavLM last_hidden_state
         batch [B, F, D]. This is the only genuinely custom model; load its
         checkpoint in load_eend_head() below.

Before a real run, the gate SHA-256-verifies every staged weight; an unverified
or missing weight halts. Run the stdlib self-tests first (see the runbook).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The gate must export the offline env vars BEFORE transformers/HF import anywhere.
from environment_gate import enforce_offline
enforce_offline()

from forensics.manifest import ManifestWriter
from pipeline_runner import build_real_adapters, run_session, EXPECTED_MODELS


# =============================================================================
# Seam 1 — exact-PTS video frame decode (return the FULL frame; InsightFace
# re-detects on it and matches the YOLO box by IoU, so do NOT pre-crop).
# =============================================================================
def make_decode_fn():
    """Returns decode_fn(video_path, pts_ms) -> HxWx3 numpy frame.

    OpenCV is the simplest dependency; for guaranteed frame-exact PTS prefer
    `decord` or `pyav` (OpenCV's POS_MSEC seek can land on the nearest keyframe
    on some codecs). WIRE THIS to whatever your operators already trust."""
    import cv2  # opencv-python (or swap for decord/pyav)

    def decode_fn(video_path, pts_ms):
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(pts_ms))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"decode failed: {video_path} @ {pts_ms}ms")
            return frame
        finally:
            cap.release()

    return decode_fn


# =============================================================================
# Seam 2 — Silero VAD's get_speech_timestamps (vendored, pinned, offline).
# =============================================================================
def make_get_speech_timestamps(models_dir):
    """Returns the vendored Silero `get_speech_timestamps`. The jit model itself
    is loaded inside build_real_adapters from models/silero_vad.jit; this is just
    the helper function from the same pinned silero-vad checkout."""
    # If you vendored the silero-vad repo under models/ (offline), import from it:
    #   from silero_vad import get_speech_timestamps
    # The torch.hub form below works only if the cache is pre-populated offline.
    import torch
    _, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                              trust_repo=True, source="github")
    get_speech_timestamps = utils[0]
    return get_speech_timestamps


# =============================================================================
# Seam 3 — the EEND-M2F diarization head over SHARED WavLM features.
# =============================================================================
def load_eend_head(models_dir, device):
    """WIRE THIS. Load your EEND-M2F (Mask2Former) head checkpoint from
    models/eend_m2f/ and return a callable:

        eend_forward_fn(features) -> tensor/array [B, n_speakers, n_frames] of {0,1}

    Contract:
      * `features` is the WavLM `last_hidden_state` BATCH: shape [B, F, D], on
        `device`, already computed by the shared WavLM (you do NOT run WavLM —
        that is the whole point of sharing it). B = all of one file's fixed-length
        windows; F = WavLM frames per window (50 fps ⇒ 20 ms/frame); D = 768/1024.
      * Return one binary activity mask per (window, speaker, frame). The frame
        axis must align 1:1 with F (so --frame-ms below = 20). Threshold your
        soft masks to {0,1} HERE, deterministically and conservatively (a missed
        overlap is the dangerous forensic error — see the deep-dive doc).
      * Run under torch.no_grad(); the gate has already set the determinism flags.

    Until you wire it, this raises so a run cannot silently produce garbage."""
    import torch  # noqa: F401  (your head will need it)

    # --- replace this block with your real head ---------------------------
    raise NotImplementedError(
        "load_eend_head() is the one model you must supply. Load your EEND-M2F "
        "head from models/eend_m2f/, move it to `device`, set .eval(), and return "
        "eend_forward_fn(features[B,F,D]) -> [B, n_speakers, n_frames]{0,1}. "
        "See run_ubuntu.py:load_eend_head and UBUNTU_DEPLOYMENT_RUNBOOK.md §5."
    )
    # Example skeleton once you have a head module:
    #   head = MyEendM2fHead.from_checkpoint(models_dir / "eend_m2f" / "model.pt")
    #   head = head.to(device).eval()
    #   def eend_forward_fn(features):
    #       logits = head(features)                 # [B, n_speakers, F]
    #       return (logits.sigmoid() >= 0.5).int()  # conservative, deterministic
    #   return eend_forward_fn
    # ----------------------------------------------------------------------


def make_dry_run_head():
    """A trivial, deterministic stand-in for the EEND-M2F head, for the
    `--dry-run` pre-flight. It runs over the SAME shared-WavLM feature batch the
    real head will (so this genuinely exercises the GPU path — WavLM resident in
    VRAM, the batched forward, the RAM cache — plus Layer 0 ffmpeg/Silero and
    Layer 1 YOLO/InsightFace). It declares ONE speaker, active across every frame,
    and never overlap:

        eend_forward_fn(features[B, F, D]) -> ones[B, 1, F]  (int {0,1})

    Forensically this output is meaningless (no real diarization, so no
    overlap/NaN protection is exercised in earnest) — it exists ONLY to prove the
    plumbing and measure VRAM before the real head is injected."""
    def eend_forward_fn(features):
        import torch
        b, f = int(features.shape[0]), int(features.shape[1])
        return torch.ones((b, 1, f), dtype=torch.int64, device=features.device)
    return eend_forward_fn


def _build_parser():
    p = argparse.ArgumentParser(description="WavLM+EEND-M2F forensic pipeline — "
                                            "real GPU run (Ubuntu).")
    p.add_argument("inputs", nargs="*", help="session clips, canonical order")
    p.add_argument("--clicks", help="Operator Clicks JSON")
    p.add_argument("--models-dir", help="pre-staged weights dir")
    p.add_argument("--checksums", help="checksums.json (default: <models-dir>/checksums.json)")
    p.add_argument("--out", default="session_out", help="output directory")
    p.add_argument("--device", default="cuda", help="cuda | cuda:N | cpu")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel ffmpeg decode workers (e.g. 44)")
    p.add_argument("--frame-ms", type=int, default=20,
                   help="EEND output frame period (WavLM is 50 fps ⇒ 20 ms)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--dry-run", action="store_true",
                   help="exercise the FULL GPU path with a trivial placeholder "
                        "EEND head (no custom head needed; output is NOT forensic)")
    p.add_argument("--selftest", action="store_true",
                   help="stdlib-only wiring check (no torch, no weights)")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    if not (args.inputs and args.clicks and args.models_dir):
        print("ERROR: inputs, --clicks and --models-dir are required "
              "(or use --selftest)", file=sys.stderr)
        return 2

    models_dir = Path(args.models_dir)
    checksums = Path(args.checksums) if args.checksums else models_dir / "checksums.json"
    out = Path(args.out)
    clicks_obj = json.loads(Path(args.clicks).read_text(encoding="utf-8"))

    # Build the three seams. In --dry-run the EEND head is a trivial placeholder
    # and eend_m2f is dropped from the required-weights set (you may not have a
    # head checkpoint staged yet) — every OTHER model still runs and is verified.
    decode_fn = make_decode_fn()
    get_speech_timestamps = make_get_speech_timestamps(models_dir)
    if args.dry_run:
        print("=== DRY RUN — placeholder EEND head; output is NOT forensically "
              "valid. Exercises ffmpeg/VAD/YOLO/InsightFace/shared-WavLM only. ===",
              file=sys.stderr)
        eend_forward_fn = make_dry_run_head()
        expected_models = EXPECTED_MODELS - {"eend_m2f"}
    else:
        eend_forward_fn = load_eend_head(models_dir, args.device)
        expected_models = EXPECTED_MODELS

    adapters = build_real_adapters(
        models_dir, args.inputs,
        eend_forward_fn=eend_forward_fn, decode_fn=decode_fn,
        get_speech_timestamps=get_speech_timestamps,
        sr=args.sr, eend_frame_ms=args.frame_ms, device=args.device)

    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "session.manifest.jsonl"
    with ManifestWriter(manifest_path) as mw:
        l0, l1, l2, l3 = run_session(
            inputs=args.inputs, clicks_obj=clicks_obj, manifest_writer=mw,
            adapters=adapters, models_dir=models_dir, checksums_path=checksums,
            out_dir=out / "clean", work_dir=out / "work",
            sr=args.sr, seed=args.seed, extract_workers=args.workers,
            expected_models=expected_models)

    tag = "DRY RUN OK (plumbing only)" if args.dry_run else "OK"
    print(f"{tag} — {len(l3.clean_segments)} clean target clip(s) in {out/'clean'}")
    print(f"     clean={l3.clean_ms}ms  contaminated(NaN)={l3.contaminated_ms}ms  "
          f"short-dropped={l3.discarded_short}")
    print(f"     manifest: {manifest_path}")
    print(f"     output_fingerprint: {l3.output_fingerprint}")
    return 0


# =============================================================================
# Self-test (stdlib only — verifies launcher wiring without torch/weights)
# =============================================================================

def _selftest():
    # argparse wiring: dry-run + real paths both parse.
    a = _build_parser().parse_args(["c.mp4", "--clicks", "k.json",
                                    "--models-dir", "m", "--dry-run"])
    assert a.dry_run is True and a.inputs == ["c.mp4"] and a.frame_ms == 20

    # the dry-run reduces the required-weights set by exactly eend_m2f.
    assert EXPECTED_MODELS - {"eend_m2f"} == {"wavlm", "yolov8", "insightface"}
    assert "eend_m2f" in EXPECTED_MODELS

    # the placeholder head is constructed without torch; shape logic is exercised
    # against a tiny fake feature batch (duck-typed .shape, ones via a stub).
    head = make_dry_run_head()
    assert callable(head)

    class _FakeFeats:
        shape = (3, 7, 768)                 # B=3 windows, F=7 frames, D=768
    try:
        import torch  # noqa: F401
        out = head(_FakeFeats())
        assert tuple(out.shape) == (3, 1, 7) and int(out.sum()) == 3 * 7
        torch_note = "torch present: head shape verified"
    except ImportError:
        torch_note = "torch absent: head shape logic skipped (runs on Ubuntu)"

    print(f"run_ubuntu.py self-test: OK (arg wiring, dry-run weight-set reduction, "
          f"placeholder head; {torch_note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
