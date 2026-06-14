"""
Forensic pipeline — pipeline_runner.py
=======================================

Purpose:    Orchestrate the whole session through ONE manifest writer:
              session_start
                → environment_gate (offline + checksums + determinism)
                → Layer 0 (preprocess: ffmpeg + VAD)
                → Layer 1 (enroll: visual anchor → WavLM seed)
                → Layer 2 (track: EEND-M2F target + overlap maps)
                → Layer 3 (contaminate: NaN-only discard + clean wavs)
              session_end (output fingerprint)

Single-writer custody: exactly one ManifestWriter threads every layer, so the
            hash chain is one continuous, verifiable record of the run.

Dependency injection: all model-backed work goes through adapters
            (VAD/Frame/Detector/Embedder/WavLM/EEND-M2F). `run_session` takes an
            `Adapters` bundle, so the SAME orchestration runs with real models
            (Ubuntu, via `build_real_adapters`) or with stubs (the stdlib
            integration self-test here). The dev box never needs torch/CUDA.

Run (Ubuntu):
    python3 pipeline_runner.py clip1.mp4 clip2.mp4 ... \
        --clicks operator_clicks.json --models-dir ./models --out ./session_out
Self-test:
    python3 pipeline_runner.py --selftest      (stdlib only — gate→L0..L3 with stubs)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from forensics.manifest import ManifestWriter
from environment_gate import run_gate
from layer0_preprocessor import preprocess
from layer1_enrollment import enroll
from layer2_tracker import track
from layer3_contamination import contaminate

EXPECTED_MODELS = frozenset({"wavlm", "eend_m2f", "yolov8", "insightface"})


@dataclass
class Adapters:
    vad: object
    frame_source: object
    detector: object
    embedder: object
    wavlm: object
    eend: object
    audio_cache: object = None          # RAM PCM store shared by the real adapters


def run_session(*, inputs, clicks_obj, manifest_writer, adapters, models_dir,
                checksums_path, out_dir, work_dir, sr=16000, seed=0,
                expected_models=EXPECTED_MODELS, extract_workers=1):
    """Drive gate → L0 → L1 → L2 → L3 on one manifest writer. Returns
    (L0, L1, L2, L3)."""
    manifest_writer.append("session_start", {
        "seed": seed, "n_inputs": len(inputs), "sample_rate": sr,
        "inputs": [str(p) for p in inputs],
    })
    run_gate(manifest_writer, models_dir=models_dir, checksums_path=checksums_path,
             seed=seed, expected_models=expected_models)

    l0 = preprocess(inputs, manifest_writer=manifest_writer, vad=adapters.vad,
                    work_dir=work_dir, sr=sr, max_workers=extract_workers)
    # Pull the whole session's 16 kHz audio into RAM once (512 GB box) so Layer 1
    # enrollment and Layer 2 per-window re-ID never re-read the wavs from disk.
    if adapters.audio_cache is not None:
        adapters.audio_cache.preload([f.wav_path for f in l0.files])
    l1 = enroll(l0, clicks_obj, manifest_writer=manifest_writer,
                frame_source=adapters.frame_source, detector=adapters.detector,
                embedder=adapters.embedder, wavlm=adapters.wavlm)
    l2 = track(l0, l1.seed, manifest_writer=manifest_writer,
               model=adapters.eend, wavlm=adapters.wavlm)
    l3 = contaminate(l0, l2, manifest_writer=manifest_writer, out_dir=out_dir)

    manifest_writer.append("session_end", {
        "output_fingerprint": l3.output_fingerprint,
        "n_outputs": len(l3.clean_segments),
        "clean_ms": l3.clean_ms,
        "contaminated_ms": l3.contaminated_ms,
        "discarded_short": l3.discarded_short,
    })
    return l0, l1, l2, l3


# =============================================================================
# Real adapters (Ubuntu GPU box). Lazy imports; weights from models_dir per
# UBUNTU_SETUP_GUIDE.md. The two genuinely project-specific seams — the
# EEND-M2F forward pass and the video frame decoder — are injected callables.
# =============================================================================

def build_real_adapters(models_dir, inputs, *, eend_forward_fn, decode_fn,
                        get_speech_timestamps, sr=16000, eend_frame_ms=20,
                        device="cuda", feature_input=True):
    """Construct model-backed, GPU-resident adapters. Runs only where torch + the
    staged weights exist.

    Key properties (the engineering pass): every model is moved to `device` (the
    RTX 6000 Ada) and so are all input tensors; WavLM is loaded ONCE and shared
    between Layer 1's seed embedder and Layer 2's EEND head (one resident copy in
    VRAM); a single `AudioCache` is shared by every audio adapter so the session's
    PCM is read from RAM, not re-read from disk.

    The operator injects two seams:
      * `eend_forward_fn(features) -> [B, n_speakers, n_frames]{0,1}` — the
        Mask2Former diarization head over the SHARED WavLM `last_hidden_state`
        batch `[B, F, D]` (set `feature_input=False` to receive a raw `[B, T]`
        wav batch instead).
      * `decode_fn(video_path, pts_ms) -> frame` — exact-PTS frame decode."""
    import torch
    from layer0_preprocessor import SileroVad
    from layer1_enrollment import (VideoFrameSource, YoloFaceDetector,
                                    InsightFaceEmbedder, WavLMSeedEmbedder)
    from layer2_tracker import WavLMEendM2f
    from gpu_runtime import AudioCache, pick_device, device_index

    device = pick_device(device)
    ctx_id = device_index(device)
    models_dir = Path(models_dir)
    audio_cache = AudioCache(sr=sr)

    # Silero VAD (CPU is fine and cheap, but honour the chosen device).
    silero = torch.jit.load(str(models_dir / "silero_vad.jit"), map_location=device)
    silero.eval()
    vad = SileroVad(silero, get_speech_timestamps)

    # ONE WavLM, shared by the seed embedder and the EEND head.
    from transformers import WavLMModel
    wavlm_model = WavLMModel.from_pretrained(str(models_dir / "wavlm"),
                                             local_files_only=True).to(device).eval()
    wavlm = WavLMSeedEmbedder(wavlm_model, sr=sr, device=device,
                              audio_cache=audio_cache)

    from ultralytics import YOLO
    detector = YoloFaceDetector(YOLO(str(models_dir / "yolov8.pt")), device=device)

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", root=str(models_dir / "insightface"))
    app.prepare(ctx_id=ctx_id)
    embedder = InsightFaceEmbedder(app)

    frame_source = VideoFrameSource({i: p for i, p in enumerate(inputs)}, decode_fn)
    eend = WavLMEendM2f(eend_forward_fn, frame_ms=eend_frame_ms, sr=sr,
                        device=device, wavlm_model=wavlm_model,
                        audio_cache=audio_cache, feature_input=feature_input)
    return Adapters(vad, frame_source, detector, embedder, wavlm, eend,
                    audio_cache=audio_cache)


def main(argv=None):
    p = argparse.ArgumentParser(description="WavLM+EEND-M2F forensic diarization "
                                            "pipeline (gate→L0→L1→L2→L3).")
    p.add_argument("inputs", nargs="*", help="session clips in canonical order")
    p.add_argument("--clicks", help="Operator Clicks JSON")
    p.add_argument("--models-dir", help="directory of pre-staged weights")
    p.add_argument("--checksums", help="checksums.json (default: <models-dir>/checksums.json)")
    p.add_argument("--out", default="session_out", help="output directory")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true",
                   help="stdlib-only integration self-test (stubs, no torch)")
    args = p.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not (args.inputs and args.clicks and args.models_dir):
        print("ERROR: inputs, --clicks and --models-dir are required (or --selftest)",
              file=sys.stderr)
        return 2

    # Real run (Ubuntu). The operator supplies the EEND-M2F forward + decoder.
    print("ERROR: real-model run must be launched from the Ubuntu wiring script "
          "that injects eend_forward_fn/decode_fn into build_real_adapters() — "
          "see UBUNTU_SETUP_GUIDE.md.", file=sys.stderr)
    return 3


# =============================================================================
# Integration self-test (stdlib only — gate → L0 → L1 → L2 → L3 with stubs)
# =============================================================================

def _selftest():
    import tempfile
    import wave
    from forensics.manifest import sha256_file, load_manifest, verify_chain
    from layer0_preprocessor import StubVad
    from layer1_enrollment import (StubFrameSource, StubDetector, StubEmbedder,
                                    StubWavLM)
    from layer2_tracker import StubEend, SpeakerActivity

    with tempfile.TemporaryDirectory() as dd:
        d = Path(dd)
        src = d / "clip0.wav"
        with wave.open(str(src), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x01\x00" * (16000 * 6))         # 6000 ms

        # synthetic staged weights + checksum registry
        models = d / "models"
        (models / "m").mkdir(parents=True)
        registry = {}
        for name in EXPECTED_MODELS:
            wp = models / "m" / f"{name}.bin"
            wp.write_bytes(name.encode())
            registry[name] = {"path": f"m/{name}.bin", "sha256": sha256_file(wp)}
        checks = d / "checksums.json"
        checks.write_text(json.dumps(registry))

        clicks = {"speaking_click": {"file_index": 0, "x": 150, "y": 150, "pts_ms": 1000}}

        TARGET, OTHER = (120, 120, 180, 180, 0.9), (300, 100, 400, 200, 0.8)

        def detect_fn(frame):
            fi, pts = frame
            if pts == 1000:
                return [(100, 100, 300, 300, 0.7), TARGET]   # click frame
            if 5000 <= pts < 6000:
                return [TARGET, OTHER]                        # visually contaminated
            return [TARGET]

        def embed_fn(frame, box):
            return [1.0, 0.0, 0.0] if box[:4] == TARGET[:4] else [0.0, 1.0, 0.0]

        def wavlm_fn(wav, a, b):
            return [1.0, 0.0, 0.0] if a in (0, 3000) else [0.0, 1.0, 0.0]

        n = 300                                              # 6000ms / 20ms
        s0 = [1 if (0 <= f < 100 or 150 <= f < 250) else 0 for f in range(n)]
        s1 = [1 if 175 <= f < 225 else 0 for f in range(n)]
        act = SpeakerActivity(frame_ms=20, activity=[s0, s1])

        adapters = Adapters(
            vad=StubVad([[(0, 64000), (80000, 88000)]]),     # speech (0,4000),(5000,5500)
            frame_source=StubFrameSource(),
            detector=StubDetector(detect_fn),
            embedder=StubEmbedder(embed_fn),
            wavlm=StubWavLM(wavlm_fn),
            eend=StubEend(lambda wav, ws, we, sr: act))

        out = d / "out"
        mpath = out / "session.manifest.jsonl"
        with ManifestWriter(mpath, fsync=False) as mw:
            l0, l1, l2, l3 = run_session(
                inputs=[src], clicks_obj=clicks, manifest_writer=mw,
                adapters=adapters, models_dir=models, checksums_path=checks,
                out_dir=out / "clean", work_dir=out / "work", sr=16000, seed=0)

        assert len(l1.seed) == 3
        assert l2.target_active == [(0, 2000), (3000, 5000)], l2.target_active
        assert l2.overlap == [(3500, 4500)], l2.overlap
        # (0,2000) clean -> trimmed clip; (3000,5000) touches overlap -> NaN
        assert len(l3.clean_segments) == 1, l3.clean_segments
        assert Path(l3.clean_segments[0]["wav_path"]).is_file()

        entries = load_manifest(mpath)
        assert verify_chain(entries) is True
        ops = [e["operation"] for e in entries]
        for required in ("session_start", "environment", "model_checksums",
                         "layer0_file", "layer1_seed", "layer2_summary",
                         "layer3_segment", "output_hash", "session_end"):
            assert required in ops, (required, ops)
        assert entries[-1]["operation"] == "session_end"
        assert entries[-1]["payload"]["output_fingerprint"] == l3.output_fingerprint

    print("pipeline_runner.py self-test: OK (gate→L0→L1→L2→L3 on one manifest "
          "chain; deterministic output fingerprint)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
