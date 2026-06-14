"""
SOTA SANDBOX — run_sota.py  (ISOLATED A/B inference harness)
============================================================

Purpose:    Run an end-to-end State-of-the-Art neural diarizer
            (PyAnnote 3.1 by default) over a single clip — NT-clip27.mp4
            or its 16 kHz extracted wav — on the CUDA device, and write
            the resulting speaker turns to a clean, simple JSON file
            (``sota_output.json``): a list of {start_ms, end_ms,
            speaker_id} plus a small provenance header. This JSON is the
            SOTA side of the A/B test; ``compare_results.py`` lines it up
            against the Forensic Pipeline (FP) timeline.

Isolation contract (the whole point of this sandbox):
            This file lives ONLY under sota_sandbox/. It imports NOTHING
            from the Forensic Pipeline (no layer0..layer3, no
            session_manifest, no environment_gate) and writes NOTHING
            outside this directory. Heavy third-party deps (torch,
            pyannote.audio) come from sota_sandbox/requirements-sota.txt
            in a SEPARATE virtualenv — never the pipeline's
            requirements.txt. The FP tree is read-only to us; we
            never touch it.

Dev/deploy split:
            This is authored on the macOS dev box but EXECUTES on the
            Ubuntu RTX 6000 Ada machine (CUDA, FP32/FP16, plenty of
            VRAM). The heavy imports (torch, pyannote.audio) are LAZY —
            done inside run(), never at module import — so ``--help`` and
            ``--selftest`` work on plain Python 3.10 with zero pip
            installs, mirroring the pipeline's stdlib-self-test rule.

Token:      PyAnnote 3.1 is a gated model and needs a Hugging Face token.
            Resolution order (first hit wins):
              1. --token VALUE
              2. $HF_TOKEN  or  $HUGGINGFACE_TOKEN
              3. sota_sandbox/.hf_token   (gitignored; one line)
              4. interactive getpass prompt (only if a TTY is attached)
            The token is never printed, never written to sota_output.json,
            never committed.

Determinism:
            All RNGs are seeded and deterministic algorithms requested.
            PyAnnote's agglomerative clustering can still introduce minor
            run-to-run jitter on some CUDA kernels; the seed and library
            versions are recorded in the output header so a run is
            reproducible/attributable. This sandbox makes NO claim to
            the FP's bit-exact forensic determinism — it is the
            comparison baseline, not the system of record.

Run:        python3 run_sota.py --input /path/NT-clip27.mp4 [--device auto]
                [--model pyannote/speaker-diarization-3.1]
                [--out sota_output.json] [--token TOKEN] [--seed 0]
Self-test:  python3 run_sota.py --selftest      (stdlib only — no torch,
            no CUDA, no network: exercises the pure helpers)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SANDBOX_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SANDBOX_DIR / "sota_output.json"
DEFAULT_TOKEN_FILE = SANDBOX_DIR / ".hf_token"
WORK_DIR = SANDBOX_DIR / "work"               # gitignored scratch (extracted wavs)
DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
TARGET_SR = 16000
OUTPUT_SCHEMA = "sota-sandbox-output-v1"


# =============================================================================
# Pure helpers (covered by --selftest; no torch, no network, no CUDA)
# =============================================================================

def resolve_token(cli_token, token_file=DEFAULT_TOKEN_FILE, environ=None,
                  allow_prompt=True):
    """Resolve the Hugging Face token by priority. Returns the token string.

    Order: --token > $HF_TOKEN/$HUGGINGFACE_TOKEN > .hf_token file >
    interactive getpass. Raises RuntimeError if nothing is found and no
    prompt is possible. The value is returned, never logged."""
    environ = os.environ if environ is None else environ
    if cli_token:
        return cli_token.strip()
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = environ.get(var)
        if val and val.strip():
            return val.strip()
    token_file = Path(token_file)
    if token_file.is_file():
        text = token_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    if allow_prompt and sys.stdin and sys.stdin.isatty():
        import getpass
        entered = getpass.getpass("Hugging Face token (input hidden): ").strip()
        if entered:
            return entered
    raise RuntimeError(
        "No Hugging Face token found. Provide --token, set $HF_TOKEN, or "
        f"write it to {token_file} (one line). The pyannote/speaker-"
        "diarization-3.1 model is gated — accept its terms on huggingface.co "
        "and use a token with read access."
    )


def ms_from_seconds(seconds):
    """PyAnnote yields float seconds; the FP speaks integer PTS ms.
    Convert with round-half-to-even applied via int(round())."""
    return int(round(float(seconds) * 1000.0))


def segment_from_turn(start_s, end_s, speaker_id):
    """One diarization turn -> the clean output record shape."""
    return {
        "start_ms": ms_from_seconds(start_s),
        "end_ms": ms_from_seconds(end_s),
        "speaker_id": str(speaker_id),
    }


def build_ffmpeg_cmd(src, dst, sr=TARGET_SR):
    """Deterministic 16 kHz mono PCM extraction command (no re-dither,
    fixed resampler). Audio only (-vn); overwrite (-y)."""
    return [
        "ffmpeg", "-nostdin", "-y",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(dst),
    ]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_wav_meta(path):
    """Actual (sample_rate, channels) of a wav via the stdlib `wave` module,
    or (None, None) if it cannot be read (e.g. a float/extensible-format wav).
    Used so the recorded provenance reflects the ACTUAL audio scored — not an
    assumed 16 kHz — when a pass-through wav is supplied."""
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate(), w.getnchannels()
    except Exception:
        return None, None


def assemble_output(segments, *, source, wav_path, wav_sha256, model, device,
                    seed, sample_rate=TARGET_SR, library_versions=None):
    """Fold turns + provenance into the on-disk JSON object. Segments are
    sorted by (start_ms, end_ms, speaker_id) for stable, diffable output."""
    ordered = sorted(
        segments, key=lambda s: (s["start_ms"], s["end_ms"], s["speaker_id"])
    )
    speakers = sorted({s["speaker_id"] for s in ordered})
    return {
        "schema": OUTPUT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(source),
        "wav_path": str(wav_path) if wav_path is not None else None,
        "wav_sha256": wav_sha256,
        "sample_rate": sample_rate,
        "model": model,
        "device": device,
        "seed": seed,
        "library_versions": library_versions or {},
        "num_speakers": len(speakers),
        "speakers": speakers,
        "num_segments": len(ordered),
        "total_speech_ms": sum(s["end_ms"] - s["start_ms"] for s in ordered),
        "segments": ordered,
    }


# =============================================================================
# Heavy path (lazy imports — only runs on the Ubuntu GPU box)
# =============================================================================

def ensure_wav(input_path, *, log):
    """Return a path to a 16 kHz mono wav. If the input already is a wav we
    use it as-is (we do NOT re-encode — that would perturb the bytes);
    otherwise we extract one into work/ with ffmpeg."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input not found: {input_path}")
    if input_path.suffix.lower() == ".wav":
        log(f"  input is already a wav — using as-is: {input_path}")
        return input_path
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dst = WORK_DIR / (input_path.stem + f".{TARGET_SR // 1000}k.wav")
    cmd = build_ffmpeg_cmd(input_path, dst)
    log(f"  extracting 16 kHz mono wav -> {dst}")
    log("  $ " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH — install it "
                           "(apt-get install ffmpeg) on the Ubuntu box.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "ffmpeg failed:\n" + exc.stderr.decode("utf-8", "replace")[-2000:]
        )
    return dst


def pick_device(requested, *, log):
    """Resolve auto/cuda/cpu against what torch actually sees."""
    import torch
    if requested == "cpu":
        return "cpu"
    cuda_ok = torch.cuda.is_available()
    if requested == "cuda":
        if not cuda_ok:
            raise RuntimeError(
                "--device cuda requested but torch.cuda.is_available() is "
                "False. This sandbox is meant to run on the Ubuntu RTX 6000 "
                "Ada box. (On the macOS dev machine there is no CUDA — use "
                "--selftest here, run the real job on Ubuntu.)"
            )
        log(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        return "cuda"
    # auto
    if cuda_ok:
        log(f"  CUDA available -> using cuda ({torch.cuda.get_device_name(0)})")
        return "cuda"
    log("  CUDA NOT available -> falling back to cpu (expected only off-box)")
    return "cpu"


def seed_everything(seed, *, log):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # older torch
        pass
    log(f"  seeded all RNGs with {seed} (deterministic algorithms requested)")


def library_versions():
    out = {}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    try:
        import pyannote.audio as pa
        out["pyannote.audio"] = pa.__version__
    except Exception:
        pass
    return out


def run_diarization(wav_path, *, token, device, model_name, seed, log):
    """Load the gated pipeline, move it to the device, run it, and return a
    list of {start_ms, end_ms, speaker_id}."""
    import torch
    from pyannote.audio import Pipeline

    log(f"  loading pipeline: {model_name}")
    try:
        # pyannote 3.1 / huggingface_hub <0.21 use `use_auth_token`.
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
    except TypeError:
        # newer huggingface_hub renamed the kwarg to `token` — stay compatible.
        pipeline = Pipeline.from_pretrained(model_name, token=token)
    if pipeline is None:
        raise RuntimeError(
            f"Pipeline.from_pretrained({model_name!r}) returned None — this "
            "almost always means the token is missing/invalid or you have not "
            "accepted the model's user conditions on huggingface.co."
        )
    pipeline.to(torch.device(device))
    seed_everything(seed, log=log)

    log("  running diarization ...")
    diarization = pipeline(str(wav_path))

    segments = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        segments.append(segment_from_turn(turn.start, turn.end, speaker))
    log(f"  diarization produced {len(segments)} turns across "
        f"{len({s['speaker_id'] for s in segments})} speakers")
    return segments


def run(args, *, log=print):
    log("SOTA sandbox — inference run")
    token = resolve_token(args.token)             # raises if absent
    wav = ensure_wav(args.input, log=log)
    device = pick_device(args.device, log=log)
    segments = run_diarization(
        wav, token=token, device=device, model_name=args.model,
        seed=args.seed, log=log,
    )
    wav_sha = sha256_file(wav)
    sr, ch = read_wav_meta(wav)
    if sr is None:
        sr = TARGET_SR                    # unreadable header — fall back to assumed rate
    elif sr != TARGET_SR or ch != 1:
        log(f"  WARNING: wav is {sr} Hz / {ch}ch (expected {TARGET_SR} Hz mono); "
            "PyAnnote resamples internally, but provenance records the actual file")
    output = assemble_output(
        segments, source=args.input, wav_path=wav, wav_sha256=wav_sha,
        model=args.model, device=device, seed=args.seed,
        sample_rate=sr, library_versions=library_versions(),
    )
    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    log(f"  wrote {output['num_segments']} segments -> {out_path}")
    log(f"  speakers: {', '.join(output['speakers']) or '(none)'}")
    log(f"  total speech: {output['total_speech_ms'] / 1000:.1f}s")
    log("Done. Next: python3 compare_results.py --sota "
        f"{out_path.name} --manifest <session.manifest.jsonl>")
    return output


# =============================================================================
# Self-test (stdlib only)
# =============================================================================

def _selftest():
    import tempfile
    # ms conversion: round-half-to-even via int(round())
    assert ms_from_seconds(1.2345) == 1234, ms_from_seconds(1.2345)
    assert ms_from_seconds(0) == 0
    assert ms_from_seconds(12.000) == 12000

    rec = segment_from_turn(1.5, 2.0, "SPEAKER_01")
    assert rec == {"start_ms": 1500, "end_ms": 2000,
                   "speaker_id": "SPEAKER_01"}, rec

    cmd = build_ffmpeg_cmd("a.mp4", "b.wav", sr=16000)
    assert cmd[0] == "ffmpeg" and "16000" in cmd and "-vn" in cmd, cmd

    # token resolution priority
    assert resolve_token("  TT  ", allow_prompt=False) == "TT"
    assert resolve_token(None, environ={"HF_TOKEN": "envtok"},
                         allow_prompt=False) == "envtok"
    with tempfile.TemporaryDirectory() as d:
        tf = Path(d) / ".hf_token"
        tf.write_text("filetok\n")
        assert resolve_token(None, token_file=tf, environ={},
                             allow_prompt=False) == "filetok"
        missing = Path(d) / "nope"
        try:
            resolve_token(None, token_file=missing, environ={},
                          allow_prompt=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError when no token present")

    # output assembly: sorting, dedup of speakers, totals, no token leakage
    segs = [
        {"start_ms": 5000, "end_ms": 6000, "speaker_id": "SPEAKER_01"},
        {"start_ms": 0, "end_ms": 2000, "speaker_id": "SPEAKER_00"},
        {"start_ms": 2000, "end_ms": 3000, "speaker_id": "SPEAKER_00"},
    ]
    out = assemble_output(segs, source="NT-clip27.mp4", wav_path="x.wav",
                          wav_sha256="deadbeef", model=DEFAULT_MODEL,
                          device="cuda", seed=0)
    assert out["segments"][0]["start_ms"] == 0, "not sorted"
    assert out["num_speakers"] == 2 and out["speakers"] == [
        "SPEAKER_00", "SPEAKER_01"]
    assert out["total_speech_ms"] == 2000 + 1000 + 1000
    blob = json.dumps(out)
    assert "envtok" not in blob and "filetok" not in blob
    assert out["schema"] == OUTPUT_SCHEMA

    # read_wav_meta on a tiny stdlib-written wav (and on a missing file)
    import wave
    with tempfile.TemporaryDirectory() as d:
        wp = Path(d) / "t.wav"
        with wave.open(str(wp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 100)
        assert read_wav_meta(wp) == (16000, 1), read_wav_meta(wp)
    assert read_wav_meta("/nonexistent/x.wav") == (None, None)

    print("run_sota.py self-test: OK (pure helpers verified, stdlib only)")
    return 0


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="SOTA-sandbox neural diarization (PyAnnote 3.1) — "
                    "isolated A/B baseline against the Forensic Pipeline.")
    p.add_argument("--input", default="NT-clip27.mp4",
                   help="path to the clip (.mp4) or a 16 kHz wav "
                        "(default: NT-clip27.mp4)")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT),
                   help="output JSON path (default: sota_sandbox/sota_output.json)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HF pipeline id (default: %(default)s)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="inference device (default: auto -> cuda if present)")
    p.add_argument("--token", default=None,
                   help="HF token (prefer $HF_TOKEN or sota_sandbox/.hf_token)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    p.add_argument("--selftest", action="store_true",
                   help="run stdlib-only self-test and exit (no torch/CUDA)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    try:
        run(args)
    except Exception as exc:  # surface a clean message, not a stacktrace wall
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
