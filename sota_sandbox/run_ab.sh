#!/usr/bin/env bash
# =============================================================================
# SOTA SANDBOX — run_ab.sh : turnkey A/B test (RUN ON THE UBUNTU RTX 6000 ADA)
# =============================================================================
# One command does the whole thing:
#   preflight -> PyAnnote 3.1 inference (CUDA) -> redact manifest -> comparison.
#
# Usage:
#   ./run_ab.sh /path/to/NT-clip27.mp4 /path/to/session.manifest.jsonl
#   # or rely on defaults / env:
#   INPUT=NT-clip27.mp4 MANIFEST=session.manifest.jsonl ./run_ab.sh
#
# Env: DEVICE (cuda), OUT (sota_output.json), WIDTH (120), SANITIZE (1 = redact
#      the pipeline name from a manifest copy before comparing; 0 = skip).
#
# Does NOT touch the Forensic Pipeline. Reads the manifest read-only; any
# redacted copy is a NEW file, the original is never modified.
# Requires: the .venv-sota venv active (or torch+pyannote importable), ffmpeg,
# and a Hugging Face token via $HF_TOKEN or sota_sandbox/.hf_token.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

INPUT="${1:-${INPUT:-NT-clip27.mp4}}"
MANIFEST="${2:-${MANIFEST:-session.manifest.jsonl}}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-sota_output.json}"
WIDTH="${WIDTH:-120}"

say() { printf '\033[1m[run_ab]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[run_ab] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------
say "preflight checks ..."
command -v python3 >/dev/null || die "python3 not found"
command -v ffmpeg  >/dev/null || die "ffmpeg not found — sudo apt-get install -y ffmpeg"

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU: /'
else
  say "WARNING: nvidia-smi not found — are you on the RTX 6000 Ada box? (DEVICE=$DEVICE)"
fi

python3 - <<'PY' || die "torch/pyannote not importable — activate .venv-sota and: pip install -r requirements-sota.txt (torch from the cu121 index)"
import importlib
for m in ("torch", "pyannote.audio"):
    importlib.import_module(m)
import torch
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
PY

[ -f "$INPUT" ]    || die "input clip not found: $INPUT"
[ -f "$MANIFEST" ] || die "FP manifest not found: $MANIFEST"
if [ -z "${HF_TOKEN:-}" ] && [ ! -s "$HERE/.hf_token" ]; then
  die "no HF token — export HF_TOKEN=... or put it in $HERE/.hf_token"
fi
say "preflight OK"

# ----------------------------------------------------------------------------
say "step 1/2 — SOTA inference (PyAnnote 3.1) on $INPUT [$DEVICE]"
python3 run_sota.py --input "$INPUT" --device "$DEVICE" --out "$OUT"

# Redact the pipeline name from the manifest by default (SANITIZE=0 to skip).
# Falls back to the original if there is no detectable name to remove.
CMP_MANIFEST="$MANIFEST"
if [ "${SANITIZE:-1}" = "1" ]; then
  base="$(basename "$MANIFEST")"; base="${base%.manifest.jsonl}"; base="${base%.jsonl}"
  CLEAN="${base}.clean.manifest.jsonl"
  if python3 sanitize_manifest.py "$MANIFEST" -o "$CLEAN"; then
    CMP_MANIFEST="$CLEAN"
  else
    say "no redactable name detected — comparing against the manifest as-is"
  fi
fi

say "step 2/2 — comparing against $CMP_MANIFEST"
python3 compare_results.py --manifest "$CMP_MANIFEST" --sota "$OUT" \
        --width "$WIDTH" --all-speakers

say "done — SOTA output in $OUT"
