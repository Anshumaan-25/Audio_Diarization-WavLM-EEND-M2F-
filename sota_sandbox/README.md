# SOTA Sandbox — Forensic Pipeline vs. neural diarizer A/B test

An **isolated** harness for comparing the in-house **Forensic Pipeline (FP)**
against a State-of-the-Art neural diarizer (PyAnnote 3.1 by default) on
`NT-clip27.mp4`.

> **Isolation guarantees.** Everything lives under `sota_sandbox/`. Nothing
> here imports from, edits, moves, or writes into the Forensic Pipeline
> (`layer0..layer3`, `session_manifest`, `click_ui`, `audit_visualizer`, …).
> Dependencies are pinned in `requirements-sota.txt` (a **separate** venv) —
> the pipeline's `requirements.txt` is never touched. The pipeline is
> **read-only** to this sandbox; the only coupling is the manifest's on-disk
> format (a `*-manifest-v1` schema), which `compare_results.py` re-parses
> read-only and sanity-checks.

## Files

| File | What it does | Deps |
|------|--------------|------|
| `run_sota.py` | Runs PyAnnote 3.1 on the clip (CUDA), writes `sota_output.json` | torch + pyannote (Ubuntu/GPU) |
| `compare_results.py` | Reads `sota_output.json` + a finished `session.manifest.jsonl`, prints a side-by-side report + ASCII timeline | **stdlib only** |
| `sanitize_manifest.py` | Writes a **redacted copy** of a manifest (strips the pipeline name) — never modifies the original | **stdlib only** |
| `run_ab.sh` | Turnkey: preflight → infer → compare | — |
| `requirements-sota.txt` | Isolated, pinned deps | — |
| `.hf_token` | Your Hugging Face token, one line (gitignored, created by you) | — |
| `TECHNICAL_REFERENCE.md` | Full deep-dive documentation | — |

## Dev vs. deploy

This is authored on the macOS dev box but **runs on the Ubuntu RTX 6000 Ada
machine** — there is no CUDA on the Mac. On the Mac you can only run the
stdlib self-tests:

```bash
python3 run_sota.py --selftest          # pure helpers, no torch/CUDA
python3 compare_results.py --selftest   # parse + interval math + render
```

## Setup (on the Ubuntu box)

```bash
sudo apt-get install -y ffmpeg                       # system dep for wav extract
cd sota_sandbox
python3.10 -m venv .venv-sota && source .venv-sota/bin/activate
pip install --upgrade pip
pip install torch==2.1.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-sota.txt
```

### Hugging Face token (gated model)

PyAnnote 3.1 is gated. Once, on huggingface.co, accept the user conditions for
**pyannote/speaker-diarization-3.1** and **pyannote/segmentation-3.0**, then
provide a read token one of these ways (checked in this order):

```bash
export HF_TOKEN=hf_xxxxxxxx          # preferred — never lands on disk in repo
# or:
printf 'hf_xxxxxxxx' > sota_sandbox/.hf_token   # gitignored
# or pass --token hf_xxxxxxxx        # least preferred (shell history)
```

## Confidentiality — redacting the manifest

The FP writes its own pipeline name into every manifest (in the `schema`
field). If that name must not appear in study artifacts, produce a redacted
copy first — this only **reads** the source and **writes a new file**, never
touching the original:

```bash
python3 sanitize_manifest.py /path/to/session.manifest.jsonl \
    -o session.clean.manifest.jsonl
# then compare against the clean copy
```

## Run the A/B test

**One command (turnkey — preflight → infer → redact manifest → compare):**

```bash
./run_ab.sh /path/to/NT-clip27.mp4 /path/to/session.manifest.jsonl
```

Pass the **raw** manifest — `run_ab.sh` writes a redacted copy itself (the
`sanitize_manifest.py` step) and compares against that, so the pipeline name
never reaches the comparison. It verifies python/ffmpeg/GPU/torch/pyannote/
token/inputs up front, runs the CUDA inference, then prints the comparison.
Override defaults via env: `DEVICE`, `OUT`, `WIDTH`, `SANITIZE` (`0` to skip
redaction).

**Or step by step:**

```bash
# 1) SOTA inference (GPU)
python3 run_sota.py --input /path/to/NT-clip27.mp4 --device cuda
#    -> writes sota_output.json

# 2) Compare against the (redacted) FP manifest
python3 compare_results.py \
    --manifest session.clean.manifest.jsonl \
    --sota sota_output.json \
    --all-speakers
```

## How to read the comparison

The FP enrolls **one** forensic subject (the target) and emits only its
**verified clean** speech as `layer3_segment` (decision `CLEAN`); overlapped or
contaminated target audio is deliberately dropped (NaN). PyAnnote labels every
speaker and keeps overlapped speech. So the comparator:

1. picks the PyAnnote speaker that **best overlaps** the FP target,
2. reports `IoU`, intersection, and the two one-sided differences:
   - **FP-only** → target speech SOTA missed or attributed to another speaker,
   - **SOTA-only** → speech SOTA kept that the FP rejected as contaminated/overlapped (expected — the FP is conservative by design),
3. draws an ASCII timeline (`FP`, `SOTA`, `DIFF` lanes) and a per-segment
   `AGREE / PARTIAL / MISS` table.

A high IoU with most divergence on the **SOTA-only** side is the *expected*
forensic signature: the two systems agree on where the target speaks, and
disagree exactly where the FP exercised its contamination veto.
