# Audio Diarization — WavLM + EEND‑M2F (Forensic Pipeline)

A forensic speaker‑diarization pipeline that anchors a target speaker from an
operator click + visual identity, tracks them with an EEND‑M2F head over **shared
WavLM** features, and emits **only verified, uncontaminated** target‑speech clips.

## Forensic guarantees
- **Uncontaminated target speech only** — NaN‑only overlap policy: any target
  segment touching ≥2‑speaker overlap is discarded *whole* (never separated /
  reconstructed), with a conservative inward edge‑trim.
- **Offline / air‑gapped** — every model weight is SHA‑256‑verified at a pre‑flight
  gate; nothing downloads at run time.
- **Bit‑exact determinism** — fixed seeds, deterministic CUDA algorithms,
  fixed‑shape windows; re‑runs reproduce an identical output fingerprint.
- **Tamper‑evident custody** — an append‑only, SHA‑256 hash‑chained
  `session.manifest.jsonl` records every decision (content hash + wall‑clock
  custody hash).

## Layout
| Path | Role |
|---|---|
| `forensics/` | time (`pts`), determinism, hash‑chained `manifest` |
| `environment_gate.py` | offline + determinism + weight‑checksum pre‑flight |
| `layer0_preprocessor.py` | ffmpeg 16 kHz mono + Silero VAD |
| `layer1_enrollment.py` | click → YOLOv8‑face + InsightFace → WavLM seed |
| `layer2_tracker.py` | EEND‑M2F over WavLM, per‑window target re‑ID + overlap map |
| `layer3_contamination.py` | NaN‑only discard + edge‑trim + clean clips |
| `pipeline_runner.py` | orchestration + real GPU adapters |
| `gpu_runtime.py` | RAM audio cache, batching, device helpers |
| `run_ubuntu.py` | **the launcher** (`--dry-run`, `--selftest`) |
| `make_checksums.py` | generate/verify the weight registry |

## Run it
The pipeline is authored on macOS (stdlib‑only self‑tests, no CUDA) and executes
on Ubuntu 22.04 + an NVIDIA RTX 6000 Ada. Follow
**[`UBUNTU_DEPLOYMENT_RUNBOOK.md`](UBUNTU_DEPLOYMENT_RUNBOOK.md)** end‑to‑end;
**[`UBUNTU_SETUP_GUIDE.md`](UBUNTU_SETUP_GUIDE.md)** is the deeper reference and
**[`WAVLM_EEND_TECHNICAL_DEEP_DIVE.md`](WAVLM_EEND_TECHNICAL_DEEP_DIVE.md)** is the
theory/architecture writeup.

Sanity‑check on any box (no installs, no GPU):
```bash
for m in forensics/pts.py forensics/determinism.py forensics/manifest.py \
         environment_gate.py layer0_preprocessor.py layer1_enrollment.py \
         layer2_tracker.py layer3_contamination.py pipeline_runner.py; do
  python3 "$m" --selftest || exit 1
done
python3 gpu_runtime.py --selftest && python3 run_ubuntu.py --selftest
```

> Model weights, audio, and session outputs are never committed (see
> `.gitignore`). You stage the weights and supply your EEND‑M2F head per the
> runbook.
