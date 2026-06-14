# Implementation Plan — WavLM + EEND‑M2F Forensic Pipeline

The single source of truth for **what we're building, what's done, and what's left
to implement** — for the operator and the Ubuntu AI agent. Procedure detail lives
in [`UBUNTU_DEPLOYMENT_RUNBOOK.md`](UBUNTU_DEPLOYMENT_RUNBOOK.md); theory lives in
[`WAVLM_EEND_TECHNICAL_DEEP_DIVE.md`](WAVLM_EEND_TECHNICAL_DEEP_DIVE.md). This file
is the map that points at both.

---

## 1. What we are building

A forensic speaker‑diarization pipeline that takes a session of interview clips +
an operator click (target identity), and emits **only verified, uncontaminated**
target‑speech `.wav` clips plus a tamper‑evident audit manifest.

**Four guarantees (non‑negotiable):**
1. Uncontaminated target speech — NaN‑only overlap discard (never separate audio).
2. Offline / air‑gapped — SHA‑256‑verified weights, no run‑time downloads.
3. Bit‑exact determinism — reproducible output fingerprint.
4. Hash‑chained custody — append‑only `session.manifest.jsonl`.

**Flow:** Environment Gate → L0 ffmpeg+VAD → L1 click→face→WavLM seed → L2 EEND‑M2F
tracking + overlap map → L3 NaN discard + edge‑trim + clean clips.

---

## 2. Status at a glance

| Area | State |
|---|---|
| Core layers (L0–L3), gate, manifest | ✅ Done, self‑tested |
| Determinism / custody substrate (`forensics/`) | ✅ Done, self‑tested |
| GPU engineering (shared WavLM, device wiring, batching, RAM cache, parallel ffmpeg) | ✅ Done |
| Integration fixes (InsightFace IoU, short‑slice padding, yolov8‑face) | ✅ Done |
| Launcher `run_ubuntu.py` (`--dry-run`, `--selftest`) | ✅ Done |
| Checksum tooling `make_checksums.py` | ✅ Done |
| Docs (runbook, setup guide, deep dive, this plan) | ✅ Done |
| A/B sandbox (`sota_sandbox/`, PyAnnote 3.1 comparator) | ✅ Present, optional validation |
| **Your EEND‑M2F head wired in `load_eend_head`** | ⏳ **You supply** |
| **Real GPU smoke test on the box** | ⏳ **Pending hardware** |

11/11 stdlib self‑tests pass on macOS (no torch/GPU). The two ⏳ items can only be
done on the Ubuntu RTX 6000 Ada box — they are the remaining implementation work.

---

## 3. What's left to implement (and how)

### 3.1 Wire your EEND‑M2F head — the one custom model
In `run_ubuntu.py → load_eend_head()`, load your Mask2Former diarization head from
`models/eend_m2f/` and return:

```
eend_forward_fn(features) -> [B, n_speakers, n_frames] of {0,1}
```

- `features` is the **shared** WavLM `last_hidden_state` batch `[B, F, D]`, already
  on GPU (you do **not** run WavLM — it's shared with Layer 1).
- Frame axis aligns 1:1 with `F` (WavLM 50 fps ⇒ keep `--frame-ms 20`).
- Threshold soft masks to `{0,1}` **deterministically and conservatively** — a
  *missed* overlap is the dangerous error (it lets contaminated audio through L3).

Until wired, it raises `NotImplementedError` by design.

### 3.2 Prove it on the GPU box
Follow the runbook gates in order; each is a hard acceptance gate:
- **Gate A** — stdlib self‑tests (no installs/GPU).
- **Gate B0** — `--dry-run`: full GPU path with a placeholder head (no custom head
  needed). Confirms ffmpeg/VAD/YOLO/InsightFace/shared‑WavLM/batching + VRAM.
- **Gate B** — real smoke test on one clip after 3.1 is wired; verify manifest
  chain + identical `output_fingerprint` across two runs.
- **Full run** — the session, `--workers 44`.

### 3.3 (Optional) Validate quality with the A/B sandbox
`sota_sandbox/` runs PyAnnote 3.1 on the same input and compares against the
pipeline's output manifest — independent evidence the pipeline behaves sanely. See
`sota_sandbox/README.md`. Separate install (`requirements-sota.txt`) and its own
local `.hf_token` (never committed).

---

## 4. Order of operations (the path to a working demo)

```
clone repo
  └─ Gate A: run all --selftest            (any machine, no GPU)
        └─ stage weights + make_checksums   (Ubuntu)
              └─ Gate B0: run_ubuntu.py --dry-run   (proves GPU plumbing + VRAM)
                    └─ wire load_eend_head           (§3.1 — your model)
                          └─ Gate B: real smoke test (one clip, 2× for determinism)
                                └─ full session run   (--workers 44)
                                      └─ optional: sota_sandbox A/B compare
```

---

## 5. Hard constraints (do not regress)

- Inputs/outputs/guarantees stay identical to the forensic contract above.
- Adapter seams stay intact; the stdlib self‑tests must stay green.
- Never commit weights, audio, session outputs, or `.hf_token`.
- Determinism is a gate, not a nice‑to‑have — never bypass a failing
  determinism/checksum check; fix the cause.

---

## 6. Secrets — Hugging Face token

The A/B sandbox needs an HF token to pull gated/reference models. It is **local
only**, read from `sota_sandbox/.hf_token`, and is gitignored — it must never be
committed or pasted into chat/docs.

**Rotate before first use** (a token was exposed in early planning chat → treat it
as compromised):
1. https://huggingface.co/settings/tokens → **Revoke** the old token →
   **New token** (a *read* scope is sufficient for pulling weights).
2. Save it locally (stays gitignored, no code change needed):
   ```bash
   printf '%s' 'hf_YOUR_NEW_TOKEN' > sota_sandbox/.hf_token
   ```
3. Confirm it never enters git: `git status` must not list `.hf_token`.

The core pipeline (Layers 0–3) does **not** need any token — weights are
pre‑staged and SHA‑256‑verified offline. The token is only for the optional
`sota_sandbox/` comparator.
