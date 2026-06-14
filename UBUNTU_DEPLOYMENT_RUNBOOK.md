# Ubuntu Deployment Runbook — WavLM + EEND‑M2F Forensic Pipeline

**Audience:** the operator *and* an AI agent standing this up on the GPU box.
**Goal:** go from a fresh Ubuntu 22.04 / RTX 6000 Ada machine to a verified,
deterministic forensic run — executed **top to bottom, in order**.

This runbook is the *procedure*. `UBUNTU_SETUP_GUIDE.md` is the deeper reference
(weights table, determinism internals, clicks schema). When in doubt, this file
wins on order; the guide wins on detail.

> **Hard rule for the agent:** do not skip the gates. The stdlib self-tests
> (Step 6) and the real smoke test (Step 8) are mandatory acceptance gates — a
> green checkmark on each is required before the run is trustworthy. If a step
> fails, stop and report; do not "work around" the determinism gate or the
> checksum verifier.

---

## What you will run

| File | Role |
|---|---|
| `run_ubuntu.py` | **The launcher.** Wires the 3 deployment seams; `--dry-run` exercises the whole GPU path with a placeholder head; `--selftest` checks its own wiring. |
| `make_checksums.py` | Generates/verifies `models/checksums.json` (no hand‑hashing). |
| `pipeline_runner.py` | Core orchestration (gate→L0→L1→L2→L3). Do not edit. |
| `gpu_runtime.py` | GPU/RAM/batching support. Do not edit. |
| `forensics/`, `layer*.py`, `environment_gate.py` | Audited core. Do not edit. |

The only file you edit is **`run_ubuntu.py`** — and only its three `WIRE THIS`
seams. The single seam that genuinely requires your model is the EEND‑M2F head
(`load_eend_head`).

---

## Step 1 — System prerequisites

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3.10 python3.10-venv build-essential
nvidia-smi        # must print the RTX 6000 Ada; CUDA 12.x driver present
```

## Step 2 — Python environment

```bash
cd "WavLM+EEND M2F"
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install opencv-python-headless        # for run_ubuntu.py's decode_fn (or decord/pyav)
```

## Step 3 — Stage the model weights (offline)

Place all weights under `models/`. Nothing downloads at run time
(`HF_HUB_OFFLINE=1` is enforced by the gate).

| Logical name | Suggested path under `models/` | Notes |
|---|---|---|
| `wavlm` | `wavlm/` (HF `WavLMModel` dir) | shared by L1 + L2 |
| `eend_m2f` | `eend_m2f/model.pt` | **your** Mask2Former head checkpoint |
| `yolov8` | `yolov8.pt` | **yolov8‑FACE**, not generic COCO |
| `insightface` | `insightface/` (e.g. `buffalo_l`) | |
| (vad) | `silero_vad.jit` | vendored at a pinned commit |

> ⚠️ If `yolov8.pt` is the stock COCO model it detects *persons*, and the
> operator's click will resolve against a body box, not a face. Stage a face
> checkpoint.

## Step 4 — Generate the checksum registry

Write a one‑time skeleton with just the paths, then let the helper hash them:

```bash
cat > skeleton.json <<'EOF'
{ "wavlm":       {"path": "wavlm/pytorch_model.bin"},
  "eend_m2f":    {"path": "eend_m2f/model.pt"},
  "yolov8":      {"path": "yolov8.pt"},
  "insightface": {"path": "insightface/det_10g.onnx"} }
EOF
python3 make_checksums.py --models-dir ./models --from-skeleton skeleton.json
python3 make_checksums.py --models-dir ./models --verify     # must print "checksums: OK"
```

(Restaged a weight later? `python3 make_checksums.py --models-dir ./models --update`.)

## Step 5 — Wire the three seams in `run_ubuntu.py`

Open `run_ubuntu.py` and complete each `WIRE THIS`:

1. **`make_decode_fn`** — exact‑PTS frame decode. OpenCV stub is provided; for
   frame‑exact PTS prefer `decord`/`pyav`. Must return the **full frame**.
2. **`make_get_speech_timestamps`** — the vendored Silero helper (import from your
   offline silero‑vad checkout).
3. **`load_eend_head`** — **the one model you must supply.** Load your EEND‑M2F
   head from `models/eend_m2f/`, move it to `device`, `.eval()`, and return:

   ```
   eend_forward_fn(features) -> [B, n_speakers, n_frames] of {0,1}
   ```

   `features` is the **shared** WavLM `last_hidden_state` batch `[B, F, D]`
   (already on GPU — you do **not** run WavLM). `B` = all of one file's
   fixed‑length windows; the frame axis must align 1:1 with `F` (WavLM is 50 fps
   ⇒ keep `--frame-ms 20`). Threshold soft masks to `{0,1}` here,
   **deterministically and conservatively** — a *missed* overlap is the dangerous
   forensic error (it lets contaminated audio through Layer 3's NaN filter).

Until `load_eend_head` is wired it raises `NotImplementedError` by design, so a
run can't silently produce garbage.

## Step 6 — ✅ Gate A: stdlib self-tests (zero installs, no GPU)

Run **before** any real data. Every module must print `OK`:

```bash
for m in forensics/pts.py forensics/determinism.py forensics/manifest.py \
         environment_gate.py layer0_preprocessor.py layer1_enrollment.py \
         layer2_tracker.py layer3_contamination.py pipeline_runner.py; do
  python3 "$m" --selftest || { echo "FAILED: $m"; exit 1; }
done
python3 gpu_runtime.py --selftest
python3 run_ubuntu.py --selftest
```

On the GPU box, also confirm CUDA determinism is satisfiable: the first real run
records a determinism self‑check to the manifest and **aborts** if the box can't
guarantee bit‑exactness.

## Step 7 — Operator Clicks JSON

```json
{
  "speaking_click": {"file_index": 0, "x": 640, "y": 360, "pts_ms": 12000},
  "anti_click":     {"file_index": 0, "x": 200, "y": 300, "pts_ms": 12000}
}
```
`speaking_click` is mandatory (anchors the target); `anti_click` optional (a face
to exclude). All times are integer PTS milliseconds. `file_index` indexes the
clips in the **canonical order you pass on the command line**.

## Step 7b — ✅ Gate B0: dry run (full GPU path, no custom head)

Before wiring your EEND head, prove the *plumbing* end‑to‑end. `--dry-run`
substitutes a trivial placeholder head that runs over the **real shared‑WavLM
feature batch**, so this genuinely exercises ffmpeg → Silero VAD → YOLOv8‑face →
InsightFace → the resident shared WavLM → batched forward → RAM cache → Layer 3
output — everything except your real diarization head. It also lets you watch
`nvidia-smi` and confirm WavLM's VRAM footprint.

`--dry-run` drops `eend_m2f` from the required‑weights set (you needn't have a
head checkpoint staged yet); every other weight is still SHA‑256‑verified. Use a
`skeleton.json` **without** the `eend_m2f` entry for this stage.

```bash
PYTHONHASHSEED=0 python3 run_ubuntu.py smoke_clip.mp4 \
    --clicks smoke_clicks.json --models-dir ./models \
    --out ./dryrun_out --device cuda --workers 4 --dry-run
# watch the GPU in another shell:  watch -n0.5 nvidia-smi
```

**Pass criteria:** exit 0; banner says *DRY RUN*; manifest chain verifies; the
L0/L1/L2 records are all present and `nvidia-smi` showed WavLM resident on the GPU
during the run. (The diarization itself is meaningless here — that's expected.
`layer2`/`layer3` records will reflect the trivial single‑speaker head.) Once this
is green, wire the real head (Step 5.3) and proceed to Gate B.

## Step 8 — ✅ Gate B: real GPU smoke test (one short clip)

Do **not** trust a full session until one clip runs clean end‑to‑end:

```bash
PYTHONHASHSEED=0 python3 run_ubuntu.py smoke_clip.mp4 \
    --clicks smoke_clicks.json --models-dir ./models \
    --out ./smoke_out --device cuda --workers 4
```

**Pass criteria (all must hold):**
- Exit code 0; prints an `output_fingerprint`.
- `smoke_out/session.manifest.jsonl` exists and its chain verifies:
  ```bash
  python3 -c "from forensics.manifest import load_manifest, verify_chain; \
  print('chain OK' if verify_chain(load_manifest('smoke_out/session.manifest.jsonl')) else 'CHAIN BROKEN')"
  ```
- The manifest contains, in order: `session_start`, `environment`,
  `model_checksums`, `layer0_file`/`layer0_vad` (per file), `layer0_summary`,
  `layer1_click`, `layer1_enrollment_windows`, `layer1_seed`, `layer2_window`(s),
  `layer2_file_summary`, `layer2_summary`, the Layer‑3 records
  (`layer3_segment`/`layer3_nan_block`/`layer3_discard_short`,
  `layer3_file_summary`), `output_hash`, `session_end`.
- **Determinism:** run it a second time into `smoke_out2/`; the `output_hash`
  payload's `output_fingerprint` must be **identical** to the first run.
- Spot‑check overlap recall: a clip you know contains crosstalk should produce
  `layer3_nan_block` records (overlap → whole‑segment discard), not clean output
  over the overlapped span.

If any of these fail, stop and read Step 10 — do not proceed to a real session.

## Step 9 — Full session run

```bash
PYTHONHASHSEED=0 python3 run_ubuntu.py \
    clip01.mp4 clip02.mp4 clip03.mp4 ... \
    --clicks operator_clicks.json --models-dir ./models \
    --out ./session_out --device cuda --workers 44
```

Outputs in `session_out/`:
- `clean/target_fNNN_<g0>-<g1>.wav` — verified uncontaminated target clips
  (integer‑PTS‑named, byte‑exact, SHA‑256'd in the manifest).
- `session.manifest.jsonl` — append‑only, hash‑chained audit log; the final
  `output_hash` record carries every clip checksum + one session fingerprint.

Verify the finished chain any time:
```bash
python3 -c "from forensics.manifest import load_manifest, verify_chain; \
print(verify_chain(load_manifest('session_out/session.manifest.jsonl')))"
```

---

## Step 10 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NotImplementedError: load_eend_head()` | EEND head not wired | Complete Step 5.3. |
| `GateError: checksum MISMATCH …` | weight changed / wrong file staged | re‑stage, then `make_checksums.py --update`. |
| `GateError: model weight missing` | path in registry ≠ on disk | fix the skeleton path; regenerate (Step 4). |
| `GateError: determinism self-check FAILED` | a nondeterministic CUDA kernel | this is intentional — the box can't guarantee bit‑exactness; resolve the kernel, don't bypass. |
| `RuntimeError: deterministic … not implemented` | same as above, raised by torch | identify the op; ensure cuDNN deterministic + `CUBLAS_WORKSPACE_CONFIG=:4096:8` (the gate sets these). |
| Click resolves to a body, not a face | generic COCO YOLO | stage **yolov8‑face** (Step 3). |
| `EnrollError: no InsightFace detection overlaps the detector box` | face/box mismatch or bad click | check the click lands on the target's face at that PTS; confirm InsightFace pack staged. |
| `EnrollError: no clean single-target enrollment windows` | target never sole face on screen | pick a click/clip where the target appears alone during speech, or relax the proxy. |
| `EnrollError: clean enrollment audio … < seed_min_ms` | <3 s of clean target audio | provide a clip with more clean target speech. |
| CUDA OOM on Layer 2 | window batch too large for 48 GB | reduce per‑file window batch in your head, or shorten `WINDOW_MS`. |
| Output not deterministic across runs | nondeterministic head/threshold | make `eend_forward_fn` thresholding deterministic; keep `PYTHONHASHSEED=0`. |
| `ffmpeg not found` | ffmpeg missing | `sudo apt-get install -y ffmpeg`. |

---

## Forensic acceptance checklist (sign‑off)

- [ ] Step 6 self‑tests all `OK` on the box.
- [ ] `make_checksums.py --verify` prints `checksums: OK` (all 4 required names).
- [ ] Smoke test exit 0; manifest chain verifies; expected records present.
- [ ] Two smoke runs ⇒ **identical** `output_fingerprint` (bit‑exact).
- [ ] A known‑crosstalk span produces `layer3_nan_block`, not clean output.
- [ ] `PYTHONHASHSEED=0` set for every real run.
- [ ] No network egress during the run (air‑gap holds; offline env enforced).
- [ ] Clean clips reviewed: integer‑PTS names, checksums match the manifest.

When all boxes are checked, the run is forensically sound and reproducible.
