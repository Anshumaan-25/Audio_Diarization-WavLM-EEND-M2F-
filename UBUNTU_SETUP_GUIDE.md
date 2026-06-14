# Ubuntu Setup Guide — WavLM + EEND‑M2F Forensic Diarization Pipeline

The single reference for standing up the GPU workbench. Code is **authored on
macOS** (no CUDA) but **executes here**: Ubuntu 22.04, NVIDIA RTX 6000 Ada,
CUDA 12.x, Python 3.10. Keep this file updated whenever a dependency, model, or
system requirement changes.

> Forensic guarantees this pipeline must uphold, unchanged: **uncontaminated
> target speech only** (NaN‑only overlap policy), **offline/air‑gapped with
> SHA‑256‑verified weights**, **bit‑exact reproducibility**, and the
> **hash‑chained `session.manifest.jsonl`** audit format.

## 1. System prerequisites

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3.10 python3.10-venv build-essential
# NVIDIA driver + CUDA 12.x toolkit per your standard image (nvidia-smi must work)
```

## 2. Python environment

```bash
cd "WavLM+EEND M2F"
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# 1) CUDA torch FIRST (cu121 wheels):
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# 2) everything else:
pip install -r requirements.txt
```

## 3. Stage the model weights (offline)

All weights are pre‑staged locally and verified by SHA‑256 at the gate — nothing
is downloaded at run time (`HF_HUB_OFFLINE=1` is enforced). Place them under
`models/` and record a `models/checksums.json` registry.

| Logical name (required) | Suggested path | What it is |
|---|---|---|
| `wavlm` | `models/wavlm/` | WavLM model dir (HF `WavLMModel`, `local_files_only`) |
| `eend_m2f` | `models/eend_m2f/…` | EEND‑M2F diarization head checkpoint (vendored/custom) |
| `yolov8` | `models/yolov8.pt` | YOLOv8 face/person weights |
| `insightface` | `models/insightface/` | InsightFace pack (e.g. `buffalo_l`) |
| (vad) | `models/silero_vad.jit` | Silero VAD jit (vendored at a pinned commit) |

**`models/checksums.json`** (the gate verifies every entry; any mismatch/missing
file halts the run):

```json
{
  "wavlm":       {"path": "wavlm/pytorch_model.bin", "sha256": "<hex>"},
  "eend_m2f":    {"path": "eend_m2f/model.pt",        "sha256": "<hex>"},
  "yolov8":      {"path": "yolov8.pt",                "sha256": "<hex>"},
  "insightface": {"path": "insightface/det_10g.onnx", "sha256": "<hex>"}
}
```

Compute checksums with:

```bash
sha256sum models/yolov8.pt        # etc. — paste the hex into checksums.json
```

## 4. Wire the two deployment‑specific seams

Everything model‑backed runs through adapters. Two seams are genuinely specific
to your deployment and are injected into `build_real_adapters()` in
`pipeline_runner.py`:

- **`eend_forward_fn(features) -> [B][n_speakers][n_frames] of {0,1}`** — your
  EEND‑M2F (Mask2Former) diarization head. WavLM is loaded **once** by
  `build_real_adapters` and shared, so the head receives the WavLM
  `last_hidden_state` **batch** `[B, F, D]` (already on the GPU) and returns a
  per‑window batch of speaker‑activity masks. All of a file's fixed‑shape windows
  arrive in one batch `B`, so the head runs once per file. Must run under the
  gate's determinism flags and emit a fixed frame rate (`eend_frame_ms`, e.g.
  20 ms). *If your head prefers raw audio, pass `feature_input=False` to
  `build_real_adapters` and it will instead receive a raw `[B, T]` wav batch.*
- **`decode_fn(video_path, pts_ms) -> frame`** — exact‑PTS video frame decode
  (ffmpeg/decord/opencv) for Layer 1 visual anchoring. Return the **full frame**
  (InsightFace re‑detects on it and associates to the YOLO box by IoU — do not
  pre‑crop).
- Also provide the vendored Silero **`get_speech_timestamps`**.

Write a small launcher that imports these, calls `build_real_adapters(...)`, then
`run_session(...)` with a `ManifestWriter` pointed at `session_out/session.manifest.jsonl`.

### Hardware utilisation (RTX 6000 Ada · 44 cores · 512 GB RAM)

`build_real_adapters(..., device="cuda")` resolves the GPU and moves **every**
model and input tensor onto it (WavLM, YOLOv8‑face, InsightFace, Silero). The
engineering pass added, all behind the existing seams:

- **Shared WavLM** — one resident 300 M‑param encoder feeds both Layer 1
  enrollment and Layer 2 tracking (was loaded twice).
- **Batched GPU passes** — Layer 2 runs all of a file's fixed‑length windows
  through WavLM+EEND in one forward; per‑window speaker re‑ID embeds every
  candidate speaker's slices in one WavLM call. Fixed window shape ⇒ clean
  batching; size it to the 48 GB budget via the head.
- **RAM audio cache** — the session's 16 kHz PCM is pulled into RAM once
  (`AudioCache.preload`) and every slice is served from memory; no wav is
  re‑read from disk.
- **Parallel decode** — pass `extract_workers=N` (e.g. 44) to `run_session`; the
  per‑clip ffmpeg extractions run on a thread pool while the manifest is still
  written in strict canonical order (determinism preserved).
- **Short‑slice safety** — WavLM slices below its conv receptive field are
  zero‑padded to a floor and the padding is masked out of pooling, so tiny
  per‑speaker intervals can't crash the encoder.

> **YOLO weights:** stage a **face** checkpoint at `models/yolov8.pt`
> (yolov8‑face), not the generic COCO model — the click must resolve against
> faces, not person boxes.

## 5. Determinism

The environment gate sets seeds, `torch.use_deterministic_algorithms(True)` (no
`warn_only` — a nondeterministic kernel HALTS), disables TF32/cuDNN autotuner,
and pins `CUBLAS_WORKSPACE_CONFIG`. Launch the process with a fixed hash seed so
child‑process hashing is stable too:

```bash
PYTHONHASHSEED=0 python3 your_launcher.py ...
```

The gate also runs a run‑twice self‑check and records it to the manifest; a
`FAIL` there means the box can't guarantee bit‑exactness and the run is aborted.

## 6. Verify before running real data (zero installs)

Every module self‑tests on plain Python with **no pip installs, no torch, no
GPU** — run these first on any box:

```bash
for m in forensics/pts.py forensics/determinism.py forensics/manifest.py \
         environment_gate.py layer0_preprocessor.py layer1_enrollment.py \
         layer2_tracker.py layer3_contamination.py pipeline_runner.py; do
  python3 "$m" --selftest || exit 1
done
python3 gpu_runtime.py --selftest || exit 1
```

## 7. Operator Clicks JSON

```json
{
  "speaking_click": {"file_index": 0, "x": 640, "y": 360, "pts_ms": 12000},
  "anti_click":     {"file_index": 0, "x": 200, "y": 300, "pts_ms": 12000}
}
```
`speaking_click` is mandatory (anchors target identity); `anti_click` is optional
(a face to exclude). All times are integer PTS milliseconds.

## 8. Run order & outputs

`pipeline_runner.run_session` executes: `session_start → environment_gate →
Layer 0 → Layer 1 → Layer 2 → Layer 3 → session_end`, all on one hash‑chained
manifest. Outputs land in `session_out/`:

- `clean/target_fNNN_<g0>-<g1>.wav` — verified uncontaminated target clips
  (integer‑PTS‑named, byte‑exact, SHA‑256'd in the manifest).
- `session.manifest.jsonl` — the append‑only, hash‑chained audit log; the final
  `output_hash` record carries every clip checksum + one session fingerprint.

Verify a finished manifest's chain any time with
`forensics.manifest.verify_chain(load_manifest(path))`.
