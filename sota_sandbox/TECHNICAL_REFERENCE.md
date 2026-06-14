# A/B Diarization Sandbox — Technical Reference

**Document status:** complete technical specification of the `sota_sandbox/` harness.
**Audience:** a reviewer/scientist who wants to interrogate the logic end-to-end.
**Scope:** every script, every function, every formula, every data format, and the
neural model internals — plus an anticipated-questions section.

> **Naming / confidentiality.** The in-house deterministic diarization system is
> referred to throughout as **the Forensic Pipeline** (abbreviated **FP**); its
> real project name is withheld. The sandbox **source and this document contain
> no occurrence of that name** — identifiers use neutral names (e.g.
> `extract_forensic_target()`, `MANIFEST_SCHEMA = "forensic-manifest-v1"`). The
> name does, however, still live in the FP's own **manifest data** (it writes it
> into the `schema` field); `sanitize_manifest.py` produces a redacted copy for
> study use. The neural baseline is **the SOTA model** (PyAnnote 3.1). See
> [§12](#12-confidentiality-status-of-the-name).

---

## Table of contents

1. [What this harness is, and what it measures](#1-what-this-harness-is-and-what-it-measures)
2. [Architecture & data flow](#2-architecture--data-flow)
3. [Environment, determinism, reproducibility](#3-environment-determinism-reproducibility)
4. [`run_sota.py` — the SOTA inference side](#4-run_sotapy--the-sota-inference-side)
5. [The PyAnnote 3.1 model, in depth](#5-the-pyannote-31-model-in-depth)
6. [`compare_results.py` — the comparator](#6-compare_resultspy--the-comparator)
7. [`run_ab.sh` — the orchestrator](#7-run_absh--the-orchestrator)
8. [Data formats (exact schemas)](#8-data-formats-exact-schemas)
9. [Worked example (the numbers, step by step)](#9-worked-example-the-numbers-step-by-step)
10. [Limitations, assumptions, threats to validity](#10-limitations-assumptions-threats-to-validity)
11. [Anticipated reviewer questions](#11-anticipated-reviewer-questions)
12. [Confidentiality status of the name](#12-confidentiality-status-of-the-name)
13. [Glossary & function index](#13-glossary--function-index)

---

## 1. What this harness is, and what it measures

### 1.1 The goal

A controlled **A/B comparison** between two diarizers run on the same clip:

- **A — the Forensic Pipeline (FP):** an in-house, deterministic, "defense-grade"
  diarizer. It enrolls **one** subject of interest (the *target*) and emits only
  that target's **verified clean speech**. It is conservative by construction:
  anywhere the target's audio is overlapped or contaminated, FP refuses to label
  it as clean (it is marked `NaN` rather than `CLEAN`).
- **B — the SOTA model:** PyAnnote 3.1, an off-the-shelf neural diarizer that
  labels **every** speaker it hears and keeps overlapping speech.

The harness asks one scientific question: **where do the two systems agree about
when the target is speaking, and where — and why — do they diverge?**

### 1.2 Why the comparison is asymmetric (and how we handle it)

The two systems do not produce comparable objects:

| | Forensic Pipeline | SOTA (PyAnnote 3.1) |
|---|---|---|
| Speakers labelled | one (the enrolled target) | all of them (`SPEAKER_00`, `SPEAKER_01`, …) |
| Overlapped target speech | **dropped** (not "clean") | **kept** |
| Determinism | bit-exact, by design | seeded but with minor GPU jitter |
| Output unit | "CLEAN target segment" on a global ms clock | "speaker turn" in seconds |

Because FP only describes **one** speaker, we cannot compute a classical
Diarization Error Rate (DER), which needs a *complete* multi-speaker reference.
Instead the comparator performs a **single-subject temporal overlap analysis**:

1. It identifies **which SOTA speaker best corresponds to the FP target** (the
   SOTA cluster with maximum temporal overlap with the FP clean timeline).
2. It quantifies agreement between that one SOTA speaker and the FP target using
   set-theoretic interval metrics (intersection, union, IoU, one-sided
   differences, coverage).
3. It renders the two timelines side-by-side so divergences are visually
   attributable.

### 1.3 The expected "signature"

Because FP deliberately discards contaminated target speech, the *expected*
healthy result is:

- **High intersection / IoU** — the two systems agree on where the target
  cleanly speaks.
- **Most divergence on the SOTA-only side** — i.e. PyAnnote labels the target as
  speaking in places FP vetoed for overlap/contamination. This is **not** an FP
  error; it is FP exercising its contamination veto.
- **Little or no FP-only** — if FP marks the target as cleanly speaking somewhere
  PyAnnote attributes to a different speaker or to silence, that is the
  interesting case worth investigating (a genuine disagreement).

This asymmetry is the single most important interpretive point in the whole
document; every metric below is designed around it.

---

## 2. Architecture & data flow

```
                      ┌──────────────────────────── sota_sandbox/ ───────────────────────────┐
                      │                                                                       │
 NT-clip27.mp4 ──────▶│  run_sota.py                                                          │
   (or .wav)          │  ├─ resolve_token()      HF token (env / .hf_token / --token)         │
                      │  ├─ ensure_wav()         ffmpeg → 16 kHz mono PCM wav  ──┐             │
                      │  ├─ pick_device()        auto → cuda                     │             │
                      │  ├─ run_diarization()    PyAnnote 3.1 pipeline ◀─────────┘             │
                      │  │     segmentation → embeddings → clustering → turns                  │
                      │  └─ assemble_output() ─────────────▶  sota_output.json                 │
                      │                                          (speaker turns, integer ms)   │
                      │                                                   │                    │
 session.manifest ───▶│  compare_results.py                              │                    │
   .jsonl  (from FP,  │  ├─ load_manifest()       JSON-L, schema-checked  │                    │
    supplied by you)  │  ├─ normalize_entry()     double-nesting unwrap   │                    │
                      │  ├─ extract_forensic_target() layer3_segment → target ▼ timeline           │
                      │  ├─ load_sota() ──────────────────────────────────┘                    │
                      │  ├─ match_target_speaker()  best-overlap SOTA cluster                  │
                      │  ├─ interval algebra        merge/intersect/subtract/IoU               │
                      │  └─ build_report() ───────▶  terminal report + ASCII timeline          │
                      │                                                                       │
                      │  run_ab.sh = preflight ▶ run_sota.py ▶ compare_results.py (one command) │
                      └───────────────────────────────────────────────────────────────────────┘
```

**Coupling surface.** The two halves share exactly one contract: the **on-disk
format** of the FP manifest. `compare_results.py` re-implements the manifest read
(read-only) and never imports any FP code. Nothing in the sandbox writes outside
`sota_sandbox/`.

---

## 3. Environment, determinism, reproducibility

### 3.1 Dev/deploy split

- **Authoring/self-test:** macOS (no CUDA). Only the stdlib `--selftest` paths run
  here. All heavy imports (`torch`, `pyannote.audio`) are **lazy** — performed
  *inside* functions, never at module import — so `--help` and `--selftest` work on
  plain Python with zero pip installs.
- **Real execution:** Ubuntu 22.04, NVIDIA RTX 6000 Ada (48 GB), CUDA 12.x,
  Python 3.10.

### 3.2 Pinned dependencies (`requirements-sota.txt`)

Installed into a **separate** virtualenv (`.venv-sota`), never the FP's
`requirements.txt`:

- `pyannote.audio==3.1.1` — the neural diarizer.
- `torch==2.1.2`, `torchaudio==2.1.2` — **installed from the cu121 wheel index**
  (`--index-url https://download.pytorch.org/whl/cu121`) to get CUDA builds; plain
  PyPI yields CPU-only wheels.
- `huggingface_hub` (gated-model auth), `soundfile` (wav IO), `numpy==1.26.4`
  (pinned `<2.0`; torch 2.1.2 predates the NumPy 2 ABI break).
- `ffmpeg` is a **system** package (`apt-get install ffmpeg`), not pip.

### 3.3 Determinism controls (`seed_everything`)

```python
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)          # if CUDA present
torch.use_deterministic_algorithms(True, warn_only=True)
```

- The seed (default `0`) is **recorded in `sota_output.json`** along with
  `torch`/`cuda`/`pyannote.audio` versions and the wav SHA-256.
- **Residual nondeterminism:** some cuDNN kernels are not bit-reproducible, so
  embeddings — and therefore clustering at the margins — can jitter slightly
  between runs. We do **not** claim FP's bit-exact determinism here; the recorded
  seed + versions + input hash make any run *attributable and re-runnable*, which
  is the appropriate standard for a baseline.
- We run PyAnnote at its default precision (FP32); no automatic mixed precision is
  forced, which keeps run-to-run variance low.

---

## 4. `run_sota.py` — the SOTA inference side

### 4.1 Control flow (`run()`)

```
resolve_token(args.token)           # raises if no token anywhere
ensure_wav(args.input)              # mp4 → 16 kHz mono wav (or pass through a wav)
pick_device(args.device)            # auto → cuda if available
run_diarization(wav, …)             # PyAnnote 3.1 → list of {start_ms,end_ms,speaker_id}
sha256_file(wav)                    # provenance hash of the exact audio scored
assemble_output(...)                # fold turns + provenance into the JSON object
write sota_output.json
```

Top-level `main()` wraps `run()` in a try/except so the operator sees a clean
one-line `ERROR: …` instead of a stack trace, and returns exit code `1` on failure.

### 4.2 Token resolution (`resolve_token`)

PyAnnote 3.1 is a **gated** model; loading it requires a Hugging Face token.
Resolution order (first non-empty wins):

1. `--token VALUE`
2. `$HF_TOKEN`, then `$HUGGINGFACE_TOKEN`, then `$HUGGING_FACE_HUB_TOKEN`
3. `sota_sandbox/.hf_token` (one line; git-ignored)
4. interactive `getpass` prompt — **only** if `stdin` is a TTY

If none is found it raises with actionable guidance. The token value is **never
logged, never written into `sota_output.json`, never committed**. The self-test
explicitly asserts no token string leaks into the serialized output.

### 4.3 Audio extraction (`ensure_wav` / `build_ffmpeg_cmd`)

If the input is already a `.wav`, it is used **as-is** — we deliberately do *not*
re-encode, because re-encoding would perturb the bytes (and the provenance hash)
for no benefit. Otherwise we extract with:

```
ffmpeg -nostdin -y -i <src> -vn -ac 1 -ar 16000 -c:a pcm_s16le -f wav <dst>
```

Flag-by-flag, and *why*:

| Flag | Meaning | Why |
|------|---------|-----|
| `-nostdin` | don't read stdin | avoids ffmpeg hijacking the terminal in scripts |
| `-y` | overwrite output | idempotent re-runs |
| `-vn` | drop video | diarization needs audio only |
| `-ac 1` | mono | PyAnnote expects single channel |
| `-ar 16000` | 16 kHz | the model's native sample rate (see §5) |
| `-c:a pcm_s16le` | 16-bit signed little-endian PCM | lossless, no codec nondeterminism |
| `-f wav` | WAV container | what soundfile/torchaudio read cleanly |

The extracted file lands in `work/` (git-ignored). Errors are surfaced cleanly:
`ffmpeg` missing → "install it on the Ubuntu box"; non-zero exit → the tail of
ffmpeg's stderr.

### 4.4 Device selection (`pick_device`)

- `cpu` → `cpu`.
- `cuda` → requires `torch.cuda.is_available()`; if false it **raises** (refusing to
  silently fall back), with a message reminding you this belongs on the GPU box.
- `auto` (default) → `cuda` if available, else `cpu` with a logged warning.

This makes "I thought it ran on GPU but it silently used CPU" impossible.

### 4.5 Turn extraction & millisecond conversion

PyAnnote returns a `pyannote.core.Annotation`. We iterate:

```python
for turn, _track, speaker in diarization.itertracks(yield_label=True):
    segments.append(segment_from_turn(turn.start, turn.end, speaker))
```

`turn.start`/`turn.end` are **float seconds**. `ms_from_seconds()` converts to
**integer milliseconds** via `int(round(seconds * 1000))`:

- Integer ms matches the FP's "integer PTS milliseconds" convention, so both
  timelines live on the same integer clock and interval math is exact (no float
  drift).
- `int(round(...))` uses Python's round-half-to-even; sub-ms boundaries are
  snapped to the nearest ms (worst-case 0.5 ms boundary error per endpoint).

### 4.6 Output assembly (`assemble_output`)

Segments are sorted by `(start_ms, end_ms, speaker_id)` so the file is **stable and
diffable** across runs. Provenance header fields are documented in [§8.2](#82-sota-output-sota_outputjson).

---

## 5. The PyAnnote 3.1 model, in depth

This is the section a reviewer will press hardest on: *what is the SOTA box
actually doing?* The pipeline id is `pyannote/speaker-diarization-3.1`. It is a
**three-stage** system — local segmentation, speaker embedding, global clustering —
followed by reconciliation.

### 5.1 Stage 1 — local speaker segmentation

- **Model:** `pyannote/segmentation-3.0`.
- **Operation:** the audio is processed in **sliding windows** (~10 s). For each
  window the model produces **frame-level** local speaker-activity predictions.
- **Powerset formulation (the key idea).** Rather than predicting an independent
  on/off activity per speaker (multi-label, which needs a threshold and handles
  overlap poorly), the model predicts, per frame, **one class from the *powerset***
  of up to **3 local speakers**: i.e. classes for {silence}, {spk₁}, {spk₂},
  {spk₃}, and the pairwise overlaps {spk₁spk₂}, {spk₁spk₃}, {spk₂spk₃}. Overlapped
  speech is therefore a **native output class**, not a post-hoc threshold. This is
  the contribution of Plaquet & Bredin, *"Powerset multi-class cross entropy loss
  for neural speaker diarization"* (Interspeech 2023).
- **Backbone:** pyannote's segmentation network (per the model card) — a learned
  acoustic front-end over the raw 16 kHz waveform followed by temporal modeling
  layers and a powerset classification head. (Exact layer dimensions are in the
  model card; the powerset head is the architecturally important part for this
  comparison.)
- "Local" speaker indices are **window-scoped** — `spk₁` in window *k* is not the
  same person as `spk₁` in window *k+1*. Stage 3 resolves that.

### 5.2 Stage 2 — speaker embeddings

- **Model:** `pyannote/wespeaker-voxceleb-resnet34-LM` (this is the change that
  defines **3.1** vs 3.0; 3.0 used a SpeechBrain ECAPA-TDNN via ONNX Runtime, 3.1
  moved to a **pure-PyTorch WeSpeaker ResNet34** embedding — removing the
  onnxruntime dependency and fixing a pipeline bug).
- For each local speaker region the model extracts a fixed-dimensional **speaker
  embedding** (an x-vector-like ResNet34 trained on VoxCeleb with large-margin
  finetuning). Overlapped frames are down-weighted/excluded so embeddings are
  computed from the cleanest available evidence for each local speaker.

### 5.3 Stage 3 — global clustering

- **Algorithm:** **Agglomerative Hierarchical Clustering (AHC)** over the local
  embeddings, using a **calibrated distance threshold** to decide how many global
  speakers there are. Local speakers from all windows whose embeddings are close
  enough are merged into one **global** speaker identity (`SPEAKER_00`, …).
- **Speaker count is automatic.** Our `run_sota.py` does **not** pass
  `num_speakers`/`min_speakers`/`max_speakers`, so the count is determined purely
  by the clustering threshold. (These can be supplied to constrain it.)

### 5.4 Reconciliation / aggregation

The per-window powerset segmentations are relabelled with their global speaker
ids and **aggregated across overlapping windows** into a single continuous
`Annotation`. Overlapping speech is preserved (two speakers can be simultaneously
active). Very short on/off fragments are smoothed by the pipeline's
`min_duration_on/off` parameters (pipeline defaults). The result is what
`itertracks()` yields.

### 5.5 What this means for the comparison

- PyAnnote can and will label **overlapped** target speech (the very thing FP
  vetoes) — hence the expected **SOTA-only** divergence.
- PyAnnote's number of speakers is an *estimate* from clustering; over- or
  under-clustering directly affects the speaker-matching step ([§6.5](#65-speaker-matching-match_target_speaker)).
- PyAnnote knows nothing about *which* speaker is the enrolled target — speaker
  ids are arbitrary. Aligning "SOTA speaker X" to "the FP target" is done by the
  comparator, by overlap, not by identity.

**References:** PyAnnote model cards (`pyannote/speaker-diarization-3.1`,
`/segmentation-3.0`, `/wespeaker-voxceleb-resnet34-LM`); Bredin, *"pyannote.audio
speaker diarization pipeline"*; Plaquet & Bredin (Interspeech 2023, powerset);
Wang et al., *WeSpeaker*.

---

## 6. `compare_results.py` — the comparator

Pure **Python standard library**. Read-only. It imports nothing from the FP. It is
the analytically dense half of the harness, so it gets the most space.

### 6.1 The data model: half-open integer-ms intervals

Every timeline is a set of **half-open intervals** `[start_ms, end_ms)` with
**integer** endpoints. Half-open means the endpoint is exclusive, so adjacent
segments `[0,10)` and `[10,20)` touch without overlapping. Integer ms means all
arithmetic is exact.

### 6.2 Interval algebra (the primitives)

All five operate on lists of `(a, b)` pairs and **normalize via `merge` first**, so
callers never have to pre-sort or pre-coalesce.

#### `merge(intervals)`
Sort by start, drop empty/inverted intervals (`b > a` filter), then sweep
coalescing any interval whose start is `<= ` the running end:

```python
if out and a <= out[-1][1]:
    out[-1][1] = max(out[-1][1], b)   # extend
else:
    out.append([a, b])                # new run
```

The `a <= out[-1][1]` test (**`<=`**, not `<`) means **touching** intervals merge:
`[0,10)+[10,20) → [0,20)`. Consequence: a **0-ms gap is bridged**. (FP segments are
already gap-aware on its side; this just means the rendered timeline treats
back-to-back segments as continuous.) Complexity `O(n log n)`.

#### `total_ms(intervals)`
`sum(b - a for a, b in merge(intervals))`. Merging first means **overlap is never
double-counted** — total duration is of the *union*, not the naive sum.

#### `intersect(a, b)`
Classic two-pointer sweep over both merged lists. For the current pair, the
overlap is `[max(a₀,b₀), min(a₁,b₁))`, emitted only if `hi > lo` (**strict** — so
merely touching intervals yield no spurious intersection). Advance whichever
interval ends first. `O(n+m)` after the merges.

#### `subtract(a, b)` → `a \ b`
For each interval in `a`, walk a `cursor` from its start; for every `b` interval
that overlaps, emit the gap before it (`[cursor, min(b_start, hi))`) and jump the
cursor past `b`'s end; emit any tail after the last `b`. Returns the parts of `a`
**not** covered by `b`. Used for the one-sided differences.

#### `union(a, b)`
`merge(a + b)` — coalesced union.

#### `iou(a, b)`
`total_ms(intersect(a,b)) / total_ms(union(a,b))` — the **Jaccard index** on time.
`0.0` when the union is empty. This is the single headline agreement number.

`pct(part, whole)` is a guarded percentage (`0` when `whole == 0`).

### 6.3 Manifest loading (`load_manifest`)

- Reads the FP manifest as **JSON Lines** — one JSON object per non-blank line.
- A malformed line raises `CompareError` naming the line number.
- An empty manifest raises.
- **Schema check (tolerant, non-leaking):** the first record's `schema` is
  accepted if it equals `MANIFEST_SCHEMA` (`"forensic-manifest-v1"`) **or** matches
  the family `"*-manifest-v1"`. So both a sanitized manifest and a raw one are
  accepted. The schema string is **never used to locate segments** (that keys off
  the `operation` field), so its exact value cannot break parsing.
  - On an unrecognized schema, the warning is **generic** — it does **not** echo
    the raw schema value, because an un-sanitized manifest still carries the FP's
    project name there.
  - `build_report` calls this with `strict_schema=False` (warn + continue);
    `strict_schema=True` raises — a hard gate if you want one.

### 6.4 The double-nesting unwrap (`normalize_entry`) — important

The FP writes some records (its parallel Layer-2/Layer-3 workers) **double-nested**:
the top-level entry's `payload` is *itself* a worker record:

```json
{ "operation": "layer3_segment",
  "payload": { "file_index": 0, "start_ms": 1000,
               "operation": "layer3_segment",
               "payload": { ...the real fields... } } }
```

`normalize_entry` detects this precisely and unwraps it:

```python
if (isinstance(payload, dict)
        and WORKER_WRAPPER_KEYS == set(payload.keys())   # exactly {file_index,start_ms,operation,payload}
        and payload.get("operation") == operation        # inner op echoes outer
        and isinstance(payload.get("payload"), dict)):
    return operation, payload["payload"]                  # the REAL payload
return operation, (payload if isinstance(payload, dict) else {})
```

Two guards prevent false positives: the wrapper must have **exactly** the four
keys `{file_index, start_ms, operation, payload}` (set equality, not subset), and
its inner `operation` must echo the outer one. A normal (single-level) record
falls through to the second `return`. This mirrors how the FP's own read-only
visualizer normalizes records, so parsing is faithful to the source format.

### 6.5 Forensic target extraction (`extract_forensic_target`)

Walks every normalized entry, keeps those with `operation == "layer3_segment"`
(the constant `OP_SEGMENT`), and from each real payload pulls **`start_global_ms`**
and **`end_global_ms`** — the target's clean-speech segment on the **global**
session clock. Guards: the payload must contain both fields and satisfy `g1 > g0`.

Returns `(intervals, segment_dicts)`:
- `intervals` — merged `[(start_global_ms, end_global_ms), …]`, the FP target
  timeline.
- `segment_dicts` — per-segment detail (`duration_ms`, `decision`, `block_count`,
  `bridged_gaps`) for the per-segment table, sorted by start.

These `layer3_segment` records **are** the FP's answer to "when is the target
cleanly speaking?". (The `NaN`/contaminated regions are simply *absent* from this
set — that absence is what drives the asymmetry in §1.3.)

### 6.6 SOTA loading & speaker grouping (`load_sota`)

Reads `sota_output.json`, groups `segments` by `speaker_id` into per-speaker merged
interval sets:

```
{ "SPEAKER_00": [(900,3100),(4800,5500)], "SPEAKER_01": [(3200,4500), …], … }
```

Each speaker's intervals are merged, so per-speaker totals are union durations.

### 6.7 Speaker matching (`match_target_speaker`)

```python
best, best_ov = None, -1
for spk, ivals in sorted(sota_by_speaker.items()):     # sorted → deterministic ties
    ov = total_ms(intersect(fp_intervals, ivals))
    if ov > best_ov:
        best, best_ov = spk, ov
return best, max(best_ov, 0)
```

**Picks the SOTA speaker whose speech overlaps the FP target the most** (in total
ms). This is the crux of aligning an unlabelled multi-speaker diarization to a
single enrolled subject. Properties & failure modes:

- **Deterministic:** speakers are iterated in sorted id order, so ties resolve to
  the lowest id.
- **Over-clustering** (FP target split across two SOTA clusters) → the matched
  speaker captures only part of the target → inflated **FP-only**.
- **Under-clustering** (FP target merged with another voice in one SOTA cluster) →
  matched speaker over-covers → inflated **SOTA-only**.
- **Degenerate case:** if the FP timeline is empty, every overlap is 0 and the
  match falls to the first speaker by id with 0 overlap; the report makes this
  obvious (0 s intersection).

### 6.8 Metrics computed in `build_report`

Let **P** = FP target timeline, **S** = matched SOTA speaker timeline.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| intersection | `total_ms(intersect(P,S))` | time both agree the target speaks |
| union | `total_ms(union(P,S))` | time either claims the target speaks |
| **IoU** | intersection / union | overall agreement (Jaccard), 0–1 |
| **FP-only** `P\S` | `total_ms(subtract(P,S))` | FP says clean target; SOTA missed / put it on another speaker → **investigate** |
| **SOTA-only** `S\P` | `total_ms(subtract(S,P))` | SOTA kept target speech FP vetoed (overlap/contamination) → **expected** |
| coverage P by S | intersection / `total_ms(P)` | fraction of FP target the SOTA speaker recovers |
| coverage S by P | intersection / `total_ms(S)` | fraction of the SOTA speaker FP endorses as clean |

The report also lists a **speaker inventory** (every SOTA speaker, total speech,
turn count, with the matched one flagged).

### 6.9 ASCII timeline (`render_lane`, `render_diff_lane`)

The session span `[t0, t1)` (from `_bounds`, the min start / max end across all
timelines) is quantized into `width` columns (default 100, `--width` to change).
Column *c* spans:

```
c0 = t0 + (span * c)     // width
c1 = t0 + (span * (c+1)) // width     # integer division
```

A column is **active** if **any** interval overlaps `[c0, c1)` (`a < c1 and b > c0`)
— i.e. **max-pooling** per column. Lanes:

- `FP` — `#` where the FP target is clean.
- `SOTA <matched>` — `#` where the matched SOTA speaker is active.
- (with `--all-speakers`) one `o`-lane per other SOTA speaker.
- `DIFF` — per column: `=` both active (agree), `P` FP-only, `S` SOTA-only, space =
  neither. (Legend printed beneath.)

**Quantization caveat:** each column ≈ `span/width` ms; events or gaps shorter than
one column may not be individually visible (max-pooling hides sub-column gaps).
The header prints the per-column ms so the resolution is explicit; raise `--width`
for finer granularity. Integer division distributes the off-by-one ms across
columns so lane widths are exactly `width` characters.

### 6.10 Per-segment agreement table

For each FP target segment, coverage = `intersect(segment, S) / segment.duration_ms`:

```
AGREE   if coverage ≥ 80%
PARTIAL if coverage ≥ 20%
MISS    otherwise
```

The denominator is the **segment's own reported `duration_ms`** from the manifest
(which may include FP-bridged gaps), not the naive `end−start`. This localizes
disagreement to specific FP segments — a reviewer can point at segment #N and ask
"why only 20%?".

### 6.11 Time formatting (`fmt_ms`)

Integer ms → `MM:SS.mmm` (sign-aware), via successive `divmod` by 1000 and 60.
`61234 → "01:01.234"`.

### 6.12 Self-test (`--selftest`)

Stdlib-only. Asserts: all five interval primitives on hand-checked cases; parsing
of a synthetic `forensic-manifest-v1` manifest **including the double-nested
`layer3_segment` shape** (built by `_wrap_layer3`); the speaker match; lane width &
DIFF alphabet; and `fmt_ms`. This is what lets the comparator be validated on a
laptop with no GPU, no torch, and no real data.

---

## 7. `run_ab.sh` — the orchestrator

One command runs the whole A/B test on the Ubuntu box:

```bash
./run_ab.sh /path/to/NT-clip27.mp4 /path/to/session.manifest.jsonl
```

It is defensive (`set -euo pipefail`) and does a **preflight** before touching the
model:

1. `python3` and `ffmpeg` present (fatal if not).
2. `nvidia-smi` present → prints GPU name/memory; absent → warns (are you on the
   right box?).
3. `torch` **and** `pyannote.audio` importable, and prints
   `cuda_available=…` (fatal if not importable → reminds you to install the venv).
4. Input clip and manifest both exist (fatal if not).
5. A token is available (`$HF_TOKEN` or a non-empty `.hf_token`) (fatal if not).

Then: `run_sota.py` (inference) → `sanitize_manifest.py` (redact the pipeline
name into a `*.clean.manifest.jsonl` copy — default on; if no redactable name is
detected it falls back to the original) → `compare_results.py` (report against the
clean copy). Env overrides: `DEVICE` (`cuda`), `OUT` (`sota_output.json`), `WIDTH`
(`120`), `SANITIZE` (`1`; set `0` to skip redaction). Inputs can also come from
`$INPUT`/`$MANIFEST` instead of positional args.

---

## 8. Data formats (exact schemas)

### 8.1 FP manifest input (consumed read-only)

JSON-Lines. Relevant shapes:

- **First line** carries `"schema": "forensic-manifest-v1"` (checked).
- **A `layer3_segment` worker record** (double-nested), inner real payload:

```json
{ "decision": "CLEAN",
  "start_local_ms": 1000, "end_local_ms": 3000,
  "start_global_ms": 1000, "end_global_ms": 3000,
  "duration_ms": 2000, "block_count": 4,
  "bridged_gaps": [], "wav_sha256": "…" }
```

The comparator reads **`start_global_ms` / `end_global_ms`** (global clock) and the
detail fields; it ignores the rest.

### 8.2 SOTA output (`sota_output.json`)

```json
{
  "schema": "sota-sandbox-output-v1",
  "created_utc": "2026-06-14T…Z",
  "source": "NT-clip27.mp4",
  "wav_path": "work/NT-clip27.16k.wav",
  "wav_sha256": "…",                  // hash of the exact audio scored
  "sample_rate": 16000,
  "model": "pyannote/speaker-diarization-3.1",
  "device": "cuda",
  "seed": 0,
  "library_versions": { "torch": "2.1.2", "cuda": "12.1", "pyannote.audio": "3.1.1" },
  "num_speakers": 2,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "num_segments": 7,
  "total_speech_ms": 23600,
  "segments": [ { "start_ms": 900, "end_ms": 5200, "speaker_id": "SPEAKER_00" }, … ]
}
```

`segments` is the "clean, simple" core; everything else is provenance so a run is
fully reproducible/attributable.

---

## 9. Worked example (the numbers, step by step)

Using the synthetic fixtures the harness was validated against (a ~28 s session):

- **FP target (P):** `[(1000,5000),(8000,12000),(20000,24000)]` → `total = 12.0 s`.
- **SOTA `SPEAKER_00`:** `[(900,5200),(7800,12500),(19500,25000)]` → `14.5 s`.
- **SOTA `SPEAKER_01`:** `[(5200,7800),(12500,15000),(25000,29000)]` → `9.1 s`.

Matching: overlap(P, SPEAKER_00) ≫ overlap(P, SPEAKER_01) → **matched = SPEAKER_00**.

- intersection(P, S) = 12.0 s (P sits entirely inside S's coverage).
- union(P, S) = 14.5 s.
- **IoU = 12.0 / 14.5 = 0.828.**
- FP-only `P\S` = 0.0 s → SOTA missed none of the FP-clean target.
- SOTA-only `S\P` = 2.5 s → e.g. the `24000–25000` region: PyAnnote kept it; FP
  vetoed it (overlap/contamination). **This is the expected signature.**
- coverage P-by-S = 100 %; coverage S-by-P = 82.8 %.
- Per-segment: all three FP segments at 100 % → **AGREE**.

Reading: the two systems **agree completely on where the target cleanly speaks**,
and disagree only where the SOTA model is willing to label contaminated target
speech that the FP forensically rejects. That is exactly the result the design
anticipates.

---

## 10. Limitations, assumptions, threats to validity

1. **Single-subject, not DER.** FP provides a one-speaker partial reference, so we
   do an overlap analysis, not a full DER/JER. (See [§11 Q3](#11-anticipated-reviewer-questions).)
2. **Speaker-matching ambiguity.** Over-/under-clustering by PyAnnote biases the
   one-sided differences ([§6.7](#67-speaker-matching-match_target_speaker)). Inspect the speaker inventory before
   trusting the headline IoU.
3. **Asymmetry is structural.** SOTA-only time is *expected* and is **not** a SOTA
   error nor an FP error — it is FP's contamination veto. Do not read SOTA-only as
   "disagreement to be minimized".
4. **Quantization in the ASCII view.** Sub-column events/gaps can be hidden; the
   metrics (computed at full ms resolution) are authoritative, the timeline is a
   visual aid. Raise `--width`.
5. **Millisecond rounding.** `int(round(s*1000))` introduces ≤0.5 ms per endpoint.
6. **GPU nondeterminism.** Embeddings can jitter; clustering at the margins may
   shift between runs. Seed + versions + wav hash are recorded for attribution.
7. **Schema coupling.** The comparator targets `operation == "layer3_segment"` with
   `start_global_ms`/`end_global_ms`. The schema string is checked only
   tolerantly (`"*-manifest-v1"`) and is **not** used to find segments, so it can
   be renamed/sanitized freely. If the FP renames the *operation* or the *field
   names*, the comparator extracts nothing (visible immediately) — **validate
   field names against the first real manifest** before trusting results. (This is
   the one place the harness was built to a documented schema rather than a real
   file.)
8. **ffmpeg resampling fidelity.** Down-mix to mono + 16 kHz is a fixed,
   reproducible transform but is a transform; the SOTA model scores the resampled
   audio, hashed for provenance. A pass-through wav's *actual* rate/channels are
   read (stdlib `wave`) and recorded — provenance is no longer assumed 16 kHz.
9. **Name in the manifest data.** The FP writes its project name into the manifest
   `schema` field. `sanitize_manifest.py` removes it in a redacted copy; the
   original is never modified, and removing it at the source would require changing
   the FP (out of scope).

---

## 11. Anticipated reviewer questions

**Q1. How does PyAnnote decide the number of speakers?**
Agglomerative clustering with a calibrated distance threshold over WeSpeaker
ResNet34 embeddings ([§5.3](#53-stage-3--global-clustering)). We pass no `num_speakers` constraint, so the count is
data-driven. It can be over- or under-estimated; the speaker inventory in the
report exposes the result.

**Q2. How is "the target" identified in an unlabelled diarization?**
By **maximum temporal overlap** with the FP clean timeline ([§6.7](#67-speaker-matching-match_target_speaker)). It is an
overlap-based alignment, not voice identity — stated as an explicit assumption.

**Q3. Why not Diarization Error Rate (DER)?**
DER needs a *complete* reference annotation (all speakers, all time) and an optimal
reference↔hypothesis speaker mapping. FP intentionally emits only the target's
clean speech, so a complete reference does not exist here. IoU + one-sided
differences + coverage are the correct single-subject analogue. If a full
multi-speaker reference were ever produced, DER (with `pyannote.metrics`) could be
added without changing the inference side.

**Q4. Why is the SOTA timeline usually "longer" than FP's?**
Because PyAnnote keeps overlapped speech and FP discards contaminated target
speech. The surplus shows up as **SOTA-only** and is expected, not error ([§1.3](#13-the-expected-signature)).

**Q5. Is the comparison reproducible?**
The comparator is pure/deterministic. The inference is seeded and records seed +
library versions + input audio SHA-256; residual GPU kernel nondeterminism is the
only source of variance, and it is bounded and attributable ([§3.3](#33-determinism-controls-seed_everything)).

**Q6. Why integer milliseconds and half-open intervals?**
Integer ms makes all set operations exact (no float drift) and matches the FP's
integer-PTS-ms convention. Half-open intervals make adjacency unambiguous
(`[0,10)+[10,20)` touch, don't overlap) ([§6.1](#61-the-data-model-half-open-integer-ms-intervals)).

**Q7. What stops the comparator from corrupting the FP pipeline?**
It imports no FP code, opens the manifest **read-only**, and writes nothing outside
`sota_sandbox/`. Its only knowledge of the FP is the manifest's on-disk format.

**Q8. What if the manifest schema or field names differ from what's documented?**
Non-strict schema check → a stderr warning and best-effort parse; if the operation
name differs, zero segments are extracted (visible immediately in the report).
This is called out as the primary validation step against the first real run
([§10.7](#10-limitations-assumptions-threats-to-validity)).

**Q9. Could clustering errors masquerade as agreement/disagreement?**
Yes — under-clustering inflates SOTA-only, over-clustering inflates FP-only. Always
read the **speaker inventory** and the **per-segment table** alongside the headline
IoU; they localize the cause.

---

## 12. Confidentiality status of the name

The withheld project name has been removed from everything inside this sandbox:

- **This document** — zero occurrences.
- **All sandbox source** (`run_sota.py`, `compare_results.py`, `run_ab.sh`,
  `README.md`, `requirements-sota.txt`) — scrubbed. Identifiers use neutral names:
  `extract_forensic_target`, lane label `FP`, report header *"FORENSIC PIPELINE VS
  SOTA …"*, and `MANIFEST_SCHEMA = "forensic-manifest-v1"`. **The schema constant
  was safely changed** — it is downstream/read-only, the FP never reads it, and
  parsing keys off the `operation` field, not the schema string.

The name survives in exactly **one** place the sandbox cannot rewrite at the
source: the **FP's own manifest output**, which embeds the name in each record's
`schema` field (and possibly in stored paths). That is data the FP produced; the
only fixes are (a) the FP team renames its schema at the source, or (b) you run:

```bash
python3 sanitize_manifest.py session.manifest.jsonl -o session.clean.manifest.jsonl
```

`sanitize_manifest.py` auto-detects the name token from the manifest's own schema
(so the confidential word is never hardcoded in the sandbox), writes a redacted
copy, leaves the original untouched, and never prints the token. Feed the clean
copy to `compare_results.py`. Note: the operation name `"layer3_segment"` and the
`layerN` field naming are *not* the withheld name and are retained as-is (they are
the manifest's real keys, needed to parse it).

---

## 13. Glossary & function index

**Glossary**
- **FP (Forensic Pipeline)** — the in-house deterministic single-target diarizer
  (name withheld).
- **SOTA model** — PyAnnote 3.1 neural diarizer.
- **target** — the one enrolled subject FP tracks.
- **CLEAN segment** — an FP `layer3_segment` record: verified clean target speech.
- **turn** — a PyAnnote speaker-active interval.
- **IoU** — Intersection-over-Union (Jaccard) of two timelines.
- **powerset segmentation** — PyAnnote's overlap-native local model output.
- **AHC** — Agglomerative Hierarchical Clustering (global speaker assignment).

**`run_sota.py`** — `resolve_token`, `ms_from_seconds`, `segment_from_turn`,
`build_ffmpeg_cmd`, `sha256_file`, `assemble_output`, `ensure_wav`, `pick_device`,
`seed_everything`, `library_versions`, `run_diarization`, `run`, `_selftest`,
`build_parser`, `main`.

**`compare_results.py`** — `merge`, `total_ms`, `intersect`, `subtract`, `union`,
`iou`, `pct`, `load_manifest`, `normalize_entry`, `extract_forensic_target`,
`load_sota`, `match_target_speaker`, `_bounds`, `render_lane`, `render_diff_lane`,
`fmt_ms`, `build_report`, `_wrap_layer3`, `_selftest`, `build_parser`, `main`.

**`sanitize_manifest.py`** — `detect_name_token`, `redact_text`, `sanitize`,
`_default_out`, `_selftest`, `build_parser`, `main`.

**`run_ab.sh`** — preflight → `run_sota.py` → `compare_results.py`.

*End of technical reference.*
