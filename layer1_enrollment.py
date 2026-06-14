"""
Forensic pipeline — layer1_enrollment.py
=========================================

Purpose:    Layer 1 securely anchors the TARGET's identity from the operator's
            click, then extracts a WavLM acoustic seed profile — before any
            tracking. The chain:
              1. Parse Operator Clicks JSON (mandatory speaking_click
                 {file_index,x,y,pts_ms}; optional anti_click).
              2. Seek the exact video frame at speaking_click.pts_ms (integer
                 PTS) in the referenced clip.
              3. YOLOv8 detects faces in that frame; the operator (x,y) is
                 resolved to the TIGHTEST enclosing box ⇒ the target face.
                 anti_click (if present) marks a face to EXCLUDE.
              4. InsightFace embeds the target (and anti) face.
              5. Across the session's VAD speech, keep only windows where the
                 target is the SOLE face on screen (uncontaminated) ⇒
                 enrollment windows.
              6. WavLM embeds those windows; pool ⇒ the L2-normalised seed
                 profile handed to Layer 2.

Adapter seams (exactly like Layer 0's VadAdapter):
            FrameSource (decode), FaceDetector (YOLOv8), FaceEmbedder
            (InsightFace), WavLMEmbedder (WavLM) are all abstract. Real
            implementations take their heavy deps by INJECTION and are reached
            only on the Ubuntu GPU box; `Stub*` implementations make all of the
            click-geometry / matching / windowing / pooling logic testable here
            on plain Python (pure-python cosine + pooling, no numpy/torch).

Forensic:   Integer-ms time throughout. Every decision (click resolution, the
            target/anti embedding checksums, the enrollment windows, the seed
            checksum) is logged; a click that lands on no face, or zero clean
            enrollment windows, is recorded as a guardrail failure and halts.

Run / test: python3 layer1_enrollment.py --selftest      (stdlib only)
"""
from __future__ import annotations

import abc
import math
import struct
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from forensics.pts import GlobalClock  # noqa: F401  (type clarity for Layer0Result)
from forensics.manifest import sha256_hex

# Fixed, manifest-logged enrollment parameters (engineering choices, not a
# legacy copy — tuned for an InsightFace/WavLM seed).
FACE_MATCH_THRESHOLD = 0.40    # cosine over InsightFace normed embeddings; ≥ ⇒ same identity
ENROLL_SAMPLE_MS = 250         # frame-sampling cadence inside a speech window
SEED_MIN_MS = 3000             # minimum clean target audio for a usable seed (else halt)

Box = Tuple[int, int, int, int, float]   # (x1, y1, x2, y2, score)


class EnrollError(Exception):
    """A guardrail failure in enrollment (recorded to the manifest, then raised)."""


# =============================================================================
# Operator clicks
# =============================================================================

@dataclass
class Click:
    file_index: int
    x: int
    y: int
    pts_ms: int


@dataclass
class Clicks:
    speaking: Click
    anti: Optional[Click] = None


def parse_clicks(obj):
    """Validate the Operator Clicks JSON object into a Clicks. speaking_click is
    mandatory; anti_click optional (absent or null)."""
    if not isinstance(obj, dict) or "speaking_click" not in obj:
        raise EnrollError("operator clicks: missing mandatory 'speaking_click'")

    def _one(d, name):
        try:
            return Click(int(d["file_index"]), int(d["x"]), int(d["y"]), int(d["pts_ms"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise EnrollError(f"operator clicks: malformed {name}: {exc}")

    speaking = _one(obj["speaking_click"], "speaking_click")
    anti_raw = obj.get("anti_click")
    anti = _one(anti_raw, "anti_click") if isinstance(anti_raw, dict) else None
    return Clicks(speaking=speaking, anti=anti)


# =============================================================================
# Pure geometry / identity logic (no numpy/torch — fully testable here)
# =============================================================================

def point_in_box(x, y, box):
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return x1 <= x <= x2 and y1 <= y <= y2


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def resolve_click_to_box(x, y, boxes):
    """Index of the TIGHTEST (smallest-area) box containing (x,y), or None."""
    best_i, best_area = None, None
    for i, b in enumerate(boxes):
        if point_in_box(x, y, b):
            area = box_area(b)
            if best_area is None or area < best_area:
                best_i, best_area = i, area
    return best_i


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


def l2_normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else [float(x) for x in v]


def is_target_face(emb, target_ref, anti_ref, threshold):
    """A detected face is the target iff it matches the target reference at or
    above threshold AND (no anti, or it does not match the anti more strongly)."""
    ct = cosine(emb, target_ref)
    if ct < threshold:
        return False
    if anti_ref is not None and cosine(emb, anti_ref) >= ct:
        return False
    return True


def frame_is_clean_target(boxes, embs, target_ref, anti_ref, threshold):
    """Clean iff EXACTLY ONE face is present and it is the target — a single
    on-screen face is the conservative proxy for uncontaminated target audio."""
    return len(boxes) == 1 and is_target_face(embs[0], target_ref, anti_ref, threshold)


def pool_seed(vectors):
    """Mean of L2-normalised window vectors, re-normalised. Deterministic."""
    if not vectors:
        raise EnrollError("no enrollment vectors to pool into a seed")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise EnrollError("enrollment vectors have inconsistent dimension")
    norm = [l2_normalize(v) for v in vectors]
    mean = [sum(v[i] for v in norm) / len(norm) for i in range(dim)]
    return l2_normalize(mean)


def vector_sha256(v):
    """Deterministic checksum of an embedding (packed as little-endian float64)."""
    return sha256_hex(struct.pack(f"<{len(v)}d", *[float(x) for x in v]))


def _sample_points(a, b, sample_ms):
    """Deterministic sample times inside [a, b): a grid every sample_ms, or the
    midpoint for an interval shorter than the cadence."""
    pts = list(range(a + sample_ms // 2, b, sample_ms))
    return pts or [(a + b) // 2]


# =============================================================================
# Adapter seams
# =============================================================================

class FrameSource(abc.ABC):
    @abc.abstractmethod
    def get_frame(self, file_index, pts_ms):
        ...


class FaceDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, frame) -> List[Box]:
        ...


class FaceEmbedder(abc.ABC):
    @abc.abstractmethod
    def embed(self, frame, box) -> List[float]:
        ...


class WavLMEmbedder(abc.ABC):
    @abc.abstractmethod
    def embed_window(self, wav_path, start_ms, end_ms) -> List[float]:
        ...


# ---- stubs for self-tests (deterministic, pure python) ----------------------

class StubFrameSource(FrameSource):
    def get_frame(self, file_index, pts_ms):
        return (file_index, pts_ms)               # opaque token the stubs key on


class StubDetector(FaceDetector):
    """`detect_fn(frame) -> [Box]`."""
    def __init__(self, detect_fn):
        self._fn = detect_fn

    def detect(self, frame):
        return list(self._fn(frame))


class StubEmbedder(FaceEmbedder):
    """`embed_fn(frame, box) -> vector`."""
    def __init__(self, embed_fn):
        self._fn = embed_fn

    def embed(self, frame, box):
        return list(self._fn(frame, box))


class StubWavLM(WavLMEmbedder):
    """`embed_fn(wav_path, a, b) -> vector`."""
    def __init__(self, embed_fn):
        self._fn = embed_fn

    def embed_window(self, wav_path, start_ms, end_ms):
        return list(self._fn(wav_path, start_ms, end_ms))


# =============================================================================
# Result
# =============================================================================

@dataclass
class Layer1Result:
    target_embedding: List[float]
    anti_embedding: Optional[List[float]]
    enrollment_windows: List[Tuple[int, int, int]]          # (file_index, start_ms, end_ms) local
    enrollment_windows_global: List[Tuple[int, int]] = field(default_factory=list)
    seed: List[float] = field(default_factory=list)


# =============================================================================
# Enrollment orchestration
# =============================================================================

def select_enrollment_windows(speech_by_file, *, frame_source, detector, embedder,
                              target_ref, anti_ref, threshold=FACE_MATCH_THRESHOLD,
                              sample_ms=ENROLL_SAMPLE_MS):
    """Keep VAD speech windows where EVERY sampled frame shows the target as the
    sole face. Deterministic order (sorted files, ordered intervals); no cap —
    all clean audio contributes to the seed."""
    windows: List[Tuple[int, int, int]] = []
    for fi in sorted(speech_by_file):
        for a, b in speech_by_file[fi]:
            clean = True
            for p in _sample_points(a, b, sample_ms):
                frame = frame_source.get_frame(fi, p)
                boxes = detector.detect(frame)
                embs = [embedder.embed(frame, bx) for bx in boxes]
                if not frame_is_clean_target(boxes, embs, target_ref, anti_ref, threshold):
                    clean = False
                    break
            if clean:
                windows.append((fi, a, b))
    return windows


def enroll(layer0, clicks_obj, *, manifest_writer, frame_source, detector,
           embedder, wavlm, threshold=FACE_MATCH_THRESHOLD,
           sample_ms=ENROLL_SAMPLE_MS, seed_min_ms=SEED_MIN_MS):
    """Full Layer 1. `layer0` is the Layer0Result (.files, .speech_by_file,
    .global_clock). Records every decision; raises EnrollError (after logging)
    on a guardrail failure."""
    clicks = parse_clicks(clicks_obj)
    files_by_index = {f.file_index: f for f in layer0.files}

    # 1–4: resolve the target (and anti) identity from the click frame(s).
    frame = frame_source.get_frame(clicks.speaking.file_index, clicks.speaking.pts_ms)
    boxes = detector.detect(frame)
    ti = resolve_click_to_box(clicks.speaking.x, clicks.speaking.y, boxes)
    if ti is None:
        manifest_writer.append("layer1_failure", {
            "reason": "speaking_click did not land on any detected face",
            "x": clicks.speaking.x, "y": clicks.speaking.y,
            "file_index": clicks.speaking.file_index, "pts_ms": clicks.speaking.pts_ms,
            "n_boxes": len(boxes),
        })
        raise EnrollError("speaking_click did not land on any detected face")
    target_box = boxes[ti]
    target_ref = embedder.embed(frame, target_box)

    anti_ref = None
    anti_box = None
    if clicks.anti is not None:
        af = frame_source.get_frame(clicks.anti.file_index, clicks.anti.pts_ms)
        aboxes = detector.detect(af)
        ai = resolve_click_to_box(clicks.anti.x, clicks.anti.y, aboxes)
        if ai is not None:
            anti_box = aboxes[ai]
            anti_ref = embedder.embed(af, anti_box)

    manifest_writer.append("layer1_click", {
        "file_index": clicks.speaking.file_index,
        "x": clicks.speaking.x, "y": clicks.speaking.y, "pts_ms": clicks.speaking.pts_ms,
        "target_box": [int(v) for v in target_box[:4]],
        "target_box_score": float(target_box[4]),
        "target_emb_dim": len(target_ref),
        "target_emb_sha256": vector_sha256(target_ref),
        "anti_present": anti_ref is not None,
        "anti_emb_sha256": vector_sha256(anti_ref) if anti_ref is not None else None,
        "face_match_threshold": threshold,
    })

    # 5: clean enrollment windows.
    windows = select_enrollment_windows(
        layer0.speech_by_file, frame_source=frame_source, detector=detector,
        embedder=embedder, target_ref=target_ref, anti_ref=anti_ref,
        threshold=threshold, sample_ms=sample_ms)
    windows_global = [layer0.global_clock.to_global_interval(fi, a, b)
                      for fi, a, b in windows]
    total_ms = sum(b - a for _, a, b in windows)
    manifest_writer.append("layer1_enrollment_windows", {
        "n_windows": len(windows),
        "total_ms": total_ms,
        "windows_global_ms": [[g0, g1] for g0, g1 in windows_global],
        "sample_ms": sample_ms, "seed_min_ms": seed_min_ms, "threshold": threshold,
    })
    if not windows:
        manifest_writer.append("layer1_failure", {
            "reason": "no clean single-target enrollment windows found",
        })
        raise EnrollError("no clean single-target enrollment windows found")
    if total_ms < seed_min_ms:
        manifest_writer.append("layer1_failure", {
            "reason": "clean enrollment audio below seed_min_ms",
            "total_ms": total_ms, "seed_min_ms": seed_min_ms,
        })
        raise EnrollError(f"clean enrollment audio {total_ms}ms < seed_min_ms {seed_min_ms}ms")

    # 6: WavLM seed.
    vecs = [wavlm.embed_window(files_by_index[fi].wav_path, a, b)
            for fi, a, b in windows]
    seed = pool_seed(vecs)
    manifest_writer.append("layer1_seed", {
        "dim": len(seed),
        "seed_sha256": vector_sha256(seed),
        "n_windows": len(windows),
        "total_enroll_ms": sum(b - a for _, a, b in windows),
    })

    return Layer1Result(target_embedding=target_ref, anti_embedding=anti_ref,
                        enrollment_windows=windows,
                        enrollment_windows_global=windows_global, seed=seed)


# =============================================================================
# Real adapters (heavy deps INJECTED; run only on the Ubuntu GPU box)
# =============================================================================

class VideoFrameSource(FrameSource):
    """Decodes the exact frame at an integer-ms PTS. `decode_fn(path, pts_ms)`
    is injected (ffmpeg/decord/opencv on Ubuntu). `paths` maps file_index→video."""
    def __init__(self, paths, decode_fn):
        self._paths = dict(paths)
        self._decode = decode_fn

    def get_frame(self, file_index, pts_ms):
        return self._decode(self._paths[file_index], int(pts_ms))


def _iou(box, bbox):
    """IoU of a detector box (x1,y1,x2,y2,score) and an InsightFace bbox
    (x1,y1,x2,y2). Used to associate a full-frame InsightFace detection back to
    the YOLO box the operator's click resolved to."""
    ax1, ay1, ax2, ay2 = box[0], box[1], box[2], box[3]
    bx1, by1, bx2, by2 = (float(bbox[0]), float(bbox[1]),
                          float(bbox[2]), float(bbox[3]))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class YoloFaceDetector(FaceDetector):
    """YOLOv8 FACE detector; `model` injected (ultralytics, weights pre-staged —
    stage a yolov8-FACE checkpoint, not the generic COCO model, so boxes are
    faces not persons). Runs on `device` under the gate's determinism flags.
    Returns boxes as (x1,y1,x2,y2,score)."""
    def __init__(self, model, *, conf=0.5, device="cuda"):
        self._model = model
        self._conf = conf
        self._device = device

    def detect(self, frame):
        res = self._model(frame, conf=self._conf, verbose=False, device=self._device)
        out = []
        for r in res:
            for b in r.boxes:
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                out.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], float(b.conf[0])))
        # deterministic order
        return sorted(out, key=lambda bx: (bx[0], bx[1], bx[2], bx[3]))


class InsightFaceEmbedder(FaceEmbedder):
    """InsightFace embedding for a detector box. `app` (FaceAnalysis, prepared on
    the right ctx_id) injected.

    Runs detection on the FULL frame (a tight YOLO crop frequently defeats
    InsightFace's own detector) and associates the result back to the requested
    box by best IoU. Full-frame results are cached per frame object so the
    several boxes of one frame share a single `app.get` call."""
    def __init__(self, app, *, min_iou=0.30, cache_size=8):
        self._app = app
        self._min_iou = min_iou
        self._cache: Dict[int, list] = {}
        self._order: List[int] = []
        self._cap = cache_size

    def _faces(self, frame):
        key = id(frame)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        faces = self._app.get(frame)                       # full frame, once
        self._cache[key] = faces
        self._order.append(key)
        if len(self._order) > self._cap:
            self._cache.pop(self._order.pop(0), None)
        return faces

    def embed(self, frame, box):
        faces = self._faces(frame)
        if not faces:
            raise EnrollError("InsightFace found no face in the frame")
        best = max(faces, key=lambda f: _iou(box, f.bbox))
        if _iou(box, best.bbox) < self._min_iou:
            raise EnrollError("no InsightFace detection overlaps the detector box "
                              f"(best IoU < {self._min_iou})")
        return [float(v) for v in best.normed_embedding.tolist()]


class WavLMSeedEmbedder(WavLMEmbedder):
    """Mean-pooled WavLM features over a window.

    `model` is injected and SHARED with Layer 2's EEND adapter (loaded once → one
    resident WavLM, not two). Slices come from the injected `audio_cache` (RAM,
    no disk re-reads) when present, else the stdlib `wave` module. Sub-receptive-
    field slices are zero-padded to `MIN_WAVLM_SAMPLES` and that padding is masked
    out of the mean. `embed_windows_batch` runs many slices through WavLM in ONE
    forward pass (fills the GPU); `embed_window` is the scalar fallback. Lazy torch."""
    def __init__(self, model, *, sr=16000, device="cuda", audio_cache=None):
        self._model = model
        self._sr = sr
        self._device = device
        self._cache = audio_cache

    # ---- scalar (also the no-cache fallback) --------------------------------
    def embed_window(self, wav_path, start_ms, end_ms):
        return self.embed_windows_batch(wav_path, [(start_ms, end_ms)])[0]

    # ---- batched (one WavLM forward for many slices) ------------------------
    def embed_windows_batch(self, wav_path, intervals):
        import torch
        if not intervals:
            return []
        if self._cache is not None:
            batch, lens = self._cache.batch(wav_path, intervals, device=self._device)
        else:
            batch, lens = self._read_batch(wav_path, intervals)
        with torch.no_grad():
            feats = self._model(batch).last_hidden_state        # [B, F, D]
        n_frames = feats.shape[1]
        out = []
        for i in range(feats.shape[0]):
            from gpu_runtime import AudioCache
            valid = AudioCache.feature_frames(int(lens[i].item()), n_frames)
            pooled = feats[i, :valid, :].mean(dim=0)            # mask the pad tail
            out.append([float(v) for v in pooled.cpu().tolist()])
        return out

    def _read_batch(self, wav_path, intervals):
        """RAM-cache-free path (stdlib wave): build the same padded batch tensor."""
        import array
        import wave
        import torch
        from forensics.pts import ms_to_samples
        from gpu_runtime import MIN_WAVLM_SAMPLES
        segs, lengths = [], []
        with wave.open(str(wav_path), "rb") as w:
            for s, e in intervals:
                w.setpos(ms_to_samples(s, self._sr))
                n = ms_to_samples(e, self._sr) - ms_to_samples(s, self._sr)
                frames = w.readframes(max(0, n))
                pcm = array.array("h")
                pcm.frombytes(frames)
                t = torch.tensor(pcm, dtype=torch.float32) / 32768.0
                lengths.append(int(t.numel()))
                if t.numel() < MIN_WAVLM_SAMPLES:
                    t = torch.nn.functional.pad(t, (0, MIN_WAVLM_SAMPLES - t.numel()))
                segs.append(t)
        tmax = max(t.numel() for t in segs)
        segs = [torch.nn.functional.pad(t, (0, tmax - t.numel())) if t.numel() < tmax
                else t for t in segs]
        batch = torch.stack(segs, dim=0).to(self._device)
        return batch, torch.tensor(lengths, dtype=torch.long)


# =============================================================================
# Self-test (stdlib only)
# =============================================================================

def _selftest():
    import tempfile
    from pathlib import Path
    from forensics.manifest import ManifestWriter, load_manifest, verify_chain
    from forensics.pts import GlobalClock as GC
    from layer0_preprocessor import Layer0File, Layer0Result

    # --- pure geometry / identity ---
    boxes = [(100, 100, 300, 300, 0.8), (120, 120, 180, 180, 0.9)]  # big, tight
    assert resolve_click_to_box(150, 150, boxes) == 1               # tightest enclosing
    assert resolve_click_to_box(10, 10, boxes) is None
    assert abs(cosine([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-9
    assert cosine([1, 0, 0], [0, 1, 0]) == 0.0
    assert is_target_face([1, 0, 0], [1, 0, 0], None, 0.5)
    assert not is_target_face([0, 1, 0], [1, 0, 0], None, 0.5)
    # anti pulls a face away from "target"
    assert not is_target_face([0, 1, 0], [1, 0, 0], [0, 1, 0], 0.5)
    assert pool_seed([[2, 0, 0]]) == [1.0, 0.0, 0.0]
    assert _sample_points(0, 500, 250) == [125, 375]
    assert _sample_points(1000, 1250, 250) == [1125]

    assert parse_clicks({"speaking_click": {"file_index": 0, "x": 1, "y": 2,
                                            "pts_ms": 3}}).anti is None
    try:
        parse_clicks({"nope": 1})
    except EnrollError:
        pass
    else:
        raise AssertionError("missing speaking_click not caught")

    # --- end-to-end enrollment with stubs ---
    TARGET, OTHER = (120, 120, 180, 180, 0.9), (300, 100, 400, 200, 0.8)

    def detect_fn(frame):
        fi, pts = frame
        if pts == 12000:                         # click frame: two faces, click on TARGET
            return [(100, 100, 300, 300, 0.7), TARGET]
        if 5000 <= pts < 6000:                   # contaminated region: a second face present
            return [TARGET, OTHER]
        return [TARGET]                          # clean: sole target face

    def embed_fn(frame, box):
        return [1.0, 0.0, 0.0] if box[:4] == TARGET[:4] else [0.0, 1.0, 0.0]

    def wavlm_fn(wav, a, b):
        return [2.0, 0.0, 0.0]

    files = [Layer0File(0, "a.mp4", "a.wav", "sha", 16000, 1, 6000, 0)]
    layer0 = Layer0Result(sr=16000, files=files, global_clock=GC([6000]),
                          speech_by_file={0: [(0, 4000), (5000, 5250)]},
                          speech_global=[(0, 4000), (5000, 5250)])
    clicks = {"speaking_click": {"file_index": 0, "x": 150, "y": 150, "pts_ms": 12000}}

    with tempfile.TemporaryDirectory() as d:
        mpath = Path(d) / "session.manifest.jsonl"
        with ManifestWriter(mpath, fsync=False) as w:
            w.append("session_start", {"seed": 0})
            res = enroll(layer0, clicks, manifest_writer=w,
                         frame_source=StubFrameSource(), detector=StubDetector(detect_fn),
                         embedder=StubEmbedder(embed_fn), wavlm=StubWavLM(wavlm_fn))

        # (0,4000) clean & ≥ seed_min; (5000,5250) contaminated (2nd face) -> excluded
        assert res.enrollment_windows == [(0, 0, 4000)], res.enrollment_windows
        assert res.enrollment_windows_global == [(0, 4000)]
        assert res.target_embedding == [1.0, 0.0, 0.0]
        assert res.seed == [1.0, 0.0, 0.0]

        entries = load_manifest(mpath)
        assert verify_chain(entries) is True
        ops = [e["operation"] for e in entries]
        assert ops == ["session_start", "layer1_click", "layer1_enrollment_windows",
                       "layer1_seed"], ops
        assert entries[1]["payload"]["target_box"] == [120, 120, 180, 180]
        assert entries[1]["payload"]["face_match_threshold"] == 0.40

        # failure 1: click lands on no face -> logged + raised
        bad = {"speaking_click": {"file_index": 0, "x": 5, "y": 5, "pts_ms": 12000}}
        mp2 = Path(d) / "fail_click.manifest.jsonl"
        with ManifestWriter(mp2, fsync=False) as w:
            try:
                enroll(layer0, bad, manifest_writer=w, frame_source=StubFrameSource(),
                       detector=StubDetector(detect_fn), embedder=StubEmbedder(embed_fn),
                       wavlm=StubWavLM(wavlm_fn))
            except EnrollError:
                pass
            else:
                raise AssertionError("expected EnrollError on click off all faces")
        fe = load_manifest(mp2)
        assert verify_chain(fe) and fe[-1]["operation"] == "layer1_failure"

        # failure 2: clean audio below seed_min_ms -> logged + raised
        short = Layer0Result(sr=16000, files=files, global_clock=GC([6000]),
                             speech_by_file={0: [(0, 500)]}, speech_global=[(0, 500)])
        mp3 = Path(d) / "fail_seed.manifest.jsonl"
        with ManifestWriter(mp3, fsync=False) as w:
            try:
                enroll(short, clicks, manifest_writer=w, frame_source=StubFrameSource(),
                       detector=StubDetector(detect_fn), embedder=StubEmbedder(embed_fn),
                       wavlm=StubWavLM(wavlm_fn))
            except EnrollError:
                pass
            else:
                raise AssertionError("expected EnrollError below seed_min_ms")
        se = load_manifest(mp3)
        assert verify_chain(se) and se[-1]["operation"] == "layer1_failure"
        assert "seed_min" in se[-1]["payload"]["reason"]

    print("layer1_enrollment.py self-test: OK (click→tightest box, face match, "
          "clean-window selection, seed pooling, seed_min + click-fail guardrails, "
          "manifest chain)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
