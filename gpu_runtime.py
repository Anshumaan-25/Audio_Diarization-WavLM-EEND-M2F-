"""
Forensic pipeline — gpu_runtime.py
===================================

Real-deployment runtime support for the Ubuntu RTX 6000 Ada box. NONE of this is
needed by the stdlib self-tests — it is imported only by the real adapters
(`build_real_adapters`), so it keeps `torch` lazy and never touches the pure
logic or the stubs.

Two jobs:

  * `AudioCache` — load each 16 kHz mono wav into RAM **once** and serve every
    subsequent window/interval slice from memory (no redundant disk re-reads
    across Layer 1's seed windows, Layer 2's per-window re-ID, etc.). With 512 GB
    RAM and ~15 short clips this is free and removes the pipeline's only hot I/O
    path. Slices are returned as `torch` tensors, optionally zero-padded up to a
    minimum length (guards WavLM's convolutional stack against sub-receptive-field
    inputs) and optionally batched (equal-length window batches stack cleanly;
    ragged re-ID intervals are right-padded with a returned length vector so the
    consumer can mask the padding out of any pooling).

  * `pick_device` / `device_index` — small helpers to resolve a torch device and
    its CUDA ordinal (for InsightFace's `ctx_id`).

Determinism: reads are byte-exact (stdlib `wave`), padding is deterministic
zeros, and batching preserves input order. The cache changes *where* samples come
from (RAM vs disk), never their values.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# WavLM's feature encoder downsamples 16 kHz audio by 320 (→ 50 fps) and needs a
# few hundred samples to clear its stacked convolutions. We never feed it less
# than this; short slices are zero-padded up to it (the pad is masked out of
# pooling). 640 samples = 40 ms @ 16 kHz — comfortably above the receptive field.
WAVLM_CONV_STRIDE = 320
MIN_WAVLM_SAMPLES = 640


def pick_device(prefer="cuda"):
    """Resolve a torch device string, falling back to CPU if CUDA is absent."""
    import torch
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return prefer
    return "cpu"


def device_index(device):
    """CUDA ordinal for a device string ('cuda'->0, 'cuda:2'->2, 'cpu'->-1).
    InsightFace's FaceAnalysis uses this as ctx_id."""
    if not str(device).startswith("cuda"):
        return -1
    s = str(device)
    return int(s.split(":", 1)[1]) if ":" in s else 0


class AudioCache:
    """In-RAM store of whole-file PCM, serving padded/batched slices as tensors.

    Thread-safe enough for the deterministic single-threaded GPU stages: the only
    mutation is first-touch population of a path, and slices never mutate stored
    data. `min_samples` is the zero-pad floor applied to every returned slice."""

    def __init__(self, *, sr=16000, min_samples=MIN_WAVLM_SAMPLES):
        self._sr = sr
        self._min = int(min_samples)
        self._mem: Dict[str, "object"] = {}      # path -> 1-D float32 CPU tensor

    # ---- population ---------------------------------------------------------
    def _whole(self, wav_path):
        import array
        import wave
        import torch
        key = str(wav_path)
        t = self._mem.get(key)
        if t is None:
            with wave.open(key, "rb") as w:
                if w.getsampwidth() != 2 or w.getnchannels() != 1:
                    raise ValueError(f"{key}: AudioCache expects 16-bit mono PCM")
                frames = w.readframes(w.getnframes())
            pcm = array.array("h")
            pcm.frombytes(frames)
            t = torch.tensor(pcm, dtype=torch.float32) / 32768.0   # [N], CPU
            self._mem[key] = t
        return t

    def preload(self, wav_paths):
        """Eagerly pull a batch of files into RAM (call once after Layer 0)."""
        for p in wav_paths:
            self._whole(p)

    # ---- slicing ------------------------------------------------------------
    def _bounds(self, start_ms, end_ms):
        from forensics.pts import ms_to_samples
        a = ms_to_samples(start_ms, self._sr)
        b = ms_to_samples(end_ms, self._sr)
        return a, max(a, b)

    def slice(self, wav_path, start_ms, end_ms, *, device="cpu", min_samples=None):
        """One slice as a [1, T] float32 tensor on `device`, zero-padded up to the
        minimum length. Returns (tensor, valid_samples)."""
        import torch
        t = self._whole(wav_path)
        a, b = self._bounds(start_ms, end_ms)
        seg = t[a:b]
        valid = int(seg.numel())
        floor = self._min if min_samples is None else int(min_samples)
        if seg.numel() < floor:
            seg = torch.nn.functional.pad(seg, (0, floor - seg.numel()))
        return seg.unsqueeze(0).to(device), valid

    def batch(self, wav_path, intervals, *, device="cpu", min_samples=None):
        """Right-padded batch for a list of (start_ms, end_ms). Returns
        (tensor [B, Tmax], lengths [B] valid samples). Equal-length intervals
        (e.g. fixed windows) stack with no padding; ragged intervals are padded to
        the longest and the lengths let the consumer mask the tails out of pooling."""
        import torch
        if not intervals:
            return torch.empty(0, 0, device=device), torch.empty(0, dtype=torch.long)
        t = self._whole(wav_path)
        segs, lengths = [], []
        floor = self._min if min_samples is None else int(min_samples)
        for s, e in intervals:
            a, b = self._bounds(s, e)
            seg = t[a:b]
            lengths.append(int(seg.numel()))
            if seg.numel() < floor:
                seg = torch.nn.functional.pad(seg, (0, floor - seg.numel()))
            segs.append(seg)
        tmax = max(s.numel() for s in segs)
        segs = [torch.nn.functional.pad(s, (0, tmax - s.numel())) if s.numel() < tmax
                else s for s in segs]
        batch = torch.stack(segs, dim=0).to(device)              # [B, Tmax]
        lens = torch.tensor(lengths, dtype=torch.long)
        return batch, lens

    @staticmethod
    def feature_frames(valid_samples, n_frames):
        """Number of WavLM output frames that correspond to `valid_samples` of
        input (clamped to [1, n_frames]) — used to mask padding out of mean-pooling."""
        f = valid_samples // WAVLM_CONV_STRIDE
        return max(1, min(int(n_frames), int(f)))


# =============================================================================
# Self-test (stdlib only — exercises the pure helpers; torch paths run on Ubuntu)
# =============================================================================

def _selftest():
    assert device_index("cuda") == 0
    assert device_index("cuda:3") == 3
    assert device_index("cpu") == -1
    # feature-frame masking: clamp to [1, n_frames]; floor-divide by stride
    assert AudioCache.feature_frames(0, 10) == 1
    assert AudioCache.feature_frames(320, 10) == 1
    assert AudioCache.feature_frames(3200, 10) == 10
    assert AudioCache.feature_frames(3200, 5) == 5         # clamped to available
    assert AudioCache.feature_frames(1600, 10) == 5
    print("gpu_runtime.py self-test: OK (device index, feature-frame masking; "
          "torch/RAM-cache paths exercised on the Ubuntu GPU box)")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
