"""
Forensic substrate for the WavLM + EEND-M2F diarization pipeline.

Three stdlib-only primitives every layer depends on:
  - pts.py          integer-millisecond time + global session clock + intervals
  - determinism.py  seed-all / deterministic-algorithms / run-twice self-check
  - manifest.py     append-only, SHA-256 hash-chained audit log

Nothing here imports torch/numpy at module load (heavy deps are lazy), so the
whole substrate is importable and self-testable on plain Python 3.10 with zero
pip installs — real model execution happens only on the Ubuntu GPU box.
"""
