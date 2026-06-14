"""
Forensic pipeline — forensics/pts.py
====================================

Purpose:    The pipeline's single source of truth for TIME. Every timestamp in
            the system is an INTEGER number of milliseconds derived from
            Presentation TimeStamps (PTS). Floating-point seconds are never
            used internally; the one boundary helper that accepts float seconds
            (`ms_from_seconds`) exists only to convert model outputs at the
            edge and is documented as such.

            Provides:
              - exact integer conversions (PTS↔ms, samples↔ms),
              - a GlobalClock that lays the session's clips end-to-end on one
                monotonic integer-ms timeline (offset per file, canonical order),
              - integer half-open [start, end) interval algebra (merge /
                intersect / subtract / total / overlaps) used by VAD (Layer 0)
                and the overlap-discard policy (Layer 3).

Determinism: All arithmetic is integer. Rounding rules are fixed once here
            (round-half-up for ms conversions) so re-runs are bit-identical.

Run / test:  python3 forensics/pts.py --selftest      (stdlib only)
"""
from __future__ import annotations

import sys

MS_PER_SECOND = 1000


# =============================================================================
# Exact integer time conversions
# =============================================================================

def ms_from_pts(pts, tb_num, tb_den):
    """Convert a PTS value (integer ticks) under time base tb_num/tb_den
    (seconds-per-tick) to integer milliseconds, round-half-up.

    ms = pts * tb_num / tb_den * 1000, computed in integers."""
    pts, tb_num, tb_den = int(pts), int(tb_num), int(tb_den)
    if tb_den <= 0:
        raise ValueError(f"time-base denominator must be positive, got {tb_den}")
    return (pts * tb_num * MS_PER_SECOND + tb_den // 2) // tb_den


def samples_to_ms(n_samples, sample_rate):
    """Sample count -> integer ms (round-half-up)."""
    n_samples, sample_rate = int(n_samples), int(sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    return (n_samples * MS_PER_SECOND + sample_rate // 2) // sample_rate


def ms_to_samples(ms, sample_rate):
    """Integer ms -> sample index on the sample grid (floor). Used for slicing
    wavs as [start_sample, end_sample); floor on both ends keeps slices on a
    consistent grid and never reads past the buffer."""
    return int(ms) * int(sample_rate) // MS_PER_SECOND


def ms_from_seconds(seconds):
    """BOUNDARY USE ONLY. Convert float seconds (e.g. a model's frame time) to
    integer ms, round-half-to-even via int(round()). Internal pipeline time is
    always integer ms — do not use this for anything except ingesting an
    external float at the edge."""
    return int(round(float(seconds) * 1000.0))


# =============================================================================
# Global session clock — clips laid end-to-end in canonical order
# =============================================================================

class GlobalClock:
    """Maps (file_index, local_ms) -> a single monotonic global-ms timeline.
    file_durations_ms is the per-clip duration in CANONICAL FILE ORDER."""

    def __init__(self, file_durations_ms):
        self.offsets = []
        acc = 0
        for d in file_durations_ms:
            d = int(d)
            if d < 0:
                raise ValueError(f"negative file duration: {d}")
            self.offsets.append(acc)
            acc += d
        self.total_ms = acc

    def to_global(self, file_index, local_ms):
        return self.offsets[int(file_index)] + int(local_ms)

    def to_global_interval(self, file_index, start_ms, end_ms):
        off = self.offsets[int(file_index)]
        return (off + int(start_ms), off + int(end_ms))


# =============================================================================
# Integer half-open [start, end) interval algebra
# =============================================================================

def merge(intervals):
    """Sort + coalesce overlapping/touching integer intervals."""
    ordered = sorted((int(a), int(b)) for a, b in intervals if int(b) > int(a))
    out = []
    for a, b in ordered:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def total_ms(intervals):
    return sum(b - a for a, b in merge(intervals))


def overlaps(a, b):
    """True if two single intervals share any positive-length span."""
    return a[0] < b[1] and b[0] < a[1]


def intersect(a, b):
    a, b = merge(a), merge(b)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a, b):
    """a minus b (regions of a not covered by b)."""
    a, b = merge(a), merge(b)
    out = []
    for lo, hi in a:
        cursor = lo
        for blo, bhi in b:
            if bhi <= cursor or blo >= hi:
                continue
            if blo > cursor:
                out.append((cursor, min(blo, hi)))
            cursor = max(cursor, bhi)
            if cursor >= hi:
                break
        if cursor < hi:
            out.append((cursor, hi))
    return merge(out)


def union(a, b):
    return merge(list(a) + list(b))


# =============================================================================
# Self-test (stdlib only)
# =============================================================================

def _selftest():
    # exact conversions
    assert ms_from_pts(90000, 1, 90000) == 1000, ms_from_pts(90000, 1, 90000)
    assert ms_from_pts(0, 1, 1000) == 0
    assert ms_from_pts(1, 1, 1000) == 1          # tb 1/1000 -> pts already ms
    assert samples_to_ms(16000, 16000) == 1000
    assert samples_to_ms(8000, 16000) == 500
    assert ms_to_samples(1000, 16000) == 16000
    assert ms_to_samples(500, 16000) == 8000
    assert ms_from_seconds(1.2345) == 1234

    # global clock
    gc = GlobalClock([10000, 5000, 7000])
    assert gc.offsets == [0, 10000, 15000] and gc.total_ms == 22000
    assert gc.to_global(1, 250) == 10250
    assert gc.to_global_interval(2, 100, 200) == (15100, 15200)

    # interval algebra
    assert merge([(0, 10), (10, 20), (30, 40)]) == [(0, 20), (30, 40)]  # touching merges
    assert total_ms([(0, 10), (5, 20)]) == 20
    assert overlaps((0, 10), (5, 15)) and not overlaps((0, 10), (10, 20))
    assert intersect([(0, 10)], [(5, 15)]) == [(5, 10)]
    assert subtract([(0, 10)], [(3, 6)]) == [(0, 3), (6, 10)]
    assert union([(0, 5)], [(5, 10)]) == [(0, 10)]

    print("pts.py self-test: OK (integer time, global clock, intervals)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
