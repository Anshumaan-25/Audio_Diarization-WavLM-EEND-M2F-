"""
SOTA SANDBOX — compare_results.py  (READ-ONLY A/B comparator)
============================================================

Purpose:    Line up the SOTA neural diarizer (sota_output.json, written by
            run_sota.py) against the Forensic Pipeline (FP) timeline pulled
            from a finished session manifest, and print a side-by-side
            terminal report + ASCII timeline showing exactly where the two
            AGREE and where they DISAGREE.

What "the FP timeline" means here:
            The FP enrolls ONE forensic subject (the target) and emits its
            verified clean speech as ``operation: layer3_segment`` records
            (decision = CLEAN). Those segments — pulled by their
            ``start_global_ms`` / ``end_global_ms`` — ARE the FP answer to
            "when is the target audibly, cleanly speaking?". The FP
            deliberately drops overlapped/contaminated target speech (it
            becomes NaN, not CLEAN), so it is conservative by construction.
            PyAnnote, by contrast, labels every speaker and keeps overlapped
            speech. The honest comparison is therefore: (a) which PyAnnote
            speaker best matches the FP target, and (b) where that speaker
            and the FP CLEAN timeline agree / diverge — with SOTA-only
            regions expected wherever the FP rejected contaminated audio.

Independence contract:
            Pure Python standard library. Imports NOTHING from the FP and
            writes NOTHING. The only coupling is the manifest on-disk
            FORMAT: the JSON-L read and the double-nested Layer 2/3
            worker-record unwrap are re-implemented here, read-only, and the
            schema string is sanity-checked (tolerantly). If the FP bumps
            its schema, we warn rather than mis-parse.

Schema note: This tool expects a neutral ``"<name>-manifest-v1"`` schema
            string. The FP's own manifests carry their pipeline name in that
            string; run ``sanitize_manifest.py`` first to produce a redacted
            copy if that name must not appear in study artifacts. The schema
            string is NOT used to locate segments (that keys off the
            ``operation`` field), so changing it never breaks parsing.

Run:        python3 compare_results.py --manifest <session.manifest.jsonl>
                [--sota sota_output.json] [--width 100] [--all-speakers]
Self-test:  python3 compare_results.py --selftest   (stdlib only — builds a
            synthetic double-nested manifest + synthetic SOTA output and
            asserts extraction, interval math, speaker matching)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MANIFEST_SCHEMA = "forensic-manifest-v1"     # neutral; sanitize_manifest.py emits this
WORKER_WRAPPER_KEYS = frozenset({"file_index", "start_ms", "operation", "payload"})
OP_SEGMENT = "layer3_segment"


# =============================================================================
# Interval algebra — half-open [start, end) integer-ms intervals
# =============================================================================

def merge(intervals):
    """Sort + coalesce overlapping/touching intervals."""
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


def intersect(a, b):
    """Intersection of two interval sets (both pre-merged or not)."""
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
    """a minus b (regions in a not covered by b)."""
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


def iou(a, b):
    inter = total_ms(intersect(a, b))
    uni = total_ms(union(a, b))
    return (inter / uni) if uni else 0.0


def pct(part, whole):
    return (100.0 * part / whole) if whole else 0.0


# =============================================================================
# Manifest loading + worker-record normalization
# =============================================================================

class CompareError(Exception):
    pass


def load_manifest(path, *, strict_schema=True):
    path = Path(path)
    if not path.is_file():
        raise CompareError(f"manifest not found: {path}")
    entries = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CompareError(f"{path}: line {n} is not valid JSON: {exc}")
    if not entries:
        raise CompareError(f"{path}: manifest is empty")
    # Accept the expected schema OR any same-family "*-manifest-v1" string.
    # We do NOT echo the raw schema value (it may carry a name that must not
    # appear in study artifacts) — the message stays generic.
    first = entries[0]
    schema = first.get("schema") if isinstance(first, dict) else None
    recognized = isinstance(schema, str) and (
        schema == MANIFEST_SCHEMA or schema.endswith("-manifest-v1"))
    if not recognized:
        msg = (f"{path}: first record is not a recognized '*-manifest-v1' "
               f"manifest (expected {MANIFEST_SCHEMA!r})")
        if strict_schema:
            raise CompareError(msg)
        print(f"WARNING: {msg} — continuing best-effort.", file=sys.stderr)
    return entries


def normalize_entry(entry):
    """Return (operation, real_payload). Layer 2/3 worker records are
    double-nested: entry['payload'] is itself {file_index, start_ms,
    operation, payload:{...}}; unwrap to the inner payload."""
    if not isinstance(entry, dict):
        return "", {}
    operation = entry.get("operation", "")
    payload = entry.get("payload")
    if (
        isinstance(payload, dict)
        and WORKER_WRAPPER_KEYS == set(payload.keys())
        and payload.get("operation") == operation
        and isinstance(payload.get("payload"), dict)
    ):
        return operation, payload["payload"]
    return operation, (payload if isinstance(payload, dict) else {})


def extract_forensic_target(entries):
    """Pull the target's CLEAN timeline from layer3_segment records.
    Returns (intervals, segment_dicts). Intervals are (start_global_ms,
    end_global_ms); segment_dicts keep the per-segment detail for the
    agreement table."""
    intervals, segs = [], []
    for entry in entries:
        op, payload = normalize_entry(entry)
        if op != OP_SEGMENT:
            continue
        if "start_global_ms" not in payload or "end_global_ms" not in payload:
            continue
        try:
            g0 = int(payload["start_global_ms"])
            g1 = int(payload["end_global_ms"])
            dur = int(payload.get("duration_ms", g1 - g0))
        except (TypeError, ValueError):
            continue          # malformed numeric fields — skip this record, don't crash
        if g1 <= g0:
            continue
        intervals.append((g0, g1))
        segs.append({
            "start_global_ms": g0,
            "end_global_ms": g1,
            "duration_ms": dur,
            "decision": payload.get("decision", "CLEAN"),
            "block_count": payload.get("block_count"),
            "bridged_gaps": payload.get("bridged_gaps", []),
        })
    segs.sort(key=lambda s: (s["start_global_ms"], s["end_global_ms"]))
    return merge(intervals), segs


# =============================================================================
# SOTA output loading
# =============================================================================

def load_sota(path):
    path = Path(path)
    if not path.is_file():
        raise CompareError(f"sota output not found: {path} "
                           "(run run_sota.py first)")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompareError(f"{path}: not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise CompareError(f"{path}: expected a JSON object at top level")
    by_speaker = {}
    for s in data.get("segments", []):
        try:
            a, b, spk = int(s["start_ms"]), int(s["end_ms"]), str(s["speaker_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CompareError(f"{path}: malformed segment {s!r}: {exc}")
        if b > a:
            by_speaker.setdefault(spk, []).append((a, b))
    for spk in by_speaker:
        by_speaker[spk] = merge(by_speaker[spk])
    return data, by_speaker


def match_target_speaker(fp_intervals, sota_by_speaker):
    """The SOTA speaker whose speech overlaps the FP target the most.
    Returns (speaker_id, overlap_ms) or (None, 0) if there is no SOTA."""
    best, best_ov = None, -1
    for spk, ivals in sorted(sota_by_speaker.items()):
        ov = total_ms(intersect(fp_intervals, ivals))
        if ov > best_ov:
            best, best_ov = spk, ov
    return best, max(best_ov, 0)


# =============================================================================
# Rendering
# =============================================================================

def _bounds(*interval_sets):
    los, his = [], []
    for s in interval_sets:
        for a, b in s:
            los.append(a)
            his.append(b)
    if not los:
        return 0, 1
    return min(los), max(his)


def render_lane(intervals, t0, t1, width, fill="#", empty="."):
    """Bucket [t0,t1) into `width` columns; a column is `fill` if any
    interval overlaps that column's time span."""
    if t1 <= t0:
        return empty * width
    span = t1 - t0
    cols = []
    merged = merge(intervals)
    for c in range(width):
        c0 = t0 + (span * c) // width
        c1 = t0 + (span * (c + 1)) // width
        if c1 <= c0:
            c1 = c0 + 1
        hit = any(a < c1 and b > c0 for a, b in merged)
        cols.append(fill if hit else empty)
    return "".join(cols)


def render_diff_lane(p_intervals, s_intervals, t0, t1, width):
    """=  both (agree) | P  FP-only | S  SOTA-only | space  neither."""
    if t1 <= t0:
        return " " * width
    span = t1 - t0
    p, s = merge(p_intervals), merge(s_intervals)
    cols = []
    for c in range(width):
        c0 = t0 + (span * c) // width
        c1 = t0 + (span * (c + 1)) // width
        if c1 <= c0:
            c1 = c0 + 1
        in_p = any(a < c1 and b > c0 for a, b in p)
        in_s = any(a < c1 and b > c0 for a, b in s)
        cols.append("=" if in_p and in_s else "P" if in_p else "S" if in_s else " ")
    return "".join(cols)


def fmt_ms(ms):
    ms = int(ms)
    sign = "-" if ms < 0 else ""
    ms = abs(ms)
    s, msr = divmod(ms, 1000)
    m, sr = divmod(s, 60)
    return f"{sign}{m:02d}:{sr:02d}.{msr:03d}"


def build_report(manifest_path, sota_path, *, width=100, show_all=False):
    width = max(1, int(width))            # guard --width 0/negative
    entries = load_manifest(manifest_path, strict_schema=False)
    p_intervals, p_segs = extract_forensic_target(entries)
    sota_data, sota_by_speaker = load_sota(sota_path)

    matched, _matched_ov = match_target_speaker(p_intervals, sota_by_speaker)
    s_intervals = sota_by_speaker.get(matched, []) if matched else []

    t0, t1 = _bounds(p_intervals, *sota_by_speaker.values())
    inter = intersect(p_intervals, s_intervals)
    p_only = subtract(p_intervals, s_intervals)
    s_only = subtract(s_intervals, p_intervals)

    lines = []
    add = lines.append
    bar = "=" * (width + 14)
    add(bar)
    add("  Forensic Pipeline  vs  SOTA neural diarizer — A/B comparison".upper())
    add(bar)
    add(f"  manifest : {manifest_path}")
    add(f"  sota     : {sota_path}")
    add(f"  model    : {sota_data.get('model', '?')}  "
        f"(device={sota_data.get('device', '?')}, "
        f"seed={sota_data.get('seed', '?')})")
    add(f"  session  : {fmt_ms(t0)} -> {fmt_ms(t1)}   "
        f"({fmt_ms(t1 - t0)} on the global clock)")
    add("")

    # --- headline numbers ---
    p_total = total_ms(p_intervals)
    s_total = total_ms(s_intervals)
    inter_ms = total_ms(inter)
    union_ms = total_ms(union(p_intervals, s_intervals))
    add("  SPEAKER INVENTORY (SOTA):")
    for spk, ivals in sorted(sota_by_speaker.items()):
        tag = "  <- matched to FP target" if spk == matched else ""
        add(f"    {spk:<14} {total_ms(ivals)/1000:7.1f}s  "
            f"{len(ivals):>3} turns{tag}")
    if not sota_by_speaker:
        add("    (no speakers in sota_output.json)")
    add("")
    add("  FP target (CLEAN segments):     "
        f"{p_total/1000:7.1f}s   {len(p_segs)} segments")
    add(f"  SOTA matched ({matched or '-'}):        "
        f"{s_total/1000:7.1f}s   {len(s_intervals)} turns")
    add("")
    add("  AGREEMENT:")
    add(f"    intersection (both)     {inter_ms/1000:7.1f}s")
    add(f"    union (either)          {union_ms/1000:7.1f}s")
    add(f"    IoU                     {iou(p_intervals, s_intervals):7.3f}")
    add(f"    FP-only   (P\\S)         {total_ms(p_only)/1000:7.1f}s   "
        f"-> SOTA missed / mis-attributed target speech")
    add(f"    SOTA-only (S\\P)         {total_ms(s_only)/1000:7.1f}s   "
        f"-> SOTA kept what the FP rejected (overlap/contamination)")
    add(f"    FP target covered by SOTA   {pct(inter_ms, p_total):5.1f}%")
    add(f"    SOTA covered by FP          {pct(inter_ms, s_total):5.1f}%")
    add("")

    # --- ASCII timeline (lanes share one computed label column) ---
    lanes = [("FP", render_lane(p_intervals, t0, t1, width, "#", "."))]
    lanes.append((f"SOTA {matched or '-'}",
                  render_lane(s_intervals, t0, t1, width, "#", ".")))
    if show_all:
        for spk, ivals in sorted(sota_by_speaker.items()):
            if spk == matched:
                continue
            lanes.append((spk, render_lane(ivals, t0, t1, width, "o", ".")))
    lanes.append(("DIFF",
                  render_diff_lane(p_intervals, s_intervals, t0, t1, width)))
    lw = max(len(label) for label, _ in lanes)
    rule = "  " + "-" * (width + lw + 3)
    add(f"  TIMELINE  ({fmt_ms(t0)} ... {fmt_ms(t1)}, {width} cols, "
        f"each col ~{max(1,(t1-t0)//max(width,1))} ms)")
    add(rule)
    for label, lane in lanes:
        add(f"  {label:<{lw}} | {lane}")
    add(rule)
    add("  legend:  '#' active   DIFF: '=' agree  'P' FP-only  "
        "'S' SOTA-only  ' ' silence")
    add("")

    # --- per-segment agreement table ---
    add("  PER-SEGMENT AGREEMENT (FP target vs matched SOTA speaker):")
    add(f"    {'#':>3}  {'start':>10}  {'end':>10}  {'dur':>8}  "
        f"{'covered':>8}  verdict")
    for i, seg in enumerate(p_segs, start=1):
        iv = [(seg["start_global_ms"], seg["end_global_ms"])]
        cov = total_ms(intersect(iv, s_intervals))
        frac = pct(cov, seg["duration_ms"])
        verdict = ("AGREE" if frac >= 80 else
                   "PARTIAL" if frac >= 20 else "MISS")
        add(f"    {i:>3}  {fmt_ms(seg['start_global_ms']):>10}  "
            f"{fmt_ms(seg['end_global_ms']):>10}  "
            f"{seg['duration_ms']/1000:7.1f}s  {frac:7.1f}%  {verdict}")
    if not p_segs:
        add("    (no layer3_segment / CLEAN records found in the manifest)")
    add("")
    add(bar)
    return "\n".join(lines)


# =============================================================================
# Self-test (stdlib only)
# =============================================================================

def _wrap_layer3(file_index, payload):
    """Reproduce the FP's double-nested worker-record shape so the self-test
    exercises the real unwrap path."""
    inner = {"file_index": file_index, "start_ms": payload["start_local_ms"],
             "operation": OP_SEGMENT, "payload": payload}
    return {"operation": OP_SEGMENT, "payload": inner}


def _selftest():
    # interval algebra
    assert merge([(0, 10), (5, 20), (30, 40)]) == [(0, 20), (30, 40)]
    assert total_ms([(0, 10), (5, 20)]) == 20
    assert intersect([(0, 10)], [(5, 15)]) == [(5, 10)]
    assert subtract([(0, 10)], [(3, 6)]) == [(0, 3), (6, 10)]
    assert union([(0, 5)], [(5, 10)]) == [(0, 10)]
    assert abs(iou([(0, 10)], [(5, 15)]) - (5 / 15)) < 1e-9

    # manifest parse: schema record + a top-level + a double-nested layer3
    manifest = [
        {"schema": MANIFEST_SCHEMA, "operation": "session_start", "payload": {}},
        {"operation": "layer0_file", "payload": {"file_index": 0}},
        _wrap_layer3(0, {"decision": "CLEAN", "start_local_ms": 1000,
                         "end_local_ms": 3000, "start_global_ms": 1000,
                         "end_global_ms": 3000, "duration_ms": 2000,
                         "block_count": 4, "bridged_gaps": []}),
        _wrap_layer3(0, {"decision": "CLEAN", "start_local_ms": 5000,
                         "end_local_ms": 6000, "start_global_ms": 5000,
                         "end_global_ms": 6000, "duration_ms": 1000,
                         "block_count": 2, "bridged_gaps": []}),
    ]
    p_intervals, p_segs = extract_forensic_target(manifest)
    assert p_intervals == [(1000, 3000), (5000, 6000)], p_intervals
    assert len(p_segs) == 2 and p_segs[0]["block_count"] == 4

    # synthetic SOTA: SPEAKER_00 ~ target, SPEAKER_01 the other voice
    sota_by_speaker = {
        "SPEAKER_00": merge([(900, 3100), (4800, 5500)]),     # matches target
        "SPEAKER_01": merge([(3200, 4500), (6500, 8000)]),    # interviewer
    }
    matched, ov = match_target_speaker(p_intervals, sota_by_speaker)
    assert matched == "SPEAKER_00", matched
    assert ov == total_ms(intersect(p_intervals, sota_by_speaker["SPEAKER_00"]))

    # rendering produces correct-width lanes and a non-empty diff
    lane = render_lane(p_intervals, 1000, 6000, 50)
    assert len(lane) == 50 and "#" in lane and "." in lane
    diff = render_diff_lane(p_intervals, sota_by_speaker["SPEAKER_00"],
                            1000, 6000, 50)
    assert set(diff) <= set("=PS ") and "=" in diff

    assert fmt_ms(61234) == "01:01.234", fmt_ms(61234)
    print("compare_results.py self-test: OK (parse + interval math + match + "
          "render verified, stdlib only)")
    return 0


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Side-by-side A/B report: SOTA diarizer vs Forensic "
                    "Pipeline timeline (read-only, stdlib only).")
    p.add_argument("--manifest", default="session.manifest.jsonl",
                   help="finished FP session manifest (.manifest.jsonl)")
    p.add_argument("--sota", default="sota_output.json",
                   help="SOTA output written by run_sota.py")
    p.add_argument("--width", type=int, default=100,
                   help="ASCII timeline width in columns (default: 100)")
    p.add_argument("--all-speakers", action="store_true",
                   help="render a lane for every SOTA speaker, not just the "
                        "one matched to the FP target")
    p.add_argument("--selftest", action="store_true",
                   help="run stdlib-only self-test and exit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    try:
        print(build_report(args.manifest, args.sota,
                           width=args.width, show_all=args.all_speakers))
    except CompareError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
