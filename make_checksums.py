"""
Forensic pipeline — make_checksums.py
======================================

Generate (or verify) the gate's weight registry, models/checksums.json, without
hand-hashing anything. Stdlib only — no torch, runs anywhere.

The gate (environment_gate.verify_checksums) expects:

    { "<logical-name>": {"path": "<relative/to/models-dir>", "sha256": "<hex>"}, ... }

and REQUIRES the four logical names wavlm, eend_m2f, yolov8, insightface.

Two ways to use it:

  A) From a skeleton you write once (paths only, sha256 blank/"") — fill hashes:
        python3 make_checksums.py --models-dir ./models --from-skeleton skeleton.json
     where skeleton.json is e.g.
        { "wavlm":       {"path": "wavlm/pytorch_model.bin"},
          "eend_m2f":    {"path": "eend_m2f/model.pt"},
          "yolov8":      {"path": "yolov8.pt"},
          "insightface": {"path": "insightface/det_10g.onnx"} }

  B) Re-hash an existing checksums.json in place (e.g. after restaging a weight):
        python3 make_checksums.py --models-dir ./models --update

  Verify only (no writes; exits non-zero on any mismatch/missing):
        python3 make_checksums.py --models-dir ./models --verify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forensics.manifest import sha256_file

REQUIRED = ("wavlm", "eend_m2f", "yolov8", "insightface")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fill(registry, models_dir):
    """Return a new registry with sha256 computed from models_dir/<path>."""
    models_dir = Path(models_dir)
    out = {}
    for name in sorted(registry):
        rel = registry[name]["path"]
        fpath = models_dir / rel
        if not fpath.is_file():
            raise FileNotFoundError(f"{name}: weight not found at {fpath}")
        out[name] = {"path": rel, "sha256": sha256_file(fpath)}
    return out


def verify(registry, models_dir):
    """Return (ok, problems). Checks presence, hash match, and required names."""
    models_dir = Path(models_dir)
    problems = []
    for name in sorted(registry):
        rel = registry[name]["path"]
        want = str(registry[name].get("sha256", "")).lower()
        fpath = models_dir / rel
        if not fpath.is_file():
            problems.append(f"MISSING  {name}: {fpath}")
            continue
        got = sha256_file(fpath).lower()
        if not want:
            problems.append(f"NO-HASH  {name}: registry sha256 is blank")
        elif got != want:
            problems.append(f"MISMATCH {name}: want {want[:16]}… got {got[:16]}…")
    for r in REQUIRED:
        if r not in registry:
            problems.append(f"REQUIRED missing logical name: {r}")
    return (not problems), problems


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate/verify models/checksums.json")
    p.add_argument("--models-dir", required=True)
    p.add_argument("--out", help="output registry (default: <models-dir>/checksums.json)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-skeleton", help="skeleton JSON (paths only) → fill hashes")
    g.add_argument("--update", action="store_true", help="re-hash existing registry in place")
    g.add_argument("--verify", action="store_true", help="verify only, no writes")
    args = p.parse_args(argv)

    models_dir = Path(args.models_dir)
    out = Path(args.out) if args.out else models_dir / "checksums.json"

    if args.verify:
        ok, problems = verify(_load(out), models_dir)
        for line in problems:
            print(line, file=sys.stderr)
        print("checksums: OK" if ok else "checksums: FAILED", file=sys.stderr)
        return 0 if ok else 1

    src = _load(args.from_skeleton) if args.from_skeleton else _load(out)
    registry = fill(src, models_dir)
    missing = [r for r in REQUIRED if r not in registry]
    if missing:
        print(f"WARNING: registry is missing required names: {missing} "
              f"(the gate will halt on these)", file=sys.stderr)
    out.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"wrote {out} with {len(registry)} verified entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
