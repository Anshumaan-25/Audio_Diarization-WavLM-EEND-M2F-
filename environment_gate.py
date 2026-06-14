"""
Forensic pipeline — environment_gate.py
========================================

Purpose:    The Pre-Flight gate. Runs ONCE before any audio touches any model
            and refuses to proceed unless the environment is forensically
            sound:
              1. Air-gap: export HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE etc. so
                 no model code can reach the network.
              2. Determinism: seed everything, enforce deterministic algorithms
                 (forensics.determinism), and run the run-twice self-check.
              3. Model integrity: SHA-256-verify EVERY pre-staged weight
                 (WavLM, EEND-M2F, YOLOv8, InsightFace) against a checked-in
                 registry (models/checksums.json); ANY mismatch/missing file
                 halts the run.
              4. Provenance: record the full environment, determinism report
                 and per-model checksum results into the manifest as its
                 opening records — before Layer 0 runs.

Air-gap note: offline env vars must be set BEFORE huggingface/transformers are
            imported anywhere, so the gate is the very first thing the runner
            calls. Verifying weights by our OWN sha256 (not the hub cache) is
            what makes "pre-staged + offline" auditable.

Run / test: python3 environment_gate.py --selftest      (stdlib only — uses a
            synthetic models dir + checksums.json; the torch determinism check
            reports 'skipped-no-torch' on the dev box)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from forensics.determinism import enforce_determinism, determinism_selfcheck
from forensics.manifest import ManifestWriter, sha256_file

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


class GateError(Exception):
    """Raised when the environment is not forensically sound — halts the run."""


def enforce_offline(environ=None):
    """Export the air-gap env vars. Returns the dict that was set."""
    import os
    environ = os.environ if environ is None else environ
    for k, v in OFFLINE_ENV.items():
        environ[k] = v
    return dict(OFFLINE_ENV)


def verify_checksums(checksums_path, models_dir):
    """Verify every weight listed in checksums.json against its SHA-256.

    Registry format:
        { "wavlm":      {"path": "wavlm/model.bin",      "sha256": "<hex>"},
          "eend_m2f":   {"path": "eend_m2f/model.pt",    "sha256": "<hex>"},
          "yolov8":     {"path": "yolo/yolov8.pt",       "sha256": "<hex>"},
          "insightface":{"path": "insightface/model.onnx","sha256": "<hex>"} }

    Returns a list of per-model result dicts. Raises GateError on the first
    missing file or hash mismatch."""
    checksums_path, models_dir = Path(checksums_path), Path(models_dir)
    if not checksums_path.is_file():
        raise GateError(f"checksum registry not found: {checksums_path}")
    try:
        registry = json.loads(checksums_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateError(f"{checksums_path}: invalid JSON: {exc}")
    if not isinstance(registry, dict) or not registry:
        raise GateError(f"{checksums_path}: registry must be a non-empty object")

    results = []
    for name in sorted(registry):                       # sorted -> deterministic order
        spec = registry[name]
        try:
            rel, expected = spec["path"], str(spec["sha256"]).lower()
        except (TypeError, KeyError) as exc:
            raise GateError(f"{checksums_path}: entry {name!r} missing path/sha256: {exc}")
        fpath = models_dir / rel
        if not fpath.is_file():
            raise GateError(f"model weight missing: {name} -> {fpath}")
        actual = sha256_file(fpath).lower()
        ok = actual == expected
        results.append({"name": name, "path": rel, "expected": expected,
                        "actual": actual, "ok": ok})
        if not ok:
            raise GateError(
                f"checksum MISMATCH for {name} ({rel}): expected {expected[:16]}…, "
                f"got {actual[:16]}… — refusing to run on unverified weights")
    return results


def run_gate(writer, *, models_dir, checksums_path, seed=0,
             expected_models=None):
    """Execute the full gate and record it to the manifest. `writer` is the
    pipeline's single ManifestWriter. Returns a gate report dict.

    expected_models: optional set of names that MUST be present in the registry
    (e.g. {"wavlm","eend_m2f","yolov8","insightface"}) — a guard against an
    under-specified registry."""
    offline = enforce_offline()
    det = enforce_determinism(seed)
    selfcheck = determinism_selfcheck(seed)
    if selfcheck["status"] == "FAIL":
        raise GateError("determinism self-check FAILED — environment cannot "
                        "guarantee bit-exact output; halting.")

    checksums = verify_checksums(checksums_path, models_dir)
    present = {c["name"] for c in checksums}
    if expected_models and not set(expected_models).issubset(present):
        missing = sorted(set(expected_models) - present)
        raise GateError(f"checksum registry is missing required models: {missing}")

    report = {
        "offline_env": offline,
        "determinism": det,
        "determinism_selfcheck": selfcheck,
        "models_verified": [{"name": c["name"], "sha256": c["actual"]}
                            for c in checksums],
    }
    # Provenance: opening records, written before Layer 0.
    writer.append("environment", {
        "offline_env": offline,
        "determinism": det,
        "determinism_selfcheck": selfcheck,
    })
    writer.append("model_checksums", {"models": report["models_verified"]})
    return report


# =============================================================================
# Self-test (stdlib only)
# =============================================================================

def _selftest():
    import os
    import tempfile
    from forensics.manifest import load_manifest, verify_chain

    # offline env vars get set
    env = {}
    enforce_offline(env)
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"

    with tempfile.TemporaryDirectory() as d:
        models = Path(d) / "models"
        (models / "wavlm").mkdir(parents=True)
        (models / "eend").mkdir(parents=True)
        w1 = models / "wavlm" / "model.bin"
        w2 = models / "eend" / "model.pt"
        w1.write_bytes(b"fake-wavlm-weights")
        w2.write_bytes(b"fake-eend-weights")
        registry = {
            "wavlm": {"path": "wavlm/model.bin", "sha256": sha256_file(w1)},
            "eend_m2f": {"path": "eend/model.pt", "sha256": sha256_file(w2)},
        }
        reg_path = Path(d) / "checksums.json"
        reg_path.write_text(json.dumps(registry, indent=2))

        # happy path verifies
        results = verify_checksums(reg_path, models)
        assert all(r["ok"] for r in results) and len(results) == 2

        # missing required model -> GateError
        try:
            mp = Path(d) / "m.jsonl"
            with ManifestWriter(mp, fsync=False) as w:
                run_gate(w, models_dir=models, checksums_path=reg_path, seed=0,
                         expected_models={"wavlm", "eend_m2f", "yolov8"})
        except GateError:
            pass
        else:
            raise AssertionError("expected GateError for missing required model")

        # full gate records to a verifiable manifest
        mp = Path(d) / "session.manifest.jsonl"
        with ManifestWriter(mp, fsync=False) as w:
            w.append("session_start", {"seed": 0})
            report = run_gate(w, models_dir=models, checksums_path=reg_path,
                              seed=0, expected_models={"wavlm", "eend_m2f"})
        assert len(report["models_verified"]) == 2
        entries = load_manifest(mp)
        assert verify_chain(entries) is True
        ops = [e["operation"] for e in entries]
        assert ops == ["session_start", "environment", "model_checksums"], ops

        # tampered weight -> checksum mismatch halts
        w1.write_bytes(b"TAMPERED")
        try:
            verify_checksums(reg_path, models)
        except GateError:
            pass
        else:
            raise AssertionError("tampered weight was not detected")

    print("environment_gate.py self-test: OK (offline env, checksum verify, "
          "missing/tamper halts, manifest provenance)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
