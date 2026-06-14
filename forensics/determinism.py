"""
Forensic pipeline — forensics/determinism.py
=============================================

Purpose:    Make the pipeline bit-exact on re-runs. Sets every RNG seed
            (python `random`, numpy, torch + CUDA), requests deterministic
            algorithms, disables nondeterministic fast paths (TF32, cuDNN
            autotuner), and provides a run-twice self-check that the
            environment_gate logs to the manifest before any processing.

Honest stance on GPU determinism:
            `torch.use_deterministic_algorithms(True)` is set WITHOUT
            `warn_only`, on purpose: if an op lacks a deterministic CUDA
            kernel, torch RAISES. For a forensic system that is the correct
            behavior — a nondeterministic op must be a hard failure (caught by
            the gate), not a silent source of run-to-run drift. We also pin
            CUBLAS_WORKSPACE_CONFIG, required for deterministic cuBLAS GEMMs.

Lazy deps:  numpy/torch are imported INSIDE the functions, so this module
            imports and self-tests on plain Python with no pip installs. Where
            a dep is absent (e.g. the macOS dev box), seeding/flags for that
            dep are reported as skipped rather than failing.

Run / test: python3 forensics/determinism.py --selftest      (stdlib only)
"""
from __future__ import annotations

import os
import random
import sys

DEFAULT_SEED = 0
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def seed_all(seed=DEFAULT_SEED):
    """Seed python `random`, numpy and torch (CPU+CUDA) if present. Also exports
    PYTHONHASHSEED for any child processes. Returns a report dict for the
    manifest. (Note: PYTHONHASHSEED only affects processes started AFTER it is
    set; the pipeline runner should be launched with PYTHONHASHSEED=0 too — the
    setup guide documents this. In-process determinism does not rely on it
    because all JSON is emitted with sorted keys.)"""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    seeded = ["python_random"]
    try:
        import numpy as np
        np.random.seed(seed)
        seeded.append("numpy")
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seeded.append("torch+cuda")
        else:
            seeded.append("torch")
    except Exception:
        pass
    return {"seed": seed, "seeded": seeded, "pythonhashseed": str(seed)}


def configure_torch_determinism():
    """Enforce deterministic algorithms and disable nondeterministic fast
    paths. Always sets CUBLAS_WORKSPACE_CONFIG (a plain env var). Returns a
    report dict; status is 'skipped-no-torch' when torch is absent."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    try:
        import torch
    except Exception:
        return {"status": "skipped-no-torch",
                "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG}
    torch.use_deterministic_algorithms(True)          # raises on nondeterministic ops — intended
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    return {
        "status": "ok",
        "torch": torch.__version__,
        "use_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "tf32": False,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
    }


def enforce_determinism(seed=DEFAULT_SEED):
    """Full determinism setup: seed everything, then configure torch flags.
    Returns the merged report for the environment_gate to log."""
    return {"seeds": seed_all(seed), "flags": configure_torch_determinism()}


def determinism_selfcheck(seed=DEFAULT_SEED):
    """Run a fixed tensor through a fixed op twice and assert byte-identical
    output. On the dev box (no torch) returns 'skipped-no-torch'. On the GPU box
    a 'FAIL' here means the environment cannot guarantee bit-exactness and the
    gate must halt."""
    try:
        import hashlib
        import torch
    except Exception:
        return {"status": "skipped-no-torch"}

    def _run():
        seed_all(seed)
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(64, 64, generator=g)
        y = (x @ x.t()).relu()
        return hashlib.sha256(y.cpu().contiguous().numpy().tobytes()).hexdigest()

    h1, h2 = _run(), _run()
    return {"status": "ok" if h1 == h2 else "FAIL", "hash": h1, "match": h1 == h2}


# =============================================================================
# Self-test (stdlib only)
# =============================================================================

def _selftest():
    # python random reproducibility across reseed
    seed_all(123)
    a = [random.random() for _ in range(8)]
    rep = seed_all(123)
    b = [random.random() for _ in range(8)]
    assert a == b, "python random not reproducible after seed_all"
    assert rep["seed"] == 123 and "python_random" in rep["seeded"]
    assert os.environ.get("PYTHONHASHSEED") == "123"

    flags = configure_torch_determinism()
    assert flags["status"] in ("ok", "skipped-no-torch")
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_WORKSPACE_CONFIG

    merged = enforce_determinism(7)
    assert merged["seeds"]["seed"] == 7 and "flags" in merged

    sc = determinism_selfcheck(0)
    assert sc["status"] in ("ok", "FAIL", "skipped-no-torch"), sc
    # if torch is present it MUST be reproducible
    assert sc["status"] != "FAIL", "determinism self-check FAILED on this machine"

    print(f"determinism.py self-test: OK (seeding reproducible; torch check: "
          f"{sc['status']})")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
