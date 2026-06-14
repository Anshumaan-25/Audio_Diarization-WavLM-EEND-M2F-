"""
Forensic pipeline — forensics/manifest.py
==========================================

Purpose:    The append-only, SHA-256 hash-chained audit log
            (`session.manifest.jsonl`). Every parameter, discard, guardrail
            failure and output checksum is recorded as one JSON-Lines record,
            written BEFORE any destructive op. Each record is chained to its
            predecessor so any tampering (edit/insert/delete/reorder) breaks
            verification loudly.

On-disk format (schema "forensic-manifest-v1"):
            One canonical-JSON object per line. Fields:
              schema          (first record only)
              seq             monotonic int starting at 0
              operation       string op name
              payload         dict (see worker-nesting below)
              audit           {"timestamp_utc": "<ISO-8601 UTC>"} — chain of custody
              prev_sha256     entry_sha256 of the previous record (genesis = 64 zeros)
              payload_sha256  sha256(canonical(payload))                  ← CONTENT integrity
              entry_sha256    sha256(prev_sha256 + payload_sha256
                                     + str(seq) + operation + canonical(audit))  ← CUSTODY
            Layer-2/3 worker records are DOUBLE-NESTED: payload =
            {file_index, start_ms, operation, payload:{...}} so per-file
            provenance travels with the record — the exact shape the
            sota_sandbox comparator already understands.

Integrity vs custody (the two-hash design):
            payload_sha256 hashes ONLY the payload, so it is bit-reproducible
            across re-runs — you can prove the *content* of run N equals run M.
            The wall-clock `timestamp_utc` lives in the `audit` block and is
            folded into entry_sha256, so the *chain* captures the real-world
            time of execution. Tampering with a recorded timestamp therefore
            breaks the chain — custody is protected, not traded away. The
            clock is injectable for deterministic testing.

Run / test: python3 forensics/manifest.py --selftest      (stdlib only)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "forensic-manifest-v1"
GENESIS_PREV = "0" * 64
WORKER_WRAPPER_KEYS = frozenset({"file_index", "start_ms", "operation", "payload"})


class ManifestError(Exception):
    """Raised on a broken/tampered hash chain or a malformed manifest."""


def canonical(obj):
    """Deterministic JSON encoding (sorted keys, compact, UTF-8 preserved)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def utc_now_iso_ms():
    """ISO-8601 UTC timestamp with MILLISECOND precision (3 decimals) + 'Z' —
    the legacy pipeline's exact audit format. The recipe is fixed so custody
    timestamps are byte-formatted identically to the legacy chain."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _entry_hash(prev_sha, payload_sha, seq, operation, audit):
    """The chain link — binds previous link, content hash, position, op name,
    and the audit block (which carries the execution timestamp)."""
    return sha256_hex(prev_sha + payload_sha + str(seq) + operation + canonical(audit))


class ManifestWriter:
    """Single-writer, append-only hash-chained manifest. The pipeline runner
    owns ONE instance; layers hand it records (in canonical order).

    `clock` is a no-arg callable returning the timestamp_utc string; it defaults
    to real UTC time and is injectable so tests are deterministic."""

    def __init__(self, path, *, schema=SCHEMA, fsync=True, clock=utc_now_iso_ms):
        self.path = Path(path)
        self.schema = schema
        self.fsync = fsync
        self.clock = clock
        self.seq = 0
        self.prev = GENESIS_PREV
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def append(self, operation, payload, *, file_index=None, start_ms=None):
        """Append one record. If file_index is given the payload is wrapped in
        the double-nested worker shape. Returns the written record dict."""
        if file_index is not None:
            outer = {
                "file_index": int(file_index),
                "start_ms": int(start_ms if start_ms is not None else 0),
                "operation": operation,
                "payload": payload,
            }
        else:
            outer = payload

        audit = {"timestamp_utc": self.clock()}
        payload_sha = sha256_hex(canonical(outer))
        entry_sha = _entry_hash(self.prev, payload_sha, self.seq, operation, audit)
        record = {
            "seq": self.seq,
            "operation": operation,
            "payload": outer,
            "audit": audit,
            "prev_sha256": self.prev,
            "payload_sha256": payload_sha,
            "entry_sha256": entry_sha,
        }
        if self.seq == 0:
            record["schema"] = self.schema

        self._fh.write(canonical(record) + "\n")
        self._fh.flush()
        if self.fsync:
            os.fsync(self._fh.fileno())

        self.seq += 1
        self.prev = entry_sha
        return record

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def load_manifest(path):
    """Read a manifest into a list of records (one JSON object per line)."""
    path = Path(path)
    entries = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}: line {n} is not valid JSON: {exc}")
    if not entries:
        raise ManifestError(f"{path}: manifest is empty")
    return entries


def verify_chain(entries):
    """Re-walk the chain; raise ManifestError on the first break. Recomputes
    payload_sha256 (content) AND entry_sha256 (which binds the audit timestamp),
    so altering a recorded execution time breaks verification."""
    if entries and isinstance(entries[0], dict):
        schema = entries[0].get("schema")
        if schema is not None and schema != SCHEMA and not str(schema).endswith("-manifest-v1"):
            raise ManifestError(f"unexpected manifest schema {schema!r}")
    prev = GENESIS_PREV
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise ManifestError(f"record {i} is not a JSON object")
        op = e.get("operation", "")
        audit = e.get("audit", {})
        payload_sha = sha256_hex(canonical(e.get("payload")))
        if payload_sha != e.get("payload_sha256"):
            raise ManifestError(f"seq {e.get('seq')}: payload hash mismatch (tampered content)")
        if e.get("prev_sha256") != prev:
            raise ManifestError(f"seq {e.get('seq')}: prev_sha256 mismatch (chain broken)")
        entry_sha = _entry_hash(prev, payload_sha, e.get("seq"), op, audit)
        if entry_sha != e.get("entry_sha256"):
            raise ManifestError(f"seq {e.get('seq')}: entry hash mismatch "
                                "(tampered record, timestamp, or order)")
        if e.get("seq") != i:
            raise ManifestError(f"seq out of order at index {i}: {e.get('seq')}")
        prev = entry_sha
    return True


# =============================================================================
# Self-test (stdlib only)
# =============================================================================

def _build(path, clock):
    with ManifestWriter(path, fsync=False, clock=clock) as w:
        w.append("session_start", {"seed": 0, "files": 3})
        w.append("environment", {"offline": True})
        w.append("layer3_segment",
                 {"decision": "CLEAN", "start_global_ms": 1000,
                  "end_global_ms": 3000, "duration_ms": 2000},
                 file_index=0, start_ms=1000)


def _selftest():
    import re
    import tempfile

    # legacy ms-precision format: exactly 3 decimals + Z
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", utc_now_iso_ms()), \
        utc_now_iso_ms()

    with tempfile.TemporaryDirectory() as d:
        # two runs with DIFFERENT injected clocks
        m1 = Path(d) / "run1.manifest.jsonl"
        m2 = Path(d) / "run2.manifest.jsonl"
        _build(m1, clock=lambda: "2026-01-01T00:00:00.000Z")
        _build(m2, clock=lambda: "2026-06-14T12:34:56.789Z")

        e1, e2 = load_manifest(m1), load_manifest(m2)
        assert len(e1) == 3 and e1[0]["schema"] == SCHEMA
        assert e1[0]["prev_sha256"] == GENESIS_PREV
        assert verify_chain(e1) and verify_chain(e2)

        # CONTENT integrity: payload hashes identical across runs (reproducible)
        assert [r["payload_sha256"] for r in e1] == [r["payload_sha256"] for r in e2], \
            "payload_sha256 should be reproducible across runs"
        # CUSTODY: timestamps + entry chain differ because execution time differs
        assert e1[0]["audit"]["timestamp_utc"] != e2[0]["audit"]["timestamp_utc"]
        assert [r["entry_sha256"] for r in e1] != [r["entry_sha256"] for r in e2], \
            "entry_sha256 must capture execution time"

        # worker record correctly double-nested for the comparator's unwrap
        worker = e1[2]["payload"]
        assert set(worker.keys()) == set(WORKER_WRAPPER_KEYS)
        assert worker["operation"] == "layer3_segment"
        assert worker["payload"]["start_global_ms"] == 1000

        # tamper 1: flip a payload value -> chain breaks
        raw = m1.read_text().splitlines()
        bad = raw[:]
        bad[1] = bad[1].replace('"offline":true', '"offline":false')
        p = Path(d) / "tamper_payload.jsonl"
        p.write_text("\n".join(bad) + "\n")
        try:
            verify_chain(load_manifest(p))
        except ManifestError:
            pass
        else:
            raise AssertionError("payload tampering not detected")

        # tamper 2: alter a recorded timestamp -> chain breaks (custody protected)
        bad2 = raw[:]
        bad2[0] = bad2[0].replace("2026-01-01T00:00:00.000Z",
                                  "2025-01-01T00:00:00.000Z")
        p2 = Path(d) / "tamper_time.jsonl"
        p2.write_text("\n".join(bad2) + "\n")
        try:
            verify_chain(load_manifest(p2))
        except ManifestError:
            pass
        else:
            raise AssertionError("timestamp tampering not detected (custody unprotected)")

    print("manifest.py self-test: OK (chain; payload reproducible; entry binds "
          "timestamp; worker-nesting; payload+timestamp tamper both detected)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
