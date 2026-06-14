"""
SOTA SANDBOX — sanitize_manifest.py  (READ-ONLY redactor)
=========================================================

Purpose:    Produce a REDACTED COPY of a Forensic-Pipeline session manifest
            with the pipeline's project name stripped out, so it can be
            shared in study artifacts where that name must not appear.

            The manifest carries the pipeline name inside its ``schema``
            string (``<name>-manifest-v1``) and possibly elsewhere (paths,
            etc.). This tool auto-detects that name token FROM THE MANIFEST
            ITSELF (so the confidential word is never hardcoded in this
            source file), then replaces every case-insensitive occurrence
            with a neutral replacement (default: ``forensic``). The schema
            therefore becomes ``forensic-manifest-v1``, which
            ``compare_results.py`` accepts cleanly.

Safety:     READ-ONLY on the source — it never modifies the original. It
            writes ONE new file (``-o``/auto-named) and refuses to write
            over the input. It does not print the detected name token, so
            the confidential word never reaches the terminal/logs.

Run:        python3 sanitize_manifest.py <session.manifest.jsonl>
                [-o session.clean.manifest.jsonl] [--replacement forensic]
                [--redact TOKEN]   # force a token instead of auto-detecting
Self-test:  python3 sanitize_manifest.py --selftest    (stdlib only)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_REPLACEMENT = "forensic"
_SCHEMA_RE = re.compile(r"^(?P<name>.+)-manifest-v\d+$")


def detect_name_token(first_line):
    """Given the manifest's first JSON-L record, return the project-name
    token embedded in its ``schema`` (the part before ``-manifest-vN``), or
    None if the schema does not follow that pattern."""
    try:
        rec = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    schema = rec.get("schema")
    if not isinstance(schema, str):
        return None
    m = _SCHEMA_RE.match(schema)
    if not m:
        return None
    name = m.group("name").strip()
    return name or None


def redact_text(text, token, replacement=DEFAULT_REPLACEMENT):
    """Replace every case-insensitive occurrence of `token` with
    `replacement`. Returns (new_text, count). Operates on raw text so the
    token is scrubbed wherever it appears (schema, embedded paths, …)."""
    if not token:
        return text, 0
    pattern = re.compile(re.escape(token), re.IGNORECASE)
    new_text, count = pattern.subn(replacement, text)
    return new_text, count


def sanitize(in_path, out_path, *, replacement=DEFAULT_REPLACEMENT,
             forced_token=None, log=print):
    in_path, out_path = Path(in_path), Path(out_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"manifest not found: {in_path}")
    if out_path.resolve() == in_path.resolve():
        raise ValueError("refusing to overwrite the source manifest — "
                         "choose a different --out path")
    text = in_path.read_text(encoding="utf-8")

    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    token = forced_token or detect_name_token(first_line)
    if not token:
        raise ValueError(
            "could not auto-detect the pipeline name token from the manifest "
            "schema (expected a '<name>-manifest-vN' schema). Pass --redact "
            "TOKEN explicitly.")
    # A short auto-detected token (e.g. a 1-2 char schema prefix) would match
    # as a substring all over the file and corrupt it. Require confirmation.
    if not forced_token and len(token) < 4:
        raise ValueError(
            f"auto-detected name token is only {len(token)} char(s); a global "
            "substring replace could corrupt the manifest. Re-run with "
            "--redact TOKEN to confirm the exact token to remove.")

    new_text, count = redact_text(text, token, replacement)
    out_path.write_text(new_text, encoding="utf-8")
    # Deliberately do NOT print the detected token (it is the confidential word).
    log(f"  redacted {count} occurrence(s) of the pipeline name token "
        f"-> '{replacement}'")
    log(f"  wrote sanitized copy: {out_path}  (source untouched: {in_path})")
    return count


def _default_out(in_path):
    p = Path(in_path)
    stem = p.name
    for suffix in (".manifest.jsonl", ".jsonl"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return Path.cwd() / f"{stem}.clean.manifest.jsonl"


# =============================================================================
# Self-test (stdlib only)
# =============================================================================

def _selftest():
    import tempfile
    secret = "Ztoken"           # stand-in for the confidential word (not real)
    manifest = "\n".join([
        json.dumps({"schema": f"{secret}-manifest-v1", "operation": "start"}),
        json.dumps({"operation": "layer3_segment",
                    "payload": {"wav_path": f"/data/{secret}/clip.wav"}}),
    ]) + "\n"

    # auto-detection from the schema
    first = manifest.splitlines()[0]
    assert detect_name_token(first) == secret, detect_name_token(first)
    assert detect_name_token('{"schema": "no-version-here"}') is None
    assert detect_name_token("not json") is None

    # redaction: case-insensitive, counts both the schema and the path
    new_text, n = redact_text(manifest, secret, "forensic")
    assert n == 2, n
    assert secret.lower() not in new_text.lower(), "token survived"
    # output still parses and the schema is now neutral
    first_obj = json.loads(new_text.splitlines()[0])
    assert first_obj["schema"] == "forensic-manifest-v1", first_obj["schema"]

    # end-to-end via files; source must be left untouched
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "session.manifest.jsonl"
        src.write_text(manifest, encoding="utf-8")
        dst = Path(d) / "session.clean.manifest.jsonl"
        count = sanitize(src, dst, replacement="forensic", log=lambda *_: None)
        assert count == 2
        assert src.read_text() == manifest, "source was modified!"
        assert secret.lower() not in dst.read_text().lower()

        # refusing to overwrite the source
        try:
            sanitize(src, src, log=lambda *_: None)
        except ValueError:
            pass
        else:
            raise AssertionError("expected refusal to overwrite source")

        # short-token guard fires on auto-detect, but --redact bypasses it
        short = json.dumps({"schema": "ab-manifest-v1", "operation": "x"}) + "\n"
        sp = Path(d) / "short.manifest.jsonl"
        sp.write_text(short, encoding="utf-8")
        try:
            sanitize(sp, Path(d) / "short.out.jsonl", log=lambda *_: None)
        except ValueError:
            pass
        else:
            raise AssertionError("expected short-token guard to fire")
        sanitize(sp, Path(d) / "short.forced.jsonl", forced_token="ab",
                 log=lambda *_: None)

    print("sanitize_manifest.py self-test: OK (detect + redact + file IO, "
          "guards, source untouched, stdlib only)")
    return 0


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Write a redacted copy of an FP manifest (strips the "
                    "pipeline name). Read-only on the source.")
    p.add_argument("manifest", nargs="?", help="path to the source manifest")
    p.add_argument("-o", "--out", default=None,
                   help="output path (default: <name>.clean.manifest.jsonl in CWD)")
    p.add_argument("--replacement", default=DEFAULT_REPLACEMENT,
                   help="neutral replacement token (default: %(default)s)")
    p.add_argument("--redact", default=None,
                   help="force a specific token to redact instead of "
                        "auto-detecting it from the schema")
    p.add_argument("--selftest", action="store_true",
                   help="run stdlib-only self-test and exit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.manifest:
        print("ERROR: provide a manifest path (or --selftest)", file=sys.stderr)
        return 2
    out = args.out or _default_out(args.manifest)
    try:
        sanitize(args.manifest, out, replacement=args.replacement,
                 forced_token=args.redact)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
