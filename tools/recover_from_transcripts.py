"""Reconstruct deleted .py sources from Cursor agent transcripts.

Strategy: replay Write / StrReplace / Delete tool_use events in order, then
validate each reconstructed file against the on-disk .pyc by comparing compiled
code objects (robust to filename/mtime). Files that match the pyc exactly are
"verified"; others are "best effort".

Usage:
  python tools/recover_from_transcripts.py            # dry run, report only
  python tools/recover_from_transcripts.py --write     # write verified files
  python tools/recover_from_transcripts.py --write-all # write verified + best-effort
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import marshal
import os
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Muhammad Nabil Wafi\Documents\projects\xauusd")
TRANSCRIPTS = (
    r"C:\Users\Muhammad Nabil Wafi\.cursor\projects"
    r"\c-Users-Muhammad-Nabil-Wafi-Documents-projects-xauusd\agent-transcripts"
)

# Only recover source we care about (keeps blast radius small).
INCLUDE_PREFIXES = ("production/", "apps/", "settings/", "research/", "sql/", "tests/", "pipeline/")


def _norm(p: str) -> str:
    p = str(p).replace("\\", "/")
    low = p.lower()
    idx = low.find("/xauusd/")
    if idx >= 0:
        return p[idx + len("/xauusd/") :]
    return p


def _iter_tool_events(paths: list[str]):
    """Yield (transcript_index, line_no, tool_name, input) in file order."""
    for ti, fp in enumerate(paths):
        with open(fp, encoding="utf-8") as f:
            for ln, line in enumerate(f):
                if "tool_use" not in line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                msg = obj.get("message", {})
                cont = msg.get("content") if isinstance(msg, dict) else None
                if not isinstance(cont, list):
                    continue
                for c in cont:
                    if not isinstance(c, dict) or c.get("type") != "tool_use":
                        continue
                    yield ti, ln, c.get("name", ""), c.get("input")


def _apply_v4a(files: dict[str, str | None], patch: str) -> None:
    """Apply an apply_patch V4A string to the reconstructed file map."""
    lines = patch.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("*** Add File: "):
            rel = _norm(line[len("*** Add File: ") :].strip())
            i += 1
            body = []
            while i < n and not lines[i].startswith("*** "):
                body.append(lines[i][1:] if lines[i].startswith("+") else lines[i])
                i += 1
            if any(rel.startswith(p) for p in INCLUDE_PREFIXES):
                files[rel] = "\n".join(body) + "\n"
            continue
        if line.startswith("*** Delete File: "):
            rel = _norm(line[len("*** Delete File: ") :].strip())
            if any(rel.startswith(p) for p in INCLUDE_PREFIXES):
                files[rel] = None
            i += 1
            continue
        if line.startswith("*** Update File: "):
            rel = _norm(line[len("*** Update File: ") :].strip())
            i += 1
            hunk: list[str] = []
            relevant = any(rel.startswith(p) for p in INCLUDE_PREFIXES)
            while i < n and not lines[i].startswith("*** "):
                hunk.append(lines[i])
                i += 1
            if relevant and files.get(rel) is not None:
                _apply_hunks(files, rel, hunk)
            continue
        i += 1


def _apply_hunks(files: dict[str, str | None], rel: str, hunk_lines: list[str]) -> None:
    content = files[rel] or ""
    # Split into hunks on '@@' markers.
    groups: list[list[str]] = []
    cur: list[str] = []
    for l in hunk_lines:
        if l.startswith("@@"):
            if cur:
                groups.append(cur)
            cur = []
        else:
            cur.append(l)
    if cur:
        groups.append(cur)
    for g in groups:
        before, after = [], []
        for l in g:
            tag = l[:1]
            rest = l[1:] if l else ""
            if tag == "-":
                before.append(rest)
            elif tag == "+":
                after.append(rest)
            else:
                before.append(rest)
                after.append(rest)
        b = "\n".join(before)
        a = "\n".join(after)
        if b and b in content:
            content = content.replace(b, a, 1)
    files[rel] = content


def reconstruct(transcript_order: list[str]) -> dict[str, str | None]:
    """Return {relpath: content|None}. None means Delete-d."""
    files: dict[str, str | None] = {}
    for _ti, _ln, name, inp in _iter_tool_events(transcript_order):
        if name == "ApplyPatch":
            if isinstance(inp, str):
                _apply_v4a(files, inp)
            elif isinstance(inp, dict):
                patch = inp.get("patch") or inp.get("input") or ""
                if patch:
                    _apply_v4a(files, patch)
            continue
        if not isinstance(inp, dict):
            continue
        p = inp.get("path") or inp.get("file_path") or inp.get("target_file") or ""
        if not p:
            continue
        rel = _norm(p)
        if not any(rel.startswith(pre) for pre in INCLUDE_PREFIXES):
            continue
        if name == "Write":
            c = inp.get("contents", inp.get("content"))
            if c is not None:
                files[rel] = c
        elif name == "StrReplace":
            old = inp.get("old_string")
            new = inp.get("new_string", "")
            cur = files.get(rel)
            if cur is None or old is None:
                continue
            if inp.get("replace_all"):
                files[rel] = cur.replace(old, new)
            elif old in cur:
                files[rel] = cur.replace(old, new, 1)
        elif name == "Delete":
            files[rel] = None
    return files


def _code_sig(code) -> tuple:
    """Structural signature of a code object, ignoring filename/firstlineno."""
    consts = tuple(
        _code_sig(c) if hasattr(c, "co_code") else repr(c) for c in code.co_consts
    )
    return (
        code.co_name,
        code.co_argcount,
        getattr(code, "co_posonlyargcount", 0),
        code.co_kwonlyargcount,
        code.co_flags,
        code.co_code,
        consts,
        code.co_names,
        code.co_varnames,
    )


def _load_pyc_code(pyc: Path):
    data = pyc.read_bytes()
    return marshal.loads(data[16:])


def _pyc_for(rel: str) -> Path | None:
    src = ROOT / rel
    stem = src.stem
    cache = src.parent / "__pycache__"
    if not cache.exists():
        return None
    hits = list(cache.glob(f"{stem}.*.pyc"))
    return hits[0] if hits else None


def verify(rel: str, content: str) -> str:
    """Return 'verified' | 'mismatch' | 'no-pyc' | 'compile-error'."""
    pyc = _pyc_for(rel)
    if pyc is None:
        return "no-pyc"
    try:
        want = _load_pyc_code(pyc)
    except Exception:
        return "no-pyc"
    try:
        got = compile(content, str(ROOT / rel), "exec")
    except SyntaxError:
        return "compile-error"
    return "verified" if _code_sig(got) == _code_sig(want) else "mismatch"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--write-all", action="store_true")
    args = ap.parse_args()

    # Chronological session order (from first user message timestamps).
    session_order = [
        "ded74095-3b7a-4c95-9c46-3b60d28e3d19",  # Jul 5
        "22e7c9f7-e888-4c88-9bbf-dad0f97f46df",  # Jul 8
        "136e70ce-bc9a-4552-bdd2-e2ecb4ab91d3",  # Jul 16-25
    ]
    order = []
    for sess in session_order:
        subs = sorted(glob.glob(os.path.join(TRANSCRIPTS, sess, "subagents", "*.jsonl")))
        main = os.path.join(TRANSCRIPTS, sess, f"{sess}.jsonl")
        order.extend(subs)  # subagent edits happen during the session
        if os.path.exists(main):
            order.append(main)

    files = reconstruct(order)
    rows = []
    for rel, content in sorted(files.items()):
        if content is None:
            rows.append((rel, "deleted", 0))
            continue
        status = verify(rel, content)
        rows.append((rel, status, len(content)))

    verified = [r for r in rows if r[1] == "verified"]
    others = [r for r in rows if r[1] not in ("verified", "deleted")]
    print(f"total={len(rows)} verified={len(verified)} other={len(others)}")
    print("\n=== VERIFIED (exact match to pyc) ===")
    for rel, st, n in verified:
        print(f"  {n:6d}  {rel}")
    print("\n=== NOT VERIFIED ===")
    for rel, st, n in others:
        print(f"  {st:14s} {n:6d}  {rel}")

    if args.write or args.write_all:
        wrote = 0
        for rel, content in files.items():
            if content is None:
                continue
            st = verify(rel, content)
            if st == "verified" or (args.write_all and st != "compile-error"):
                dest = ROOT / rel
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
                wrote += 1
                print(f"WROTE [{st}] {rel}")
        print(f"\nwrote {wrote} files")


if __name__ == "__main__":
    main()
