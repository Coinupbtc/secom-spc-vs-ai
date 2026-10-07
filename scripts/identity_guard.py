"""Generic identifier guard. This file contains no guarded values.

Guarded identifiers are configured OUTSIDE the repository, in one of two ways:

1. CI (keyed): env IDENTITY_GUARD_KEY (secret HMAC key) and IDENTITY_GUARD_DIGESTS (comma-separated
   hex HMAC-SHA256 digests of the normalized identifiers), both supplied from GitHub Actions secrets.
   Without them (forks, or secrets not configured) the check is skipped with a notice.
2. Local: --patterns-file PATH, a private file (chmod 600, outside the repo) with one identifier per line.

Normalization: lowercase, keep alphanumeric runs, and concatenate n-grams of 1..4 consecutive tokens, so spacing,
punctuation and case variants are caught. Scope: tracked files, plus full history patches, commit metadata and annotated tag
messages for the given revisions (default: --all). On a hit, only the location is printed, never the matched text.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import subprocess
import sys

MAX_N = 4
TOK = re.compile(r"[a-z0-9]+")


def norm(s: str) -> str:
    return "".join(TOK.findall(s.lower()))


def make_matcher(args):
    if args.patterns_file:
        with open(os.path.expanduser(args.patterns_file), encoding="utf-8") as fh:
            pats = {norm(l) for l in fh if l.strip() and not l.lstrip().startswith("#")}
        pats.discard("")
        return (lambda g: g in pats), f"{len(pats)} local patterns"
    key, dig = os.environ.get("IDENTITY_GUARD_KEY", ""), os.environ.get("IDENTITY_GUARD_DIGESTS", "")
    if key and dig:
        digs = {d.strip().lower() for d in dig.split(",") if d.strip()}
        k = key.encode()
        return (lambda g: hmac.new(k, g.encode(), hashlib.sha256).hexdigest() in digs), f"{len(digs)} keyed digests"
    return None, None


def has_hit(text: str, match) -> bool:
    toks = TOK.findall(text.lower())
    for i in range(len(toks)):
        s = ""
        for n in range(min(MAX_N, len(toks) - i)):
            s += toks[i + n]
            if match(s):
                return True
    return False


def git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, errors="ignore").stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns-file")
    ap.add_argument("--revs", nargs="*", default=["--all"], help="revisions whose history to scan (default --all)")
    args = ap.parse_args()
    match, desc = make_matcher(args)
    if match is None:
        print("::notice::identifier guard skipped: IDENTITY_GUARD_KEY / IDENTITY_GUARD_DIGESTS not configured")
        return 0
    bad = []
    files = [f for f in git("ls-files", "-z").split("\0") if f]
    for f in files:
        try:
            with open(f, "rb") as fh:
                txt = fh.read().decode("utf-8", errors="ignore")
        except OSError:
            continue
        if has_hit(txt, match) or has_hit(f, match):
            bad.append(f"tracked file: {f}")
    # history: one chunk per commit (patch + author/committer/message)
    log = git("log", "-p", "--format=%x00%H%n%an%n%ae%n%cn%n%ce%n%B", *args.revs)
    for chunk in log.split("\0"):
        if chunk.strip() and has_hit(chunk, match):
            bad.append(f"history: commit {chunk.split(chr(10), 1)[0][:12]}")
    for line in git("for-each-ref", "refs/tags", "--format=%(refname:short)%00%(objecttype)").splitlines():
        name, typ = line.split("\0")
        content = git("cat-file", "-p", name) if typ == "tag" else ""
        if has_hit(name + "\n" + content, match):
            bad.append(f"tag: {name}")
    if bad:
        print("IDENTIFIER GUARD FAILED; guarded identifier found in:")
        print("\n".join("  " + b for b in sorted(set(bad))))
        return 1
    print(f"identifier guard OK ({len(files)} tracked files, history, tags; {desc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
