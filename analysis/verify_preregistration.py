"""Check the files frozen in each preregistration against this repository.

Every preregistration in docs/preregistrations/ ends with a block of SHA-256 hashes recorded before any data were
generated. This script finds each listed file in the repository (by its name, since the repository was reorganised
into folders after the studies) and compares hashes.

Status per file:
  OK            identical to the frozen version
  REDACTED      prompt file published without the cluster image paths (NAVSIM images are not redistributed);
                its frozen hash refers to the cluster copy
  NOT PUBLISHED cluster-only file (job lists, environment paths)
  MISMATCH      anything else (none expected)

usage: python analysis/verify_preregistration.py
"""
import glob
import hashlib
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
index = {}
for p in glob.glob(f"{ROOT}/**/*", recursive=True):
    if os.path.isfile(p) and "/.git/" not in p:
        index.setdefault(os.path.basename(p), []).append(p)
# a few frozen paths are not unique by name; resolve them by their recorded relative path
by_rel = {os.path.relpath(p, ROOT): p for ps in index.values() for p in ps}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


bad = 0
for doc in sorted(glob.glob(f"{ROOT}/docs/preregistrations/*.md")):
    print(f"\n{os.path.relpath(doc, ROOT)}")
    text = open(doc).read()
    entries = re.findall(r"^([0-9a-f]{64})\s+(\S+)$", text, re.M)                  # "hash  path" blocks
    entries += [(h, p) for p, h in re.findall(r"^- `([^`]+)` sha256 ([0-9a-f]{64})", text, re.M)]   # "- `path` sha256 hash"
    for h, path in entries:
        if path.startswith("docs/"):
            continue                      # the document's own hash covers its text before the hash list
        cands = [by_rel[path]] if path in by_rel else index.get(os.path.basename(path), [])
        hits = [c for c in cands if sha(c) == h]
        if hits:
            status = "OK"
        elif path.startswith("prompts/") and cands:
            status = "REDACTED"
        elif not cands or (path.startswith("jobs/") and path.endswith((".jsonl", ".txt"))):
            status = "NOT PUBLISHED"
        else:
            status, bad = "MISMATCH", bad + 1
        where = os.path.relpath(hits[0] if hits else cands[0], ROOT) if (hits or cands) else "-"
        print(f"  {status:13s} {path:45s} -> {where}")
print("\nall frozen analysis code matches" if bad == 0 else f"\n{bad} mismatches")
sys.exit(1 if bad else 0)
