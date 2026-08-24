#!/usr/bin/env python3
"""E178 step 1b. Editable-archive digest per board submission.

Exact tree identity (`e178_tree_groups.py`) is the strictest criterion but it
also separates snapshots that differ only in inert files such as notes. Yukon
packages only `benchmark.json:editablePaths`, so two snapshots with the same
content under those paths ship the same archive and build the same candidate
against the trusted harness of their day.

This script walks the blobless submission trees in /tmp/subtrees and emits, per
submission id, a sha256 over the sorted `mode path blobsha` lines of every
editable path. Writes /tmp/e178/editable_digest.json.
"""
import hashlib
import json
import os
import subprocess

REPO = "/tmp/subtrees2"
OUT = "/tmp/e178/editable_digest.json"


def editable_paths():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = json.load(open(os.path.join(root, "benchmark.json")))
    return sorted(spec["editablePaths"])


def main():
    paths = editable_paths()
    refs = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)", "refs/bn/*"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1")
    out = {}
    for ref in refs:
        sid = ref.rsplit("/", 1)[-1]
        proc = subprocess.run(
            ["git", "ls-tree", "-r", "--full-tree", ref, "--"] + paths,
            cwd=REPO, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            out[sid] = None
            continue
        lines = sorted(proc.stdout.splitlines())
        out[sid] = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"))
    ok = sum(1 for v in out.values() if v)
    print("digested %d/%d submission trees -> %s" % (ok, len(out), OUT))


if __name__ == "__main__":
    main()
