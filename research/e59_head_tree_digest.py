#!/usr/bin/env python3
"""Recompute the MTP head TREE digest that `mtp-head.manifest.json` declares.

The manifest's `sha256` is NOT the digest of `model.safetensors`. It is a tree
digest, defined once in `Sources/MLXFastTrustedHarness/QwenMTPHeadDeclaration.swift`
(`computeQwenMTPHeadProvenance`, lines 181-233) and mirrored in
`mtp-head/README.md`:

    SHA-256 over the concatenation, in `LC_ALL=C` sorted relative-path order, of
    `"<hex file sha256>  <relative path>\\n"` for every regular file in the tree
    except a top-level `README.md`.

Mistaking one number for the other has already cost this campaign time, so this
tool exists to put the comparable number in every leg record.

Usage:
    e59_head_tree_digest.py DIR                  # tree digest of the whole tree
    e59_head_tree_digest.py DIR --only NAME      # tree digest of one named file
    e59_head_tree_digest.py DIR --json           # digest plus the per-file table
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_READ_CHUNK = 1 << 20


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_tree_digest(root: str, only: str | None = None):
    """Return (tree_digest, total_bytes, entries) for `root`."""
    entries = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            absolute = os.path.join(dirpath, filename)
            # `enumerator.fileAttributes[.type] == .typeRegular` skips symlinks
            # and devices; `os.path.isfile` follows links, so test the link too.
            if os.path.islink(absolute) or not os.path.isfile(absolute):
                continue
            relative = os.path.relpath(absolute, root)
            if relative == "README.md":
                continue
            if only is not None and relative != only:
                continue
            entries.append(
                (relative, sha256_of_file(absolute), os.path.getsize(absolute))
            )

    if only is not None and not entries:
        raise SystemExit(f"no regular file named {only!r} under {root}")

    # Swift sorts `String` by unicode scalar, which equals byte order for the
    # ASCII relative paths a head tree uses, and equals `LC_ALL=C` sort.
    entries.sort(key=lambda entry: entry[0])

    hasher = hashlib.sha256()
    total_bytes = 0
    for relative, digest, size in entries:
        hasher.update(f"{digest}  {relative}\n".encode())
        total_bytes += size
    return hasher.hexdigest(), total_bytes, entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument(
        "--only",
        metavar="NAME",
        help="restrict the tree to one relative path, to produce the number "
        "that is comparable with a single-file manifest declaration",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        raise SystemExit(f"not a directory: {args.directory}")

    tree_digest, total_bytes, entries = head_tree_digest(args.directory, args.only)

    if args.json:
        json.dump(
            {
                "directory": args.directory,
                "only": args.only,
                "tree_sha256": tree_digest,
                "bytes": total_bytes,
                "file_count": len(entries),
                "files": [
                    {"path": path, "sha256": digest, "bytes": size}
                    for path, digest, size in entries
                ],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        print(tree_digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
