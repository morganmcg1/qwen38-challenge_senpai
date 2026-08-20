#!/usr/bin/env python3
"""Print the sha256 of the __TEXT,__text section of a Mach-O executable.

A whole-file hash of a Swift product is not reproducible: the Mach-O build UUID
and the ad-hoc code signature differ between two builds of identical source.
Hashing the executable machine code separates a real code change from that link
metadata, which is what certifies that a timed arm binary and a committed tree
are the same program.
"""
import hashlib
import subprocess
import sys


def text_section_sha256(path: str) -> tuple[str, int, int]:
    listing = subprocess.run(
        ["otool", "-l", path], capture_output=True, text=True, check=True
    ).stdout
    offset = size = None
    in_text = False
    for line in listing.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "sectname":
            in_text = fields[1] == "__text"
            offset = size = None
        elif in_text and fields[0] == "offset":
            offset = int(fields[1], 0)
        elif in_text and fields[0] == "size":
            size = int(fields[1], 0)
        if in_text and offset is not None and size is not None:
            break
    if offset is None or size is None:
        raise SystemExit(f"e60: no __TEXT,__text section found in {path}")
    with open(path, "rb") as handle:
        handle.seek(offset)
        body = handle.read(size)
    if len(body) != size:
        raise SystemExit(f"e60: truncated __text section in {path}")
    return hashlib.sha256(body).hexdigest(), offset, size


if __name__ == "__main__":
    for target in sys.argv[1:]:
        digest, offset, size = text_section_sha256(target)
        print(f"{digest} {size} {offset} {target}")
