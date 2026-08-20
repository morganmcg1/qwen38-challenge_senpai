#!/usr/bin/env python3
"""Per-section digests for the Mach-O runtime worker.

The advisor asked for the `__TEXT,__text` digest instead of the whole-file
hash, and that request exposes a real property of these arms. The arm
difference in E59 is one template argument inside a JIT kernel source string,
so it lands in `__TEXT,__cstring`, not in machine code. A section table
therefore says which of the two things moved:

  * `__text` equal and `__cstring` different -> same host code, new kernel
    source, which is what a routing-only arm must look like;
  * `__text` different -> a host-side change came along for the ride, and the
    contrast is no longer a clean routing contrast.

The whole-file digest cannot answer either question, because it also moves
with `LC_UUID` and the code-signature slots on every relink.

  python3 research/e59_worker_digest.py .build-worker/release/mlxfast-runtime-worker
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

SECTIONS = ("__TEXT,__text", "__TEXT,__cstring", "__TEXT,__const",
            "__DATA_CONST,__const")

_FIELD = re.compile(r"^\s*(sectname|segname|size|offset)\s+(\S+)\s*$")


def section_table(path: pathlib.Path) -> dict[str, dict]:
    out = subprocess.run(["otool", "-l", str(path)], check=True,
                         capture_output=True, text=True).stdout
    table: dict[str, dict] = {}
    current: dict[str, str] = {}
    for line in out.splitlines():
        match = _FIELD.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if key == "sectname":
            current = {"sectname": value}
            continue
        if not current:
            continue
        current[key] = value
        if {"sectname", "segname", "size", "offset"} <= set(current):
            name = f"{current['segname']},{current['sectname']}"
            table.setdefault(name, {
                "offset": int(current["offset"]),
                "size": int(current["size"], 16),
            })
            current = {}
    return table


def digests(path: pathlib.Path, sections=SECTIONS) -> dict:
    table = section_table(path)
    blob = path.read_bytes()
    out = {
        "path": str(path),
        "file_bytes": len(blob),
        "file_sha256": hashlib.sha256(blob).hexdigest(),
        "sections": {},
    }
    for name in sections:
        entry = table.get(name)
        if entry is None:
            out["sections"][name] = None
            continue
        chunk = blob[entry["offset"]:entry["offset"] + entry["size"]]
        out["sections"][name] = {
            "bytes": len(chunk),
            "sha256": hashlib.sha256(chunk).hexdigest(),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", type=pathlib.Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    record = digests(args.binary)
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    print("%-24s %12d  %s" % ("file", record["file_bytes"],
                              record["file_sha256"]))
    for name, entry in record["sections"].items():
        if entry is None:
            print("%-24s %12s  %s" % (name, "-", "absent"))
            continue
        print("%-24s %12d  %s" % (name, entry["bytes"], entry["sha256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
