#!/usr/bin/env python3
"""E87 arm-C liveness control: clone a cluster-index head and reverse its perm.

`clusterCandidateIDs` returns nil when a head ships no index, so a silent
fall-back to the dense readout would report a plausible time for the wrong
mechanism. Reversing `draft_cluster.perm` keeps every shape, dtype and ID legal
but destroys the row-to-token mapping, so a LIVE cluster path must collapse the
accepted-draft rate while a DEAD one must leave it unchanged.

    usage: research/e87_damage_head.py SRC_RUN_DIR DST_RUN_DIR
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    # APFS clone: 605 MB in constant time, and the copy is independent.
    subprocess.run(["cp", "-Rc", src, dst], check=True)

    path = os.path.join(dst, "model.safetensors")
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    entry = header["draft_cluster.perm"]
    assert entry["dtype"] == "I32", entry
    base = 8 + header_len
    start, end = entry["data_offsets"]
    with open(path, "r+b") as f:
        f.seek(base + start)
        perm = np.frombuffer(f.read(end - start), dtype="<i4")
        assert perm.shape == tuple(entry["shape"]), (perm.shape, entry["shape"])
        f.seek(base + start)
        f.write(perm[::-1].tobytes())

    print(f"e87-damage: {path} perm reversed, n={perm.size}")
    print(f"e87-damage: perm[:4]={perm[:4].tolist()} -> {perm[::-1][:4].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
