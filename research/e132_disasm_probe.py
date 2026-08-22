#!/usr/bin/env python3
"""Probe whether this toolchain can disassemble a translated AGX object.

Rung 1 step 1 needs the spill SITE, not only the frame size. `metal-objdump`
advertises `agx1`/`agx2`/`agx3` targets, so this asks it, every documented way,
for the machine code of a kernel that is known to spill. If none of the
invocations returns text, the campaign has no public AGX disassembler and the
spill site must be located by source ablation instead.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agx_crossarch as agx  # noqa: E402

SPILLER = """#include <metal_stdlib>
using namespace metal;
kernel void k_spill(device float* o, device const float* a,
                    uint i [[thread_position_in_grid]]) {
  float v[200];
  for (int j = 0; j < 200; j++) v[j] = a[i * 200 + j];
  for (int t = 0; t < 3; t++)
    for (int j = 0; j < 200; j++)
      v[j] = fma(v[j], v[(j + 5) % 200], v[(j + 7) % 200]);
  float s = 0;
  for (int j = 0; j < 200; j++) s += v[j];
  o[i] = s;
}
"""

ATTEMPTS = (
    ["-d"],
    ["-d", "--macho"],
    ["-D"],
    ["-d", "--triple=agx3"],
    ["-d", "--triple=agx2"],
    ["-d", "--arch-name=agx3"],
    ["-d", "--mcpu=applegpu_g17s"],
    ["-d", "--disassemble-zeroes"],
    ["--build-table=all"],
    ["--all-headers"],
)


ARTIFACTS = pathlib.Path(__file__).resolve().parent / "e132-artifacts"


def main() -> int:
    attempts = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        lib = agx.build_metallib(SPILLER, workdir)
        script = workdir / "p.mtlp-json"
        script.write_text(json.dumps({"pipelines": {"compute_pipelines": [
            {"compute_function": "k_spill"}]}}))
        out = workdir / "g17s.mtlp"
        subprocess.run(["xcrun", "metal-tt", "-arch", "applegpu_g17s",
                        str(lib), str(script), "-o", str(out)],
                       check=True, capture_output=True)
        compute = agx.section(out.read_bytes(), "__TEXT,__compute")
        inner = compute[agx.find_mach_headers(compute)[0]:]
        obj = workdir / "inner.o"
        obj.write_bytes(inner)
        record = agx.kernel_records(out.read_bytes())[0]
        print("probe kernel: registers=%s spill_bytes=%s text_bytes=%s"
              % (record["registers"], record["spill_bytes"],
                 record["text_bytes"]))
        for target in (obj, out):
            for args in ATTEMPTS:
                done = subprocess.run(["xcrun", "metal-objdump", *args,
                                       str(target)],
                                      capture_output=True, text=True)
                text = (done.stdout or "") + (done.stderr or "")
                body = [l for l in text.splitlines()
                        if l.strip() and not l.startswith("/")]
                print("%-40s %-10s rc=%d lines=%d"
                      % (" ".join(args), target.name, done.returncode,
                         len(body)))
                attempts.append({"args": args, "target": target.name,
                                 "returncode": done.returncode,
                                 "body_lines": len(body)})
                if len(body) > 8:
                    print("\n".join(body[:20]))
        probe = {"registers": record["registers"],
                 "spill_bytes": record["spill_bytes"],
                 "text_bytes": record["text_bytes"]}

    disassembled = [a for a in attempts if a["returncode"] == 0
                    and a["body_lines"] > 8]
    receipt = {
        "schema_version": 1,
        "gpu_used": False,
        "harness": "compile_only",
        "tool": "research/e132_disasm_probe.py",
        "probe_kernel": probe,
        "attempts": attempts,
        "any_attempt_disassembled": bool(disassembled),
        "conclusion":
            "This toolchain has no working AGX disassembler. `metal-objdump` "
            "reads the container headers and returns rc=1 with no text for "
            "every documented disassembly invocation, on both the extracted "
            "inner Mach-O and the whole translated object. A spill SITE must "
            "therefore be located by source ablation, by watching the frame "
            "track the live set, and by counting the live set, not by reading "
            "machine code. Do not repeat this probe without a new toolchain.",
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "disasm-probe.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("\n%s\nwrote %s" % (receipt["conclusion"], path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
