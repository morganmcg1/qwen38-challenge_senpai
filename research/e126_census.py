#!/usr/bin/env python3
"""Census the E121 revert across dispatched widths on both GPU generations.

E121 published 3.26815344 on the ranked runner and the advisor's per-prompt
diagnosis attributes a +2.10 % candidate-leg regression to it, shaped as a cost
per VERIFIED ROW at R2 0.9492. This module reads the two trees that bracket
that regression without spending a submission.

THE DISCRIMINATOR. E121 makes four changes to the wide entry point: the
`SHARE_SUMS = NA <= 4` gate with its half-ownership split, a `sums_xchg`
exchange carrying two threadgroup barriers per k-block, a threadgroup
allocation at the entry point, and two extra parameters threaded through
`qmv_fast_crossrow_affine4_g64_wide` and `qmv_fast_crossrow_affine4_g64_m`.

The shipped dispatch at `quantized.h:1930-1975` sends each draft width M to a
hand-set IPG, and the wide helper runs once per group of IPG rows:

    M      3    4    5    6    7    8    9
    IPG    3    4    5    3    4    4    3
    NA    [3]  [4]  [5] [3,3] [4,3] [4,4] [3,3,3]

`SHARE_SUMS` is therefore false at M = 5 and true at every other dispatched
width. M = 5 pays only the parameters and the allocation, never the split, the
exchange or the barriers, so it is a single-width control:

  - a cost absent at M = 5 and present at M = 6, 7, 8 comes from the split,
    the exchange or the barriers;
  - a cost present at every width including M = 5 comes from the parameters or
    the threadgroup allocation.

WHY THE ENTRY POINT IS CENSUSED SEPARATELY. One register allocation serves the
whole entry point with every width inlined, so a whole-kernel register effect
reaches M = 5 even though `SHARE_SUMS` is false there. A per-width body census
cannot see that, and the two cells together separate the two cases.

Residency is derived, `floor(register_file_bytes / (registers * 128))`. It is a
cost observation and never correctness evidence (Rule 73).

    python3 research/e126_census.py --out research/e126-artifacts/rung4-census.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from agx_crossarch import (  # noqa: E402
    LOCAL_ARCH, RANKED_ARCH, build_metallib, translate,
)
from e104_variant_sources import emit_base  # noqa: E402
from e121_arms import (  # noqa: E402
    ENTRY_RE, REGISTER_FILE, air_stats, simdgroups,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADER = "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
TWIN = "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"

# The pre-revert tree carries E121; the working tree is the reverted base.
E121_COMMIT = "89a426cf9517b26ed35db4d9b0422877cc44cf03"
TREES = ("e121", "reverted")

# The shipped dispatch table. Read from the source, not assumed.
M_IPG = {3: 3, 4: 4, 5: 5, 6: 3, 7: 4, 8: 4, 9: 3}
NAS = (2, 3, 4, 5)

# The entry point's own exchange allocation, copied verbatim so the wrapper
# prices the same threadgroup footprint the shipped kernel pays.
XCHG = "  threadgroup float sums_xchg[1 * 4 * 32];\n"

M_KERNEL = """
[[kernel]] void e126_m%(m)d(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
%(alloc)s  qmv_fast_crossrow_affine4_g64_m<bfloat16_t, %(m)d, %(ipg)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      tid, simd_gid, simd_lid%(extra)s);
}
"""

NA_KERNEL = """
[[kernel]] void e126_na%(na)d(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device bfloat16_t* y [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
%(alloc)s  const int first_m = int(tid.x) * %(na)d;
  if (first_m >= %(na)d) {
    return;
  }
  qmv_fast_crossrow_affine4_g64_wide<bfloat16_t, %(na)d, true>(
      w, scales, biases, x, y, in_vec_size, out_vec_size,
      first_m, int(tid.y) * 8 + int(simd_gid) * 4,%(wide_extra)s simd_lid%(tail)s);
}
"""

M_RE = re.compile(r"e126_m(\d+)$")
NA_RE = re.compile(r"e126_na(\d+)$")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), check=True,
                          capture_output=True, text=True).stdout


@contextlib.contextmanager
def checked_out(commit: str | None):
    """Expose `commit`'s candidate files, then restore the committed tree.

    The census reads the JIT source string through the working tree, so the
    pre-revert source is only reachable by materialising it. Restoring from
    HEAD is unconditional so an exception cannot leave the scored surface
    holding another commit's bytes.
    """
    if commit is None:
        yield
        return
    dirty = git("status", "--porcelain", "--", HEADER, TWIN).strip()
    if dirty:
        raise SystemExit("e126_census: candidate files are dirty, refusing")
    try:
        git("checkout", commit, "--", HEADER, TWIN)
        yield
    finally:
        git("checkout", "HEAD", "--", HEADER, TWIN)
        left = git("status", "--porcelain", "--", HEADER, TWIN).strip()
        if left:
            raise SystemExit("e126_census: restore failed, tree still dirty")


def dispatch_table(source: str) -> dict[int, int]:
    """Read M -> IPG out of the shipped dispatch rather than trusting M_IPG."""
    found = {int(m): int(ipg) for m, ipg in re.findall(
        r"qmv_fast_crossrow_affine4_g64_m<T, (\d+), (\d+), true>", source)}
    if found != M_IPG:
        raise SystemExit(
            "e126_census: dispatch table changed: %s" % sorted(found.items()))
    return found


def probe_source(base: str) -> str:
    """Append one wrapper per dispatched width and one per NA body."""
    shared = "threadgroup float* sums_xchg" in base
    alloc = XCHG if shared else ""
    text = base
    for m, ipg in sorted(dispatch_table(base).items()):
        text += M_KERNEL % {
            "m": m, "ipg": ipg, "alloc": alloc,
            "extra": ", sums_xchg" if shared else ""}
    for na in NAS:
        text += NA_KERNEL % {
            "na": na, "alloc": alloc,
            "wide_extra": " simd_gid," if shared else "",
            "tail": ", sums_xchg" if shared else ""}
    return text


def census_tree(name: str, commit: str | None, workdir: pathlib.Path) -> dict:
    with checked_out(commit):
        base = emit_base(workdir / ("base_%s.metal" % name))
        source = probe_source(base)
    path = workdir / ("probe_%s.metal" % name)
    path.write_text(source)
    air_dir = workdir / ("air_" + name)
    air_dir.mkdir(parents=True, exist_ok=True)
    row: dict = {"air": air_stats(path, air_dir),
                 "shared": "threadgroup float* sums_xchg" in base,
                 "source_bytes": len(source)}
    lib = build_metallib(source, workdir / name)
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        for kernel, record in translate(lib, arch, workdir / name).items():
            hit = M_RE.search(kernel)
            na_hit = NA_RE.search(kernel)
            if hit is not None:
                key = "M%s" % hit.group(1)
            elif na_hit is not None:
                key = "NA%s" % na_hit.group(1)
            elif ENTRY_RE.search(kernel):
                key = "entry"
            else:
                continue
            registers = record.get("registers")
            row.setdefault(arch, {})[key] = {
                "registers": registers,
                "spill_bytes": record.get("spill_bytes", 0),
                "text_bytes": record.get("text_bytes"),
                "text_sha8": record.get("text_sha8"),
                "resident_simdgroups":
                    simdgroups(arch, registers) if registers else None,
            }
    return row


def delta_table(rows: dict, arch: str, keys: list[str]) -> list[dict]:
    out = []
    for key in keys:
        pre = rows["e121"].get(arch, {}).get(key)
        post = rows["reverted"].get(arch, {}).get(key)
        if pre is None or post is None:
            continue
        out.append({
            "cell": key,
            "registers_reverted": post["registers"],
            "registers_e121": pre["registers"],
            "registers_delta": pre["registers"] - post["registers"],
            "spill_reverted": post["spill_bytes"] or 0,
            "spill_e121": pre["spill_bytes"] or 0,
            "text_reverted": post["text_bytes"],
            "text_e121": pre["text_bytes"],
            "sg_reverted": post["resident_simdgroups"],
            "sg_e121": pre["resident_simdgroups"],
            "sg_delta": (pre["resident_simdgroups"]
                         - post["resident_simdgroups"]),
        })
    return out


def report(rows: dict, out: pathlib.Path | None) -> int:
    m_keys = ["M%d" % m for m in sorted(M_IPG)]
    na_keys = ["NA%d" % na for na in NAS]
    tables: dict = {}
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        for label, keys in (("dispatched_width", m_keys),
                            ("na_body", na_keys), ("entry", ["entry"])):
            tables.setdefault(arch, {})[label] = delta_table(rows, arch, keys)

    for arch in (LOCAL_ARCH, RANKED_ARCH):
        print("\n=== %s ===" % arch)
        for label in ("dispatched_width", "na_body", "entry"):
            print("\n%s: registers rev->e121 (delta) | spill | "
                  "resident simdgroups rev->e121 (delta)" % label)
            for row in tables[arch][label]:
                print("  %-6s R %3d -> %3d (%+d)  sp %d/%d  text %6d/%6d  "
                      "sg %2d -> %2d (%+d)" % (
                          row["cell"], row["registers_reverted"],
                          row["registers_e121"], row["registers_delta"],
                          row["spill_reverted"], row["spill_e121"],
                          row["text_reverted"], row["text_e121"],
                          row["sg_reverted"], row["sg_e121"], row["sg_delta"]))

    verdict = discriminate(tables)
    print("\n=== discriminator ===")
    for line in verdict["lines"]:
        print("  " + line)

    payload = {
        "harness": "local",
        "timing_valid": False,
        "cool_gate_passed_real_gate": False,
        "gate_qualified_for_timing": False,
        "official_or_ranked_score": False,
        "e121_commit": E121_COMMIT,
        "reverted_commit": git("rev-parse", "HEAD").strip(),
        "dispatch_table": M_IPG,
        "register_file_bytes": dict(REGISTER_FILE),
        "shared": {t: rows[t]["shared"] for t in TREES},
        "tables": tables,
        "discriminator": verdict,
        "raw": rows,
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print("\nwrote %s" % out)
    return 0


def discriminate(tables: dict) -> dict:
    """Read the pre-registered branch from the M = 5 control."""
    lines, verdict = [], {}
    for arch in (LOCAL_ARCH, RANKED_ARCH):
        rows = {r["cell"]: r for r in tables[arch]["dispatched_width"]}
        control = rows.get("M5")
        shared_cells = [r for c, r in rows.items() if c != "M5"]
        if control is None or not shared_cells:
            continue
        control_hit = control["sg_delta"] != 0 or control["registers_delta"] != 0
        shared_hit = any(r["sg_delta"] != 0 or r["registers_delta"] != 0
                         for r in shared_cells)
        if control_hit and shared_hit:
            reading = "parameters_or_allocation"
        elif shared_hit:
            reading = "split_exchange_or_barriers"
        elif control_hit:
            reading = "control_only_unexpected"
        else:
            reading = "no_static_signature"
        verdict[arch] = {
            "reading": reading,
            "m5_registers_delta": control["registers_delta"],
            "m5_sg_delta": control["sg_delta"],
            "shared_width_sg_deltas": {
                r["cell"]: r["sg_delta"] for r in shared_cells},
        }
        lines.append("%s: %s (M5 dR %+d, dsg %+d)" % (
            arch, reading, control["registers_delta"], control["sg_delta"]))
    return {"per_arch": verdict, "lines": lines}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--keep", type=pathlib.Path)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        workdir = args.keep or pathlib.Path(tmp)
        workdir.mkdir(parents=True, exist_ok=True)
        rows = {}
        for name, commit in (("e121", E121_COMMIT), ("reverted", None)):
            rows[name] = census_tree(name, commit, workdir)
            print("censused %s" % name)
        return report(rows, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
