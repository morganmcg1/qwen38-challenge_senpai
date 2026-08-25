#!/usr/bin/env bash
# E216 Stage-0 desk census (RULE 400): register pressure, occupancy and
# threadgroup memory of the shipped `split` weight-stream mapping and of the
# E216 `coop` mapping.
#
#   research/e216_occupancy.sh [TAG]
#
# `maxTotalThreadsPerThreadgroup` is the register budget the Metal back end
# assigned to a kernel and `staticThreadgroupMemoryLength` is its threadgroup
# allocation, so the pair reads both occupancy limits the paired mapping could
# move. Rung A stages nothing, so the expectation is that both mappings compile
# the identical instantiation set at the identical budget.
#
# HARD CONSTRAINT (FINDING 552): no plan may compile <NA=7, ROWS=2>. This probe
# holds rows_per_simd at the shipped 4 and never emits that form; the check
# below fails the census if it ever appears.
#
# No dispatch, no timing, no model: this compiles kernels and builds pipeline
# states only.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:-e216/occupancy}"
out="research/out/${tag}"
rm -rf "${out}"
mkdir -p "${out}"

metal_src="${out}/probe.metal"
python3 research/e216_probe_gen.py "${metal_src}" || exit 1

if grep -q "rows_per_simd = [^4]" "${metal_src}"; then
  echo "e216: probe moved rows_per_simd off the shipped 4" >&2
  exit 3
fi
if grep -qE "qwen_e120_qmv_wide<7," "${metal_src}"; then
  echo "e216: probe reached NA = 7, which FINDING 552 forbids" >&2
  exit 3
fi

xcrun -sdk macosx metal -std=metal3.1 -O2 -c "${metal_src}" -o "${out}/probe.air" \
  2> "${out}/compile.log" || { cat "${out}/compile.log"; exit 1; }
xcrun -sdk macosx metallib "${out}/probe.air" -o "${out}/probe.metallib" || exit 1

swiftc -O research/crossrow_na_occupancy.swift -o "${out}/na_occupancy" || exit 1
"${out}/na_occupancy" "${out}/probe.metallib" | tee "${out}/occupancy.txt"

python3 research/e213_regs.py "${metal_src}" "${out}/regs.json" | tee "${out}/regs.txt"

# The census decision: the paired mapping may not lose occupancy or take
# threadgroup memory anywhere.
python3 - "${out}/occupancy.txt" "${out}/census.json" <<'PY'
import json, sys

rows = {}
for line in open(sys.argv[1]).read().splitlines():
    parts = line.split()
    if len(parts) != 4 or parts[0] in ("name",):
        continue
    try:
        rows[parts[0]] = {
            "max_threads": int(parts[1]),
            "exec_width": int(parts[2]),
            "tg_mem_bytes": int(parts[3]),
        }
    except ValueError:
        continue

problems = []
pairs = []
for name, row in sorted(rows.items()):
    if "_split_" not in name and "_split" not in name:
        continue
    twin = name.replace("_split", "_coop")
    if twin not in rows:
        problems.append(f"{name}: no coop twin")
        continue
    other = rows[twin]
    pairs.append({"split": name, "coop": twin,
                  "split_max_threads": row["max_threads"],
                  "coop_max_threads": other["max_threads"],
                  "split_tg_mem_bytes": row["tg_mem_bytes"],
                  "coop_tg_mem_bytes": other["tg_mem_bytes"]})
    if other["max_threads"] < row["max_threads"]:
        problems.append(
            f"{twin}: occupancy {other['max_threads']} below split "
            f"{row['max_threads']}")
    if other["tg_mem_bytes"] > row["tg_mem_bytes"]:
        problems.append(
            f"{twin}: threadgroup memory {other['tg_mem_bytes']} above split "
            f"{row['tg_mem_bytes']}")

payload = {
    "experiment": "e216-coop-weight-stream",
    "section": "stage0-census",
    "harness": "local",
    "rung": "A",
    "threadgroup_memory_bytes_planned": 0,
    "kernels": rows,
    "pairs": pairs,
    "problems": problems,
}
open(sys.argv[2], "w").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"pairs": len(pairs), "problems": problems}, indent=2))
sys.exit(1 if problems else 0)
PY
status=$?

{
  echo "tag=${tag}"
  echo "experiment=e216-coop-weight-stream"
  echo "section=stage0-census"
  echo "harness=local"
  echo "rung=A"
  echo "base_sha=$(git rev-parse HEAD)"
  echo "host=$(hostname)"
  echo "chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  echo "gpu_cores=$(ioreg -l 2>/dev/null \
    | LC_ALL=C sed -n 's/.*"gpu-core-count" = \([0-9][0-9]*\).*/\1/p' | head -1)"
  echo "os=$(sw_vers -productVersion)"
  echo "metal=$(xcrun -sdk macosx metal --version 2>&1 | head -1)"
  echo "cool_gate_passed_real_gate=false"
  echo "gate_qualified_for_timing=false"
  echo "official_or_ranked_score=false"
  echo "exit=${status}"
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${out}/meta.txt"

exit "${status}"
