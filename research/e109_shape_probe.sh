#!/usr/bin/env bash
# E109 rung 1a/1b: run one threadgroup-shape sweep and collect every piece of
# evidence the verdict needs in one JSON.
#
#   usage: research/e109_shape_probe.sh FAMILY [OUT_DIR]
#          FAMILY is `prework` (rung 1a) or `qkrope` (rung 1b)
#
# Three things are gathered per arm:
#
#   time      research/e109_shape_probe.m, ABBA-interleaved inside one session
#   machine   pipeline limits from Metal (max threads, execution width,
#             threadgroup memory)
#   ISA       registers, spill bytes and machine-code digest from the real AGX
#             backend for BOTH the local g16s and the ranked g17s, so a shape
#             that only wins because it spills on one part is visible
#
# The occupancy hypothesis is not readable from time alone. `waves_over_cores`
# converts each arm's threadgroup count into how many rounds of work the 20
# local GPU cores must take, which is what H1 predicts the curve follows.
#
# Research-only: nothing here is on the scored path, and no arm is a candidate.
set -euo pipefail

family="${1:-}"
case "${family}" in
  prework|qkrope) ;;
  *) echo "usage: $0 prework|qkrope [OUT_DIR]" >&2; exit 2 ;;
esac
out_dir="${2:-research/out/e109-shape-${family}}"
arms_dir="${out_dir}/arms"
bin="${out_dir}/probe"
reps="${E109_SHAPE_REPS:-24}"
inner="${E109_SHAPE_INNER:-32}"
cores="${E109_GPU_CORES:-20}"

mkdir -p "${arms_dir}"

echo "e109_shape_probe: generating ${family} arms"
python3 research/e109_shape_arms.py --outdir "${arms_dir}" --family "${family}"

echo "e109_shape_probe: building the timing harness"
clang -fobjc-arc -O2 -Wno-deprecated-declarations \
  -framework Metal -framework Foundation \
  -o "${bin}" research/e109_shape_probe.m

# One metallib holding every arm of the family. `metal-tt` needs a metallib,
# and translating all arms from one library keeps the compile flags identical
# across the sweep.
echo "e109_shape_probe: translating arms for g16s and g17s"
airs=()
for src in "${arms_dir}"/arm_${family}_*.metal; do
  air="${src%.metal}.air"
  xcrun -sdk macosx metal -std=metal4.0 -O2 -fno-fast-math \
    -I "${arms_dir}" -c "${src}" -o "${air}"
  airs+=("${air}")
done
xcrun -sdk macosx metallib "${airs[@]}" -o "${out_dir}/arms.metallib"
python3 research/agx_crossarch.py census \
  --metallib "${out_dir}/arms.metallib" > "${out_dir}/census.txt"

echo "e109_shape_probe: timing (reps=${reps} inner=${inner})"
"${bin}" --spec "${arms_dir}/spec_${family}.json" \
  --out "${out_dir}/timing.json" --reps "${reps}" --inner "${inner}" \
  2> "${out_dir}/timing.log"
sed 's/^/  /' "${out_dir}/timing.log"

python3 research/e109_shape_verdict.py \
  --timing "${out_dir}/timing.json" \
  --census "${out_dir}/census.txt" \
  --spec "${arms_dir}/spec_${family}.json" \
  --cores "${cores}" \
  --out "${out_dir}/verdict.json"
