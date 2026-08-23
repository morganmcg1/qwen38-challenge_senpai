#!/usr/bin/env bash
# E149 arm A -- positive control that can fail, for the leaf-width override.
#
# Rule 114 says read the arm from the run's own behaviour, never from the
# variable you asked with. The timed legs run with `E128_NO_TRACE=1`, so the
# leaf width leaves no trace line. This control supplies the missing witness
# from a failure that is reachable only when the value arrives.
#
#   MLX_E141_ROWS_PER_LEAF=12 passes the override validator at
#   Qwen35.swift:4233-4243 (multiple of four, inside [4, 32]) and then fails
#   the divisibility guard inside `activeClusterRowsPerLeaf` at
#   Qwen35.swift:5591-5604, because 98,336 % 12 == 8. The fatalError therefore
#   fires ONLY if the value reached the derived-cluster construction in the
#   worker process. An ignored or dropped variable exits 0.
#
# Expected: non-zero exit and the divisibility fatalError in stderr.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

out="research/e149-leaf-witness.json"
runs="runs-e149-leafwitness"
width="${E149_WITNESS_WIDTH:-12}"

export MLX_E141_ROWS_PER_LEAF="${width}"
export MLX_E134_DEPTH_PRICE_ARM=ship
export E128_FORCE=1 E128_NO_TRACE=1 E128_TOKENS=16 E128_DEPTH=8
export E128_RUNS_DIR="${runs}"

# The guard fires while the worker builds the derived index, which happens on
# the reference pass before any timed leg writes its own stderr.log, so keep
# the whole session transcript instead of one leg's log.
session_log="$(mktemp -t e149-leaf-witness)"
research/e128_session.sh benchfixture >"${session_log}" 2>&1
rc=$?
cat "${session_log}"

hit=0
if grep -q "MLX_E141_ROWS_PER_LEAF=${width} does not divide" "${session_log}"; then
  hit=1
fi
rm -f "${session_log}"

python3 - "$rc" "$hit" "$width" "$out" <<'PY'
import json, sys
rc, hit, width, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
passed = bool(hit) and rc != 0
json.dump({
    "experiment": "e149-arm-a-leaf-override-positive-control",
    "harness": "local",
    "width_requested": width,
    "session_exit_code": rc,
    "divisibility_fatalerror_observed": bool(hit),
    "e149_leaf_override_positive_control_passed": passed,
    "guard_source": "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:5591-5604",
    "override_source": "Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift:4233-4243",
}, open(out, "w"), indent=1)
print(json.dumps(json.load(open(out)), indent=1))
PY

exit 0
