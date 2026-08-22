#!/usr/bin/env bash
# E138: run one isolated (M, IPG, RPS) plan-surface block and record the
# thermal envelope around it.
#
# The sweep loads no model and holds no worker, so HARNESS DEFECT 36 does not
# apply: `swift test` rebuilds the test binary it then runs. It does use the
# GPU, so the standing ungated-measurement conditions apply and this script
# records the entry and exit GPU temperature of every block next to the result.
#
#   usage: research/e138_plan_surface.sh OUT CELLS [SHAPES] [REPS] [INNER] [GRID]
#
# CELLS  is `m:ipg:rps` triples, comma separated.
# SHAPES is a comma separated subset of the seven scored shape names, or empty
#        for all seven.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

OUT="${1:?usage: e138_plan_surface.sh OUT CELLS [SHAPES] [REPS] [INNER] [GRID]}"
CELLS="${2:?a cell list is required}"
SHAPES="${3:-}"
REPS="${4:-15}"
INNER="${5:-10}"
GRID="${6:-tight}"

mkdir -p "$(dirname "$OUT")"

MACMON="${MLXFAST_MACMON_BIN:-$HOME/bin/macmon}"
gpu_temp() {
  [ -x "$MACMON" ] || return 0
  "$MACMON" pipe -s1 2>/dev/null | jq -r '.temp.gpu_temp_avg // empty'
}

# One model-holding process at a time. This block holds no model, but a
# forgotten worker would contaminate every sample, so it is reported.
RESIDENT="$(pgrep -f 'mlxfast-runtime-worker' | tr '\n' ' ')"

ENTRY_TEMP="$(gpu_temp)"
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

export MLXFAST_RUN_E138_PLAN_SURFACE=1
export MLXFAST_E138_PLAN_SURFACE_OUT="$ROOT/$OUT"
export MLXFAST_E138_CELLS="$CELLS"
export MLXFAST_E138_SHAPES="$SHAPES"
export MLXFAST_E138_REPS="$REPS"
export MLXFAST_E138_INNER="$INNER"
export MLXFAST_E138_GRID="$GRID"

swift test --force-resolved-versions --filter E138PlanSurface
STATUS=$?

EXIT_TEMP="$(gpu_temp)"
END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

python3 - "$OUT" "$ENTRY_TEMP" "$EXIT_TEMP" "$START" "$END" "$STATUS" \
    "$RESIDENT" <<'PY'
import json, pathlib, subprocess, sys

out, entry, exit_temp, start, end, status, resident = sys.argv[1:8]
side = pathlib.Path(out).with_suffix(".session.json")
side.write_text(json.dumps({
    "artifact": out,
    "started_utc": start,
    "ended_utc": end,
    "exit_status": int(status),
    "gpu_temp_entry_c": float(entry) if entry else None,
    "gpu_temp_exit_c": float(exit_temp) if exit_temp else None,
    # The three standing conditions for an ungated local timed arm. Arms are
    # ABBA-interleaved inside each timed block by the instrument itself.
    "cool_gate_passed_real_gate": False,
    "gate_qualified_for_timing": False,
    "official_or_ranked_score": False,
    "resident_worker_pids": resident.split(),
    "commit": subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip(),
    "dirty": bool(subprocess.run(["git", "status", "--porcelain"],
                                 capture_output=True, text=True).stdout.strip()),
}, indent=2, sort_keys=True) + "\n")
print("wrote %s" % side)
PY

exit "$STATUS"
