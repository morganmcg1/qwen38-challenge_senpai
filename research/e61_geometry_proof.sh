#!/usr/bin/env bash
# Research-only: prove, once per E61 session, that the two geometry levers
# actually reach the runtime worker.
#
#   research/e61_geometry_proof.sh [--tokens N]
#
# The check this replaces could not fail. `research/e61-run.sh` used to grep the
# leg log for the worker's low-memory startup notice, but `mtp-timed` never
# forwards worker stderr, so the line cannot appear under any profile and the
# grep only ever passed.
#
# Three arms on the SAME unpatched build, so no rebuild separates them:
#
#   A  profile=full, MLX_MAX_OPS_PER_BUFFER=50   the geometry every timed leg uses
#   B  profile=full, MLX_MAX_OPS_PER_BUFFER=8    must be measurably slower than A
#   C  DARKBLOOM_STARTUP_MEMORY_PROFILE=bogus    must exit non-zero
#
# A alone proves nothing. A against B proves the `MLX_` class governs the
# worker's MLX Device: `max_ops_per_buffer_` is overridden by
# `env::max_ops_per_buffer` at Vendor/.../metal/device.cpp:596, so a starved ops
# budget forces far more command-buffer commits. C proves the `DARKBLOOM_` name
# reaches the policy: RuntimeStartupMemoryPolicy.resolve has no fallback, only
# `default: preconditionFailure(...)`.
#
# Both names cross into the worker child by prefix allowlist
# (sanitizedRuntimeWorkerEnvironment, allowedPrefixes "DARKBLOOM_" and "MLX_").
#
# These arms are NOT timed evidence about any E61 hypothesis. They are short,
# uncounterbalanced, and B is deliberately pathological.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

tokens=64
while (($#)); do
  case "$1" in
    --tokens) tokens="$2"; shift 2 ;;
    *) echo "e61_geometry_proof.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done

out_dir="${repo_root}/.mlxfast-private/e61/geometry-proof"
mkdir -p "${out_dir}"
out="${out_dir}/e61-geometry-proof.json"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e61_geometry_proof.sh: worktree is dirty; refusing to run" >&2
  exit 1
fi
head_sha="$(git rev-parse HEAD)"

# Slowdown large enough that thermal drift and leg-to-leg noise cannot produce
# it. The measured same-arm session null in this experiment is under 0.2 %.
readonly MIN_SLOWDOWN_PCT=5.0

leg_seconds_per_token() {
  # The driver writes the candidate MTP leg's seconds per token into the score
  # JSON, not the log. Reading the artifact means a leg that produced no score
  # yields an empty value rather than a stale line scraped from text.
  local tag="$1"
  python3 - "${repo_root}/.mlxfast-private/e61/runs/${tag}/score-1.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if p.exists():
    print(json.loads(p.read_text())["metrics"]["mtp_seconds_per_token"])
PY
}

run_arm() {
  local name="$1" profile="$2" ops="$3"
  local log="${out_dir}/${name}.log"
  echo "=== e61_geometry_proof arm ${name}: profile=${profile} ops=${ops} ===" >&2
  env DARKBLOOM_STARTUP_MEMORY_PROFILE="${profile}" \
      MLX_MAX_OPS_PER_BUFFER="${ops}" \
      MLX_MAX_MB_PER_BUFFER=512 \
      E42_TOKENS="${tokens}" \
      research/e61-run.sh "geom_${name}" --legs 1 >"${log}" 2>&1
  echo "$?"
}

rc_a="$(run_arm a_ops50 full 50)"
spt_a="$(leg_seconds_per_token geom_a_ops50)"

rc_b="$(run_arm b_ops8 full 8)"
spt_b="$(leg_seconds_per_token geom_b_ops8)"

rc_c="$(run_arm c_bogus bogus 50)"

slowdown="nan"
if [[ -n "${spt_a}" && -n "${spt_b}" ]]; then
  slowdown="$(python3 -c "print(100.0*(${spt_b}-${spt_a})/${spt_a})")"
fi

ops_lever_proved=false
if [[ "${slowdown}" != "nan" ]] \
   && python3 -c "import sys; sys.exit(0 if ${slowdown} >= ${MIN_SLOWDOWN_PCT} else 1)"; then
  ops_lever_proved=true
fi
profile_lever_proved=false
[[ "${rc_c}" != "0" ]] && profile_lever_proved=true

python3 - "${out}" <<PY
import json, sys
payload = {
    "experiment": "e61",
    "control": "geometry-lever-reaches-the-worker",
    "head_sha": "${head_sha}",
    "tokens": ${tokens},
    "min_slowdown_pct_required": ${MIN_SLOWDOWN_PCT},
    "arms": {
        "a_ops50": {"profile": "full", "ops": 50, "rc": ${rc_a},
                    "mtp_seconds_per_token": ${spt_a:-None}},
        "b_ops8": {"profile": "full", "ops": 8, "rc": ${rc_b},
                   "mtp_seconds_per_token": ${spt_b:-None}},
        "c_bogus": {"profile": "bogus", "ops": 50, "rc": ${rc_c},
                    "mtp_seconds_per_token": None},
    },
    "b_vs_a_slowdown_pct": ${slowdown},
    "ops_lever_proved": ${ops_lever_proved^},
    "profile_lever_proved": ${profile_lever_proved^},
    "note": "arms are short and uncounterbalanced; they are a lever control, "
            "not timing evidence for any E61 hypothesis",
}
open(sys.argv[1], "w").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if [[ "${ops_lever_proved}" != "true" || "${profile_lever_proved}" != "true" ]]; then
  echo "e61_geometry_proof.sh: LEVER NOT PROVED (ops=${ops_lever_proved}," \
       "profile=${profile_lever_proved}); E61 timing numbers measured under an" \
       "unproven geometry" >&2
  exit 7
fi
echo "e61_geometry_proof.sh: both levers proved; wrote ${out}"
