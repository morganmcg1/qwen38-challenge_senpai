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
# Four arms on the SAME unpatched build, so no rebuild separates them:
#
#   a_ref     profile=full, ops=50, mb=512   the geometry every timed leg uses
#   b_mb4     profile=full, ops=50, mb=4     must be measurably slower than A
#   c_bogus   profile=bogus                  must exit non-zero
#   d_ops8    profile=full, ops=8,  mb=512   recorded null, not a gate
#
# Why the size budget and not the ops budget is the probe:
#
#   CommandEncoder::needs_commit() is
#       (buffer_ops_ > max_ops) || ((buffer_sizes_ >> 20) > max_mb)
#   an OR, and `buffer_sizes_` accumulates `a.data_size()` for every array the
#   encoder binds. This checkpoint is a 27B 4-bit model: lm_head alone is about
#   635 MB and one transformer layer binds roughly 185 MB, so at max_mb=512 the
#   size term reaches its bound first on essentially every commit and the ops
#   term never binds. An earlier run of this script measured exactly that: 8 ops
#   against 50 ops moved the leg by -0.48 %, the wrong sign and far inside
#   noise. An ops arm is therefore useless as a reachability probe, and it is
#   kept only as a recorded null. Starving `max_mb` drives the term that binds.
#
# Arm A alone proves nothing. A against B proves the `MLX_` class governs the
# worker's MLX Device: `max_mb_per_buffer_` is overridden by
# `env::max_mb_per_buffer` at device.cpp:597, which reads MLX_MAX_MB_PER_BUFFER
# at utils.h:186. C proves the `DARKBLOOM_` name reaches the policy:
# RuntimeStartupMemoryPolicy.resolve has no fallback, only
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
MIN_SLOWDOWN_PCT=5.0

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
  local name="$1" profile="$2" ops="$3" mb="$4"
  local log="${out_dir}/${name}.log"
  echo "=== e61_geometry_proof arm ${name}: profile=${profile} ops=${ops} mb=${mb} ===" >&2
  env DARKBLOOM_STARTUP_MEMORY_PROFILE="${profile}" \
      MLX_MAX_OPS_PER_BUFFER="${ops}" \
      MLX_MAX_MB_PER_BUFFER="${mb}" \
      E42_TOKENS="${tokens}" \
      research/e61-run.sh "geom_${name}" --legs 1 >"${log}" 2>&1
  echo "$?"
}

rc_a="$(run_arm a_ref full 50 512)"
spt_a="$(leg_seconds_per_token geom_a_ref)"

rc_b="$(run_arm b_mb4 full 50 4)"
spt_b="$(leg_seconds_per_token geom_b_mb4)"

rc_c="$(run_arm c_bogus bogus 50 512)"

rc_d="$(run_arm d_ops8 full 8 512)"
spt_d="$(leg_seconds_per_token geom_d_ops8)"

python3 - \
  "${out}" "${head_sha}" "${tokens}" "${MIN_SLOWDOWN_PCT}" \
  "${rc_a}" "${spt_a}" "${rc_b}" "${spt_b}" "${rc_c}" "${rc_d}" "${spt_d}" <<'PY'
import json, sys

(out, head_sha, tokens, min_slow,
 rc_a, spt_a, rc_b, spt_b, rc_c, rc_d, spt_d) = sys.argv[1:12]


def num(v):
    return float(v) if v.strip() else None


tokens, min_slow = int(tokens), float(min_slow)
a, b, d = num(spt_a), num(spt_b), num(spt_d)

slowdown = 100.0 * (b - a) / a if (a and b) else None
ops_null = 100.0 * (d - a) / a if (a and d) else None

mb_lever_proved = slowdown is not None and slowdown >= min_slow
profile_lever_proved = rc_c != "0"

payload = {
    "experiment": "e61",
    "control": "geometry-lever-reaches-the-worker",
    "head_sha": head_sha,
    "tokens": tokens,
    "min_slowdown_pct_required": min_slow,
    "arms": {
        "a_ref": {"profile": "full", "ops": 50, "mb": 512,
                  "rc": int(rc_a), "mtp_seconds_per_token": a},
        "b_mb4": {"profile": "full", "ops": 50, "mb": 4,
                  "rc": int(rc_b), "mtp_seconds_per_token": b},
        "c_bogus": {"profile": "bogus", "ops": 50, "mb": 512,
                    "rc": int(rc_c), "mtp_seconds_per_token": None},
        "d_ops8": {"profile": "full", "ops": 8, "mb": 512,
                   "rc": int(rc_d), "mtp_seconds_per_token": d},
    },
    "b_vs_a_slowdown_pct": slowdown,
    "d_vs_a_ops_null_pct": ops_null,
    "mb_lever_proved": mb_lever_proved,
    "profile_lever_proved": profile_lever_proved,
    "ops_budget_is_not_binding": True,
    "ops_note": (
        "needs_commit() is (buffer_ops_ > max_ops) || ((buffer_sizes_ >> 20) > "
        "max_mb). On a 27B 4-bit checkpoint the size term reaches its bound "
        "first, so the ops budget never binds and cannot probe reachability. "
        "d_ops8 is recorded as a null, not as a gate."
    ),
    "note": (
        "arms are short and uncounterbalanced; they are a lever control, not "
        "timing evidence for any E61 hypothesis"
    ),
}
open(out, "w").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
sys.exit(0 if (mb_lever_proved and profile_lever_proved) else 7)
PY
rc=$?

if ((rc != 0)); then
  echo "e61_geometry_proof.sh: LEVER NOT PROVED; E61 timing numbers measured" \
       "under an unproven geometry" >&2
  exit 7
fi
echo "e61_geometry_proof.sh: both levers proved; wrote ${out}"
