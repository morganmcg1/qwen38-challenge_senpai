#!/usr/bin/env bash
# E207 session: matched legs of the EMA-subtraction arm.
#
# STATUS: retained as a design record only. The advisor cancelled stage 1 and
# the `DARKBLOOM_E207_EMA_ARM` arm was removed from the scored surface for a
# research-only merge (RULE 393), so `frozen` legs run the SHIPPED policy
# today and the two arms are identical. Restore the arm commit before using
# this runner for a real contrast.
#
#   usage: research/e207_session.sh TAG_PREFIX TOKENS ARM[,ARM...]
#
#   ARM is `frozen` (DARKBLOOM_E207_EMA_ARM=frozen, the EMAs keep their seed
#   prior) or `live` (the shipped update loop). One leg runs per listed arm, in
#   the listed order, so an ABBA session is `live,frozen,frozen,live`.
#
# RULE 396: the two arms choose different depths by construction, so their
# (d, acc) trajectories differ and no round-index pairing is admissible. Every
# arm therefore gets its OWN leg and the contrast is a leg-level endpoint
# contrast on absolute candidate MTP seconds per token (RULE 394).
#
# The worker is built and witnessed BEFORE the first leg and its digest is
# compared after the last one, so a leg that timed another build is visible.
# Both arms run the same binary: the arm is an environment switch on an
# allowlisted `DARKBLOOM_` name (RULE 391(a)) with a per-round witness
# (`e207=`, `ema0_post=`) compiled into both arms (RULE 391(b), (c)).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prefix="${1:?usage: e207_session.sh TAG_PREFIX TOKENS ARM[,ARM...]}"
tokens="${2:?usage: e207_session.sh TAG_PREFIX TOKENS ARM[,ARM...]}"
arms="${3:?usage: e207_session.sh TAG_PREFIX TOKENS ARM[,ARM...]}"

senpai/rebuild-and-assert-worker.sh --no-build \
  --require "DARKBLOOM_E207_EMA_ARM" \
  --require-symbol recordAcceptOutcome || exit 1

worker_before="$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker \
  | awk '{print $1}')"

status=0
index=0
IFS=',' read -r -a arm_list <<< "${arms}"
for arm in "${arm_list[@]}"; do
  index=$((index + 1))
  tag="${prefix}-${index}-${arm}"
  echo "== e207 leg ${tag} (${tokens} tokens) =="
  case "${arm}" in
    frozen) ( export DARKBLOOM_E207_EMA_ARM=frozen
              research/e90_leg.sh "${tag}" "${tokens}" ) ;;
    live)   ( unset DARKBLOOM_E207_EMA_ARM
              research/e90_leg.sh "${tag}" "${tokens}" ) ;;
    *) echo "e207_session.sh: unknown arm '${arm}'" >&2; exit 2 ;;
  esac
  leg_status=$?
  {
    echo "experiment=e207-ema-subtraction"
    echo "e207_arm=${arm}"
    echo "e207_leg_index=${index}"
    echo "e207_arm_env=DARKBLOOM_E207_EMA_ARM=$([[ ${arm} == frozen ]] \
      && echo frozen || echo '<unset>')"
    echo "e207_instrument=arm-witness-e207-and-ema0_post-both-arms"
    echo "e207_worker_sha256_session=${worker_before}"
  } >> "research/out/${tag}/meta.txt"
  status=$(( status | leg_status ))
done

worker_after="$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker \
  | awk '{print $1}')"
echo "worker_sha256_before ${worker_before}"
echo "worker_sha256_after  ${worker_after}"
if [[ "${worker_before}" != "${worker_after}" ]]; then
  echo "e207_session: FATAL worker changed between legs" >&2
  exit 1
fi

echo "e207_session: exit ${status}"
exit "${status}"
