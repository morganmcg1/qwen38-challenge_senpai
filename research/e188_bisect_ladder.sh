#!/usr/bin/env bash
# E188 stage 2: the remaining commit-phase bisection legs, in one job.
#
# Two legs already ran, in this order: e188-head-a (HEAD), e188-pre165-a
# (806181de). This script appends the other six so the whole session is
# palindromic in arm order:
#
#   head  pre165 | e90 e134 e134 e90 | pre165 head
#
# A palindrome is the ABBA generalisation for four arms: every arm is as far
# before the session midpoint as it is after it, so monotone thermal drift
# cancels to first order in each arm mean. The legs are ungated by design
# (MLXFAST_LOCAL_COOL_GATE=0), which the program permits for counterbalanced
# local arms, and each leg records its own entry and exit GPU temperature.
#
# The four arms are the four distinct contents of the commit-phase window
# found by `e188_commit_phase_bisect.py`:
#
#   59b67f50  E90 anchor blob b946c17c
#   7a427dfa  first restoreAfterPrefixReject change, blob da7606c7
#   806181de  second restoreAfterPrefixReject change, blob f8482bbd (pre-E165)
#   HEAD      post-E165 window, blob 2457b231
#
# The run stops at the first failing leg so a broken arm cannot be read as a
# timing result.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TOKENS="${E188_TOKENS:-128}"

legs=(
  "59b67f50 e188-e90-a"
  "7a427dfa e188-e134-a"
  "7a427dfa e188-e134-b"
  "59b67f50 e188-e90-b"
  "806181de e188-pre165-b"
  "HEAD     e188-head-b"
)

for leg in "${legs[@]}"; do
  read -r rev tag <<<"${leg}"
  echo "=============== e188 ladder: ${tag} (${rev}) ==============="
  if ! research/e188_bisect_leg.sh "${rev}" "${tag}" "${TOKENS}"; then
    echo "e188 ladder: ${tag} FAILED, stopping the ladder" >&2
    exit 1
  fi
done

echo "e188 ladder: all legs finished"
