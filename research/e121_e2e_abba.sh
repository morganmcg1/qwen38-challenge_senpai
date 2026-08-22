#!/usr/bin/env bash
# E121 rung 3: the gated cross-simdgroup chunk-sum share against the unchanged
# base, ABBA-counterbalanced inside one session.
#
#   usage: research/e121_e2e_abba.sh [REPLICATES] [TOKENS] [LABEL] [FIRST]
#
# Each replicate runs base, share, share, base in that order. Both arms have
# mean position 2.5, so monotone thermal drift inside the replicate cancels to
# first order, and the two base legs bracket the replicate and give the session
# null directly.
#
# FIRST numbers the first replicate, so a later session EXTENDS an existing
# estimate instead of restarting it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

replicates="${1:-3}"
tokens="${2:-512}"
label="${3:-r3}"
first="${4:-1}"

# MATCHED THERMAL HISTORY. Counterbalancing only removes a confound that is a
# function of POSITION. The 2026-08-22T04:06Z session showed one that is a
# function of the ARM: a leg only rebuilds when its sources look newer than the
# last build, so position 3 - the one position whose arm repeats its
# predecessor's, and always `share` under this order - reached its timed run
# after 11-12 s instead of 50-53 s and entered 8 C hotter. `e121_e2e_leg.sh`
# now forces the compile on every leg and pads the remainder up to this floor,
# so prep is the same length and the same kind of work for every position. Set
# the floor a little above the natural rebuild so padding, not luck, closes the
# gap; the balance report below proves it worked.
export E121_EXPERIMENT="${E121_EXPERIMENT:-e121-rung3}"
export E121_PREP_FLOOR_SECONDS="${E121_PREP_FLOOR_SECONDS:-60}"

failures=0
for ((rep = first; rep < first + replicates; rep++)); do
  position=0
  for arm in base share share base; do
    position=$((position + 1))
    tag="e121${label}k${rep}p${position}${arm}"
    echo "=== ${tag}: arm=${arm} replicate=${rep} tokens=${tokens} ==="
    research/e121_e2e_leg.sh "${arm}" "${tag}" "${tokens}"
    status=$?
    {
      echo "e121_replicate=${rep}"
      echo "e121_position=${position}"
    } >> "research/out/${tag}/meta.txt"
    if ((status != 0)); then
      echo "e121_e2e_abba: ${tag} exited ${status}" >&2
      failures=$((failures + 1))
    fi
  done
done

echo "e121_e2e_abba: ${failures} failed legs"

# A matched design has to be checked, not asserted. This reports prep length and
# entry temperature per arm over the legs this session wrote, so a reader can
# see whether the two arms really did start from the same thermal state. It
# reports and does not fail the session: the legs are already measured, and the
# analysis needs the numbers to decide what the imbalance is worth.
E121_LABEL="${label}" E121_FIRST="${first}" E121_REPLICATES="${replicates}" \
python3 - <<'PY'
import os, re, pathlib, statistics
label, first, reps = os.environ["E121_LABEL"], int(os.environ["E121_FIRST"]), int(os.environ["E121_REPLICATES"])
legs = []
for rep in range(first, first + reps):
    for pos, arm in enumerate(["base", "share", "share", "base"], start=1):
        meta = pathlib.Path(f"research/out/e121{label}k{rep}p{pos}{arm}/meta.txt")
        if not meta.exists():
            continue
        m = dict(re.findall(r"^(\w+)=(.*)$", meta.read_text(), re.M))
        legs.append((arm, pos, float(m.get("e121_prep_seconds", "nan")),
                     float(m.get("gpu_temp_entry_c", "nan"))))
if not legs:
    raise SystemExit("e121_e2e_abba: balance report found no legs")

print("=== thermal balance ===")
print(f"{'arm':<8}{'n':<4}{'prep_s mean':<14}{'prep_s range':<16}{'entry_C mean':<14}{'entry_C range'}")
means = {}
for arm in ("base", "share"):
    v = [(p, t) for a, _, p, t in legs if a == arm]
    prep, temp = [x[0] for x in v], [x[1] for x in v]
    means[arm] = (statistics.fmean(prep), statistics.fmean(temp))
    print(f"{arm:<8}{len(v):<4}{means[arm][0]:<14.1f}"
          f"{f'{min(prep):.0f}-{max(prep):.0f}':<16}{means[arm][1]:<14.2f}"
          f"{min(temp):.2f}-{max(temp):.2f}")
d_prep = means["share"][0] - means["base"][0]
d_temp = means["share"][1] - means["base"][1]
print(f"share minus base: prep {d_prep:+.1f} s, entry {d_temp:+.2f} C")
print("MATCHED" if abs(d_temp) <= 1.0 else
      f"IMBALANCED: the arms did not start from the same thermal state "
      f"({d_temp:+.2f} C); price this against the measured effect before "
      f"reading the result")
PY

exit "${failures}"
