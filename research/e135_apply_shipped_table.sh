#!/bin/bash
# F13 item 6, the held commit. Apply only when the `572b2cc4` published score
# is at or below 3.670.
#
# Under the tight grid the one-pass table is a MEASURED LOSS of +0.2649 % on
# medpair weights, so reverting the compiled default to `shipped` is worth
# about a quarter of a percent. Two literals move together, and
# `defaultRouteWitnessNamesTheCompiledDefaults` fails if they drift apart:
# the compiled default table and the route witness string that the pipeline
# log reports.
#
# The rebuild assertion needle changes with the witness, so the caller must
# pass the new route witness to `senpai/rebuild-and-assert-worker.sh`.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

src="Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift"
old_table="public static let compiledDefault = Table.onePass67"
new_table="public static let compiledDefault = Table.shipped"
old_witness='public static let defaultRouteWitness = "e120_default_route/tiered_switch/onepass67"'
new_witness='public static let defaultRouteWitness = "e120_default_route/tiered_switch/shipped"'

for needle in "${old_table}" "${old_witness}"; do
  n="$(grep -c -F -- "${needle}" "${src}")"
  if [[ "${n}" -ne 1 ]]; then
    echo "FAIL: '${needle}' appears ${n} times in ${src}; expected exactly 1" >&2
    exit 1
  fi
done

perl -0pi -e "s/\Q${old_table}\E/${new_table}/" "${src}"
perl -0pi -e "s/\Q${old_witness}\E/${new_witness}/" "${src}"

grep -n "compiledDefault = Table\.\|defaultRouteWitness = " "${src}"
echo
echo "Applied. The rebuild assertion must now require"
echo "  'e120_default_route/tiered_switch/shipped'"
echo "and forbid 'e120_default_route/tiered_switch/onepass67'."
