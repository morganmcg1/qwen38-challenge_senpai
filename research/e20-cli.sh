#!/usr/bin/env bash
# Research-only passthrough for MLXFAST_SWIFT_BIN (qwen38-r1-e20).
#
# Same idea as research/capture-cli.sh -- keep the per-verb CLI report the
# benchmark wrapper's EXIT trap would delete -- plus one thing E20 needs that
# the shared wrapper cannot provide: a per-verb attribution sink.
#
# --local-iterate drives four model-holding legs (correctness, mtp-verify,
# serial mtp-timed, MTP mtp-timed). The worker only ever sees MLX_QWEN_ATTRIB_OUT
# and suffixes it with its own pid, so without a per-verb stem all four legs
# land in one directory with nothing but mtime to say which is the width-varied
# decode. Deriving the stem here, where the verb is in argv, makes that mapping
# recorded rather than inferred.
set -uo pipefail

real="${MLXFAST_CAPTURE_REAL_BIN:?research/e20-cli.sh: set MLXFAST_CAPTURE_REAL_BIN}"
keep="${MLXFAST_CAPTURE_DIR:?research/e20-cli.sh: set MLXFAST_CAPTURE_DIR}"
mkdir -p "${keep}"

verb="${1:-noverb}"
seq_file="${keep}/.seq"
seq=$(( $(cat "${seq_file}" 2>/dev/null || echo 0) + 1 ))
printf '%s' "${seq}" > "${seq_file}"

stem="${keep}/$(printf '%02d' "${seq}")-${verb//\//_}"

# Depth 0 and depth N both arrive as `mtp-timed`; the offered depth is what
# separates the serial control from the speculative leg.
depth=""
prev=""
for arg in "$@"; do
  [[ "${prev}" == "--mtp-depth" ]] && depth="${arg}"
  prev="${arg}"
done
[[ -n "${depth}" ]] && stem="${stem}-d${depth}"

printf '%s\n' "$@" > "${stem}.argv"

if [[ -n "${MLX_QWEN_ATTRIB:-}" && "${MLX_QWEN_ATTRIB}" != "0" ]]; then
  export MLX_QWEN_ATTRIB_OUT="${stem}-attrib"
fi

"${real}" "$@" | tee "${stem}.json"
status="${PIPESTATUS[0]}"

out_path=""
prev=""
for arg in "$@"; do
  [[ "${prev}" == "--output" ]] && out_path="${arg}"
  prev="${arg}"
done
[[ -n "${out_path}" && -f "${out_path}" ]] && cp "${out_path}" "${stem}-output.json"

exit "${status}"
