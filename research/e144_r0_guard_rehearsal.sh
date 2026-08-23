#!/usr/bin/env bash
# E144 R-0: rehearse the senpai/submit-official.sh dirty-path gates against a
# committed in-branch mtp-head/ tree, WITHOUT calling `yukon submit`.
#
# Replays the three gates the guard runs last, verbatim from
# senpai/submit-official.sh: the skip-worktree/assume-unchanged index scan, the
# working-tree cleanliness scan, and the archive projection that `yukon submit`
# would enforce client-side.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

contract="$(cat benchmark.json)"
editable_paths=()
while IFS= read -r editable_path; do
  editable_paths+=("${editable_path}")
done < <(jq -r '.editablePaths[]' <<<"${contract}")

protected_paths=(benchmark.json)
for editable_path in "${editable_paths[@]}"; do
  protected_paths+=("${editable_path}")
done

echo "== gate 1: index entries (skip-worktree / assume-unchanged) =="
index_entries="$(git ls-files -v -- "${protected_paths[@]}")"
bad_index="$(awk '$1 == "S" || $1 ~ /^[abcdefghijklmnopqrstuvwxyz]$/' <<<"${index_entries}" || true)"
if [[ -n "${bad_index}" ]]; then
  echo "FAIL: hidden index state under submitted paths"
  printf '%s\n' "${bad_index}"
else
  echo "PASS: $(wc -l <<<"${index_entries}" | tr -d ' ') entries, none S/lowercase"
fi

echo "== gate 2: working tree cleanliness under submitted paths =="
working_status="$(git status --porcelain=v1 --untracked-files=all \
  --ignored=matching -- "${protected_paths[@]}")"
if [[ -n "${working_status}" ]]; then
  echo "FAIL: dirty/untracked/ignored entries under submitted paths"
  printf '%s\n' "${working_status}" | head -20
else
  echo "PASS: clean"
fi

echo "== gate 3: yukon submit archive projection =="
# yukon.js createSubmissionArchive: tar --gzip over manifest.editablePaths from
# the working tree, read whole into memory, then
# SUBMISSION_ARCHIVE_MAX_BYTES = 25 MiB and
# SUBMISSION_ARCHIVE_MAX_EXPANDED_BYTES = 512 MiB.
archive="$(mktemp "${TMPDIR:-/tmp}/e144-r0-archive.XXXXXX.tar.gz")"
trap 'rm -f -- "${archive}"' EXIT
present_paths=()
for editable_path in "${editable_paths[@]}"; do
  [[ -e "${editable_path}" ]] && present_paths+=("${editable_path}")
done
tar -czf "${archive}" "${present_paths[@]}"
archive_bytes="$(wc -c < "${archive}" | tr -d ' ')"
expanded_bytes="$(find "${present_paths[@]}" -type f -print0 \
  | xargs -0 -n1 wc -c | awk '{ total += $1 } END { print total + 0 }')"
archive_cap=$((25 * 1024 * 1024))
expanded_cap=$((512 * 1024 * 1024))

printf 'archive   %12d B  cap %12d B  %s\n' \
  "${archive_bytes}" "${archive_cap}" \
  "$( ((archive_bytes <= archive_cap)) && echo PASS || echo FAIL)"
printf 'expanded  %12d B  cap %12d B  %s\n' \
  "${expanded_bytes}" "${expanded_cap}" \
  "$( ((expanded_bytes <= expanded_cap)) && echo PASS || echo FAIL)"
