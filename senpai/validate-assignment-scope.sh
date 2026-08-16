#!/usr/bin/env bash
# Validate proposed Yukon-submitted paths against the experiment base contract.
set -euo pipefail

usage() {
  echo "usage: $0 BASE_SHA SUBMITTED_PATH [SUBMITTED_PATH ...]" >&2
}

if (( $# < 2 )); then
  usage
  exit 2
fi

base_input="$1"
shift

if [[ "${base_input}" == *[!0-9a-fA-F]* ]] \
  || { [[ ${#base_input} -ne 40 ]] && [[ ${#base_input} -ne 64 ]]; }
then
  echo "assignment scope: BASE_SHA must be a full 40- or 64-character commit hash" >&2
  exit 2
fi

for command_name in git jq; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "assignment scope: ${command_name} is required" >&2
    exit 2
  fi
done

if ! repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  echo "assignment scope: run this command inside a git worktree" >&2
  exit 2
fi
cd "${repo_root}"

if ! base_sha="$(git rev-parse --verify --quiet "${base_input}^{commit}")"; then
  echo "assignment scope: BASE_SHA ${base_input} is not a commit in this repository" >&2
  exit 2
fi
if ! contract="$(git show "${base_sha}:benchmark.json" 2>/dev/null)"; then
  echo "assignment scope: ${base_sha} has no readable benchmark.json" >&2
  exit 2
fi
if ! jq -e '
  .editablePaths
  | type == "array" and length > 0
    and all(.[]; type == "string" and length > 0)
' >/dev/null <<<"${contract}"; then
  echo "assignment scope: ${base_sha}:benchmark.json has no usable editablePaths" >&2
  exit 2
fi

valid_relative_path() {
  local path="$1"
  if [[ -z "${path}" || "${path}" == /* || "${path}" == :* \
    || "${path}" == *\\* || "${path}" == *$'\n'* || "${path}" == *$'\r'* ]]
  then
    return 1
  fi
  case "/${path}/" in
    *"//"*|*"/../"*|*"/./"*) return 1 ;;
  esac
}

while IFS= read -r editable_path; do
  if ! valid_relative_path "${editable_path}"; then
    echo "assignment scope: invalid editablePaths entry '${editable_path}' in base contract" >&2
    exit 2
  fi
done < <(jq -r '.editablePaths[]' <<<"${contract}")

bad=0
for proposed_path in "$@"; do
  if ! valid_relative_path "${proposed_path}"; then
    echo "assignment scope: invalid submitted path '${proposed_path}'" >&2
    bad=1
    continue
  fi

  allowed=0
  while IFS= read -r editable_path; do
    if [[ "${proposed_path}" == "${editable_path}" ]]; then
      allowed=1
      break
    fi
    editable_type="$(git cat-file -t "${base_sha}:${editable_path}" 2>/dev/null || true)"
    if [[ "${editable_type}" == "tree" && "${proposed_path}" == "${editable_path}/"* ]]; then
      allowed=1
      break
    fi
  done < <(jq -r '.editablePaths[]' <<<"${contract}")

  if [[ "${allowed}" == "0" ]]; then
    echo "assignment scope: '${proposed_path}' is outside ${base_sha}:benchmark.json editablePaths" >&2
    bad=1
  fi
done

if [[ "${bad}" != "0" ]]; then
  exit 1
fi

echo "assignment scope OK: $# submitted path(s) against BASE_SHA=${base_sha}"
