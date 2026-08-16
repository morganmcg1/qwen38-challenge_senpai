#!/usr/bin/env bash
# Check the working tree's base-authorized editable surface before a costly run.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 GROWTH_BASE_SHA [CONTRACT_SHA]" >&2
  exit 2
fi
base_input="$1"
contract_input="${2:-$1}"

for named_input in "GROWTH_BASE_SHA:${base_input}" "CONTRACT_SHA:${contract_input}"; do
  input_name="${named_input%%:*}"
  input_value="${named_input#*:}"
  if [[ "${input_value}" == *[!0-9a-fA-F]* ]] \
    || { [[ ${#input_value} -ne 40 ]] && [[ ${#input_value} -ne 64 ]]; }
  then
    echo "editable budget: ${input_name} must be a full 40- or 64-character commit hash" >&2
    exit 2
  fi
done
for command_name in git jq find wc grep tr mktemp rm; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "editable budget: ${command_name} is required" >&2
    exit 2
  fi
done

if ! repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  echo "editable budget: run this command inside a git worktree" >&2
  exit 2
fi
cd "${repo_root}"
if ! base_sha="$(git rev-parse --verify --quiet "${base_input}^{commit}")"; then
  echo "editable budget: GROWTH_BASE_SHA ${base_input} is not a commit in this repository" >&2
  exit 2
fi
if ! contract_sha="$(git rev-parse --verify --quiet "${contract_input}^{commit}")"; then
  echo "editable budget: CONTRACT_SHA ${contract_input} is not a commit in this repository" >&2
  exit 2
fi
if ! contract="$(git show "${contract_sha}:benchmark.json" 2>/dev/null)"; then
  echo "editable budget: ${contract_sha} has no readable benchmark.json" >&2
  exit 2
fi
if ! jq -e '
  .editablePaths
  | type == "array" and length > 0
    and all(.[]; type == "string" and length > 0)
' >/dev/null <<<"${contract}"; then
  echo "editable budget: ${contract_sha}:benchmark.json has no usable editablePaths" >&2
  exit 2
fi

if ! review_policy="$(
  git show "${contract_sha}:.github/scripts/run-submission-static-review.sh" 2>/dev/null
)"; then
  echo "editable budget: ${contract_sha} has no trusted static-review policy" >&2
  exit 2
fi

policy_default() {
  local policy_name="$1" line value
  if [[ "$(grep -Ec "^${policy_name}=" <<<"${review_policy}")" != "1" ]]; then
    echo "editable budget: could not uniquely resolve ${policy_name} from trusted policy" >&2
    exit 2
  fi
  line="$(grep -E "^${policy_name}=" <<<"${review_policy}")"
  value="${line##*:-}"
  value="${value%%\}*}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "editable budget: ${policy_name} has no positive-integer default" >&2
    exit 2
  fi
  printf '%s\n' "${value}"
}

MAX_TOTAL_BYTES="$(policy_default MAX_BYTES)"
MAX_FILE_BYTES="$(policy_default MAX_FILE_BYTES)"
MAX_GROWTH_BYTES="$(policy_default MAX_GROWTH_BYTES)"

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

editable_paths=()
while IFS= read -r editable_path; do
  if ! valid_relative_path "${editable_path}"; then
    echo "editable budget: invalid editablePaths entry '${editable_path}'" >&2
    exit 2
  fi
  editable_paths+=("${editable_path}")
done < <(jq -r '.editablePaths[]' <<<"${contract}")

exempt_paths=()
while IFS= read -r exempt_path; do
  [[ -n "${exempt_path}" ]] || continue
  if ! valid_relative_path "${exempt_path}"; then
    echo "editable budget: invalid exemptPaths entry '${exempt_path}'" >&2
    exit 2
  fi
  exempt_paths+=("${exempt_path}")
done < <(jq -r '(.editableSurfaceByteBudget.exemptPaths // [])[]' <<<"${contract}")

exempt_max_bytes="$(jq -r '.editableSurfaceByteBudget.exemptPathMaxBytes // empty' <<<"${contract}")"
if (( ${#exempt_paths[@]} > 0 )); then
  if ! [[ "${exempt_max_bytes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "editable budget: exempt paths require a positive exemptPathMaxBytes" >&2
    exit 2
  fi
else
  exempt_max_bytes=0
fi

is_exempt() {
  local candidate="$1" exempt_path
  for exempt_path in ${exempt_paths[@]+"${exempt_paths[@]}"}; do
    if [[ "${candidate}" == "${exempt_path}" \
      || "${candidate}" == "${exempt_path}/"* ]]
    then
      return 0
    fi
  done
  return 1
}

temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/qwen38-budget.XXXXXX")"
trap 'rm -rf "${temporary_dir}"' EXIT
base_seen="${temporary_dir}/base-seen"
working_seen="${temporary_dir}/working-seen"
: > "${base_seen}"
: > "${working_seen}"

base_total=0
base_count=0
base_exempt_total=0
working_total=0
working_count=0
working_exempt_total=0

account_base_file() {
  local path="$1" bytes
  if grep -Fqx -- "${path}" "${base_seen}"; then
    return
  fi
  printf '%s\n' "${path}" >> "${base_seen}"
  bytes="$(git cat-file -s "${base_sha}:${path}")"
  base_count=$((base_count + 1))
  if is_exempt "${path}"; then
    base_exempt_total=$((base_exempt_total + bytes))
  else
    base_total=$((base_total + bytes))
  fi
}

account_working_file() {
  local path="$1" bytes
  if [[ "${path}" == *$'\n'* || "${path}" == *$'\r'* ]]; then
    echo "editable budget: control character in editable filename" >&2
    exit 1
  fi
  if grep -Fqx -- "${path}" "${working_seen}"; then
    return
  fi
  printf '%s\n' "${path}" >> "${working_seen}"
  bytes="$(wc -c < "${path}" | tr -d ' ')"
  working_count=$((working_count + 1))
  if is_exempt "${path}"; then
    working_exempt_total=$((working_exempt_total + bytes))
    if (( working_exempt_total > exempt_max_bytes )); then
      echo "editable budget: exempt paths are at least ${working_exempt_total} bytes; limit is ${exempt_max_bytes}" >&2
      exit 1
    fi
    return
  fi
  if (( bytes > MAX_FILE_BYTES )); then
    echo "editable budget: ${path} is ${bytes} bytes; per-file limit is ${MAX_FILE_BYTES}" >&2
    exit 1
  fi
  working_total=$((working_total + bytes))
  if (( working_total > MAX_TOTAL_BYTES )); then
    echo "editable budget: source surface is at least ${working_total} bytes; total limit is ${MAX_TOTAL_BYTES}" >&2
    exit 1
  fi
}

for editable_path in "${editable_paths[@]}"; do
  base_files="${temporary_dir}/base-files"
  working_files="${temporary_dir}/working-files"
  symlinks="${temporary_dir}/symlinks"
  if ! git ls-tree -r --name-only -z "${base_sha}" -- "${editable_path}" \
    > "${base_files}"
  then
    echo "editable budget: could not enumerate base path ${editable_path}" >&2
    exit 1
  fi
  while IFS= read -r -d '' base_file; do
    account_base_file "${base_file}"
  done < "${base_files}"

  if [[ -L "${editable_path}" ]]; then
    echo "editable budget: editable path is a symlink: ${editable_path}" >&2
    exit 1
  fi
  if [[ -f "${editable_path}" ]]; then
    account_working_file "${editable_path}"
    continue
  fi
  if [[ -d "${editable_path}" ]]; then
    if ! find "${editable_path}" -type l -print -quit > "${symlinks}"; then
      echo "editable budget: could not inspect symlinks under ${editable_path}" >&2
      exit 1
    fi
    if [[ -s "${symlinks}" ]]; then
      echo "editable budget: symlink found under editable directory ${editable_path}" >&2
      exit 1
    fi
    if ! find "${editable_path}" -type f -print0 > "${working_files}"; then
      echo "editable budget: could not enumerate working path ${editable_path}" >&2
      exit 1
    fi
    while IFS= read -r -d '' working_file; do
      account_working_file "${working_file}"
    done < "${working_files}"
  fi
done

growth=$((working_total - base_total))
if (( growth > MAX_GROWTH_BYTES )); then
  echo "editable budget: source surface grew ${growth} bytes from BASE_SHA; growth limit is ${MAX_GROWTH_BYTES}" >&2
  exit 1
fi

headroom=$((MAX_TOTAL_BYTES - working_total))
echo "editable budget OK: source=${working_total}/${MAX_TOTAL_BYTES} bytes headroom=${headroom} growth=${growth}/${MAX_GROWTH_BYTES} exempt=${working_exempt_total}/${exempt_max_bytes} files=${working_count} (growth base=${base_sha}; contract=${contract_sha}; base source=${base_total}, exempt=${base_exempt_total}, files=${base_count})"
