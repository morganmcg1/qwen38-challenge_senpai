#!/usr/bin/env bash
# Refuse an official Yukon submission unless its recorded campaign base is current.
set -euo pipefail

EXPECTED_BENCHMARK_ID="5d1ee4d7-80bd-4555-b182-6505f26ef495"
EXPECTED_BENCHMARK_NAME="eigenlabs/qwen38-challenge"
EXPECTED_ORIGIN_URL="https://github.com/morganmcg1/qwen38-challenge_senpai.git"
EXPECTED_UPSTREAM_URL="https://github.com/Layr-Labs/qwen-3.8-mtp-challenge"
EXPECTED_UPSTREAM_PUSH_URL="DISABLED"
SOURCE_BRANCH="main"
STATE_PATH="senpai/frontier-state.json"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 BASE_SHA [yukon submit arguments...]" >&2
  exit 2
fi
base_input="$1"
shift

if [[ "${base_input}" == *[!0-9a-fA-F]* ]] \
  || { [[ ${#base_input} -ne 40 ]] && [[ ${#base_input} -ne 64 ]]; }
then
  echo "official submit: BASE_SHA must be a full 40- or 64-character commit hash" >&2
  exit 2
fi

submit_args=()
has_model=0
has_note=0
while [[ $# -gt 0 ]]; do
  argument="$1"
  case "${argument}" in
    --model|--note|--note-file)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "official submit: ${argument} requires a non-empty value" >&2
        exit 2
      fi
      submit_args+=("${argument}" "$2")
      [[ "${argument}" == "--model" ]] && has_model=1 || has_note=1
      shift 2
      ;;
    --model=*|--note=*|--note-file=*)
      value="${argument#*=}"
      if [[ -z "${value}" ]]; then
        echo "official submit: ${argument%%=*} requires a non-empty value" >&2
        exit 2
      fi
      submit_args+=("${argument}")
      [[ "${argument}" == --model=* ]] && has_model=1 || has_note=1
      shift
      ;;
    --track|--track=*)
      echo "official submit: benchmark.json is schema v1; --track is invalid" >&2
      exit 2
      ;;
    --*)
      echo "official submit: unsupported Yukon option '${argument}'" >&2
      exit 2
      ;;
    *)
      echo "official submit: an explicit benchmark argument is forbidden" >&2
      echo "official submit: this wrapper is pinned to ${EXPECTED_BENCHMARK_NAME}" >&2
      exit 2
      ;;
  esac
done
if [[ "${has_model}" == "0" ]]; then
  echo "official submit: pass Yukon's exact fully qualified --model value" >&2
  exit 2
fi
if [[ "${has_note}" == "0" ]]; then
  echo "official submit: pass a reviewed public --note or --note-file" >&2
  exit 2
fi

for command_name in git jq yukon awk cmp mktemp rm; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "official submit: ${command_name} is required" >&2
    exit 2
  fi
done

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "official submit: run this command inside a git worktree" >&2
  exit 2
}
cd "${repo_root}"
base_sha="$(git rev-parse --verify --quiet "${base_input}^{commit}")" || {
  echo "official submit: BASE_SHA ${base_input} is not a local commit" >&2
  exit 2
}

for remote_name in origin upstream; do
  if ! git remote get-url "${remote_name}" >/dev/null 2>&1; then
    echo "official submit: required remote '${remote_name}' is not configured" >&2
    exit 1
  fi
done

normalize_url() {
  local value="$1"
  value="${value%.git}"
  printf '%s\n' "${value%/}"
}

benchmark_id="$(git config --get yukon.benchmark-id || true)"
benchmark_name="$(git config --get yukon.benchmark-name || true)"
source_url="$(git config --get yukon.source-url || true)"
source_branch="$(git config --get yukon.source-branch || true)"
origin_url="$(git remote get-url origin)"
origin_push_url="$(git remote get-url --push origin)"
upstream_url="$(git remote get-url upstream)"
upstream_push_url="$(git remote get-url --push upstream)"

if [[ "${benchmark_id}" != "${EXPECTED_BENCHMARK_ID}" \
  || "${benchmark_name}" != "${EXPECTED_BENCHMARK_NAME}" ]]
then
  echo "official submit: Yukon link is not pinned to ${EXPECTED_BENCHMARK_NAME}" >&2
  exit 1
fi
if [[ "$(normalize_url "${source_url}")" != "$(normalize_url "${EXPECTED_UPSTREAM_URL}")" \
  || "${source_branch}" != "${SOURCE_BRANCH}" ]]
then
  echo "official submit: Yukon source must be ${EXPECTED_UPSTREAM_URL} @ ${SOURCE_BRANCH}" >&2
  exit 1
fi
if [[ "${origin_url}" != "${EXPECTED_ORIGIN_URL}" \
  || "${origin_push_url}" != "${EXPECTED_ORIGIN_URL}" ]]
then
  echo "official submit: origin fetch/push URLs do not match the Senpai repository" >&2
  exit 1
fi
if [[ "$(normalize_url "${upstream_url}")" != "$(normalize_url "${EXPECTED_UPSTREAM_URL}")" \
  || "${upstream_push_url}" != "${EXPECTED_UPSTREAM_PUSH_URL}" ]]
then
  echo "official submit: upstream fetch URL or disabled push URL is incorrect" >&2
  exit 1
fi

fetch_remote() {
  local remote_name="$1"
  local tracking_ref="refs/remotes/${remote_name}/${SOURCE_BRANCH}"
  local fetch_args=(--no-tags "${remote_name}" "+refs/heads/${SOURCE_BRANCH}:${tracking_ref}")
  if [[ "${remote_name}" == "origin" && -f "$(git rev-parse --git-path shallow)" ]]; then
    fetch_args=(--no-tags --unshallow "${remote_name}" "+refs/heads/${source_branch}:${tracking_ref}")
  fi
  git fetch "${fetch_args[@]}"
}

if ! fetch_remote origin; then
  echo "official submit: could not refresh origin/${source_branch}; no submission was sent" >&2
  exit 1
fi
if ! fetch_remote upstream; then
  echo "official submit: could not refresh upstream/${source_branch}; no submission was sent" >&2
  exit 1
fi

main_sha="$(git rev-parse --verify "refs/remotes/origin/${SOURCE_BRANCH}^{commit}")"
upstream_sha="$(git rev-parse --verify "refs/remotes/upstream/${SOURCE_BRANCH}^{commit}")"

if ! git merge-base --is-ancestor "${base_sha}" HEAD; then
  echo "official submit: BASE_SHA ${base_sha} is not an ancestor of HEAD" >&2
  exit 1
fi
if ! git merge-base --is-ancestor "${base_sha}" "${main_sha}"; then
  echo "official submit: BASE_SHA is not an ancestor of current origin/${SOURCE_BRANCH}" >&2
  echo "official submit: campaign history moved; no submission was sent" >&2
  exit 1
fi

state="$(git show "${main_sha}:${STATE_PATH}" 2>/dev/null)" || {
  echo "official submit: current origin/${SOURCE_BRANCH} has no readable ${STATE_PATH}" >&2
  exit 1
}
if ! jq -e \
  --arg benchmark_id "${EXPECTED_BENCHMARK_ID}" \
  --arg benchmark_name "${EXPECTED_BENCHMARK_NAME}" \
  --arg source_url "$(normalize_url "${EXPECTED_UPSTREAM_URL}")" \
  --arg source_branch "${SOURCE_BRANCH}" '
    .schemaVersion == 1
    and (.observedAt | type == "string" and test("^[0-9]{4}-[0-9]{2}-[0-9]{2}T"))
    and .benchmark.id == $benchmark_id
    and .benchmark.name == $benchmark_name
    and ((.organizer.url | sub("[.]git$"; "") | sub("/$"; "")) == $source_url)
    and .organizer.branch == $source_branch
    and (.organizer.syncedCommit | type == "string" and test("^[0-9a-f]{40}$"))
    and (.promotedSubmission.id | type == "string" and length > 0)
    and (.promotedSubmission.sourceRef | type == "string" and test("^[0-9a-f]{40}$"))
    and (.promotedSubmission.score | type == "number")
  ' >/dev/null <<<"${state}"; then
  echo "official submit: ${STATE_PATH} is malformed or names another benchmark" >&2
  exit 1
fi

recorded_upstream_sha="$(jq -r '.organizer.syncedCommit' <<<"${state}")"
if ! git cat-file -e "${recorded_upstream_sha}^{commit}" 2>/dev/null \
  || ! git merge-base --is-ancestor "${recorded_upstream_sha}" "${upstream_sha}"
then
  echo "official submit: recorded organizer history is missing or was rewritten" >&2
  exit 1
fi

contract="$(git show "${main_sha}:benchmark.json" 2>/dev/null)" || {
  echo "official submit: current origin/${source_branch} has no readable benchmark.json" >&2
  exit 1
}
if ! jq -e '
  .schemaVersion == 1
  and (.editablePaths | type == "array" and length > 0
    and all(.[]; type == "string" and length > 0))
' >/dev/null <<<"${contract}"; then
  echo "official submit: current campaign manifest is not a usable schema-v1 contract" >&2
  exit 1
fi

if ! git diff --quiet "${main_sha}" "${upstream_sha}" -- benchmark.json; then
  echo "official submit: organizer benchmark.json differs from campaign main" >&2
  echo "official submit: sync organizer policy before submitting" >&2
  exit 1
fi

path_is_editable() {
  local candidate="$1" editable_path
  for editable_path in "${editable_paths[@]}"; do
    if [[ "${candidate}" == "${editable_path}" \
      || "${candidate}" == "${editable_path}/"* ]]
    then
      return 0
    fi
  done
  return 1
}

editable_paths=()
while IFS= read -r editable_path; do
  editable_paths+=("${editable_path}")
done < <(jq -r '.editablePaths[]' <<<"${contract}")

path_is_campaign_owned() {
  case "$1" in
    senpai/*|.agents/*|research/*|AGENTS.md|.gitignore) return 0 ;;
    *) return 1 ;;
  esac
}

campaign_trusted_drift=""
while IFS= read -r -d '' changed_path; do
  if ! path_is_editable "${changed_path}" \
    && ! path_is_campaign_owned "${changed_path}"
  then
    campaign_trusted_drift="${changed_path}"
    break
  fi
done < <(git diff --name-only -z "${recorded_upstream_sha}" "${main_sha}" --)
if [[ -n "${campaign_trusted_drift}" ]]; then
  echo "official submit: campaign main has unreviewed trusted drift at ${campaign_trusted_drift}" >&2
  exit 1
fi

overlay_dir="$(mktemp -d "${TMPDIR:-/tmp}/qwen38-submit-overlay.XXXXXX")"
trap 'rm -rf -- "${overlay_dir}"' EXIT

strip_campaign_block() {
  local source="$1" destination="$2" begin="$3" end="$4"
  awk -v begin="${begin}" -v end="${end}" '
    $0 == begin {
      if (inside || seen) exit 2
      inside = 1
      seen = 1
      next
    }
    $0 == end {
      if (!inside) exit 2
      inside = 0
      ended = 1
      next
    }
    !inside { print }
    END {
      if (!seen || inside || !ended) exit 2
    }
  ' "${source}" > "${destination}"
}

verify_campaign_overlay() {
  local path="$1" begin="$2" end="$3" label
  label="${path//\//_}"
  git show "${recorded_upstream_sha}:${path}" > "${overlay_dir}/${label}.upstream"
  git show "${main_sha}:${path}" > "${overlay_dir}/${label}.campaign"
  if ! strip_campaign_block \
    "${overlay_dir}/${label}.campaign" \
    "${overlay_dir}/${label}.stripped" \
    "${begin}" "${end}"
  then
    echo "official submit: ${path} has malformed campaign markers" >&2
    exit 1
  fi
  if ! cmp -s \
    "${overlay_dir}/${label}.upstream" "${overlay_dir}/${label}.stripped"
  then
    echo "official submit: ${path} differs from organizer outside its campaign block" >&2
    exit 1
  fi
}

verify_campaign_owned_file() {
  local path="$1" begin="$2" end="$3" label mode
  label="${path//\//_}"
  mode="$(git ls-tree "${main_sha}" -- "${path}" | awk '{print $1}')"
  case "${mode}" in
    100644|100755) ;;
    *)
      echo "official submit: ${path} must be a regular file in campaign main" >&2
      exit 1
      ;;
  esac
  git show "${main_sha}:${path}" > "${overlay_dir}/${label}.campaign"
  if ! strip_campaign_block \
    "${overlay_dir}/${label}.campaign" \
    "${overlay_dir}/${label}.outside" \
    "${begin}" "${end}"
  then
    echo "official submit: ${path} has malformed campaign markers" >&2
    exit 1
  fi
  if [[ -s "${overlay_dir}/${label}.outside" ]]; then
    echo "official submit: ${path} must be wholly campaign-owned" >&2
    exit 1
  fi
}

verify_campaign_owned_file \
  AGENTS.md '<!-- SENPAI-CAMPAIGN-BEGIN -->' '<!-- SENPAI-CAMPAIGN-END -->'
verify_campaign_overlay \
  .gitignore '# SENPAI-CAMPAIGN-BEGIN' '# SENPAI-CAMPAIGN-END'
rm -rf -- "${overlay_dir}"
trap - EXIT

trusted_drift=""
while IFS= read -r -d '' changed_path; do
  if ! path_is_editable "${changed_path}"; then
    trusted_drift="${changed_path}"
    break
  fi
done < <(git diff --name-only -z "${recorded_upstream_sha}" "${upstream_sha}" --)
if [[ -n "${trusted_drift}" ]]; then
  echo "official submit: organizer trusted surface advanced at ${trusted_drift}" >&2
  echo "official submit: run the sync-organizer-frontier workflow first" >&2
  exit 1
fi

protected_paths=(benchmark.json)
for editable_path in "${editable_paths[@]}"; do
  protected_paths+=("${editable_path}")
done

if ! git diff --quiet "${main_sha}" "${base_sha}" -- "${protected_paths[@]}"; then
  echo "official submit: BASE_SHA submitted snapshot differs from current origin/${SOURCE_BRANCH}" >&2
  echo "official submit: replay and remeasure the candidate on the maintained frontier" >&2
  exit 1
fi
if ! git diff --quiet "${main_sha}" HEAD -- benchmark.json; then
  echo "official submit: benchmark.json differs from current campaign main" >&2
  exit 1
fi

if ! index_entries="$(git ls-files -v -- "${protected_paths[@]}")"; then
  echo "official submit: could not inspect submitted index entries" >&2
  exit 1
fi
while IFS=' ' read -r index_tag _; do
  case "${index_tag}" in
    S|[a-z])
      echo "official submit: skip-worktree/assume-unchanged is set under submitted paths" >&2
      exit 1
      ;;
  esac
done <<<"${index_entries}"

if ! working_status="$(
  git status --porcelain=v1 --untracked-files=all --ignored=matching \
    -- "${protected_paths[@]}"
)"; then
  echo "official submit: could not inspect the submitted working tree" >&2
  exit 1
fi
if [[ -n "${working_status}" ]]; then
  echo "official submit: commit or discard changes under benchmark.json/editablePaths first" >&2
  exit 1
fi

exec yukon submit "${submit_args[@]}"
