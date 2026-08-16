#!/usr/bin/env bash
# Configure safe campaign/organizer remotes and the pinned Yukon benchmark link.
set -euo pipefail

ORIGIN_URL="https://github.com/morganmcg1/qwen38-challenge_senpai.git"
UPSTREAM_URL="https://github.com/Layr-Labs/qwen-3.8-mtp-challenge"
BENCHMARK_ID="5d1ee4d7-80bd-4555-b182-6505f26ef495"
BENCHMARK_NAME="eigenlabs/qwen38-challenge"

for command_name in git jq; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "bootstrap checkout: ${command_name} is required" >&2
    exit 2
  fi
done
repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "bootstrap checkout: run inside the Senpai Git worktree" >&2
  exit 2
}
cd "${repo_root}"
if [[ ! -f senpai/frontier-state.json ]]; then
  echo "bootstrap checkout: senpai/frontier-state.json is missing" >&2
  exit 1
fi
if ! jq -e \
  --arg benchmark_id "${BENCHMARK_ID}" \
  --arg benchmark_name "${BENCHMARK_NAME}" \
  --arg upstream_url "${UPSTREAM_URL}" '
    .schemaVersion == 1
    and (.observedAt | type == "string" and test("^[0-9]{4}-[0-9]{2}-[0-9]{2}T"))
    and .benchmark.id == $benchmark_id
    and .benchmark.name == $benchmark_name
    and .organizer.url == $upstream_url
    and .organizer.branch == "main"
    and (.organizer.syncedCommit | type == "string" and test("^[0-9a-f]{40}$"))
  ' senpai/frontier-state.json >/dev/null
then
  echo "bootstrap checkout: frontier state does not match the Qwen campaign" >&2
  exit 1
fi
synced_commit="$(jq -r '.organizer.syncedCommit' senpai/frontier-state.json)"

normalize_url() {
  local value="$1"
  value="${value%.git}"
  printf '%s\n' "${value%/}"
}

if ! current_origin="$(git remote get-url origin 2>/dev/null)"; then
  echo "bootstrap checkout: origin is missing" >&2
  exit 1
fi
if [[ "$(normalize_url "${current_origin}")" != "$(normalize_url "${ORIGIN_URL}")" ]]; then
  echo "bootstrap checkout: origin points somewhere other than ${ORIGIN_URL}" >&2
  echo "bootstrap checkout: inspect and rename remotes manually; nothing changed" >&2
  exit 1
fi

if current_upstream="$(git remote get-url upstream 2>/dev/null)"; then
  if [[ "$(normalize_url "${current_upstream}")" != "$(normalize_url "${UPSTREAM_URL}")" ]]; then
    echo "bootstrap checkout: existing upstream points somewhere unexpected" >&2
    exit 1
  fi
else
  git remote add upstream "${UPSTREAM_URL}"
fi

git remote set-url origin "${ORIGIN_URL}"
git remote set-url --push origin "${ORIGIN_URL}"
git remote set-url upstream "${UPSTREAM_URL}"
git remote set-url --push upstream DISABLED

git config yukon.benchmark-id "${BENCHMARK_ID}"
git config yukon.benchmark-name "${BENCHMARK_NAME}"
git config yukon.source-url "${UPSTREAM_URL}"
git config yukon.source-branch main
git config yukon.source-ref "${synced_commit}"

echo "bootstrap checkout OK"
git remote -v
git config --get-regexp '^yukon\.(benchmark-id|benchmark-name|source-url|source-branch|source-ref)$'
