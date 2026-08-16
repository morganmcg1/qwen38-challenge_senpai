#!/usr/bin/env bash
# Prove campaign ownership of AGENTS.md and the organizer-plus-campaign shape of
# .gitignore.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 UPSTREAM_SHA" >&2
  exit 2
fi
upstream_input="$1"
if [[ "${upstream_input}" == *[!0-9a-fA-F]* || ${#upstream_input} -ne 40 ]]; then
  echo "campaign overlay: UPSTREAM_SHA must be a full 40-character commit hash" >&2
  exit 2
fi
for command_name in git awk cmp mktemp; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "campaign overlay: ${command_name} is required" >&2
    exit 2
  fi
done

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "campaign overlay: run inside a git worktree" >&2
  exit 2
}
cd "${repo_root}"
upstream_sha="$(git rev-parse --verify --quiet "${upstream_input}^{commit}")" || {
  echo "campaign overlay: ${upstream_input} is not a local commit" >&2
  exit 2
}

temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/qwen38-overlay.XXXXXX")"
trap 'rm -rf -- "${temporary_dir}"' EXIT

strip_block() {
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

verify_file() {
  local path="$1" begin="$2" end="$3" label
  label="${path//\//_}"
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "campaign overlay: ${path} must be a regular file" >&2
    exit 1
  fi
  git show "${upstream_sha}:${path}" > "${temporary_dir}/${label}.upstream"
  if ! strip_block \
    "${path}" "${temporary_dir}/${label}.stripped" "${begin}" "${end}"
  then
    echo "campaign overlay: ${path} has malformed Senpai markers" >&2
    exit 1
  fi
  if ! cmp -s \
    "${temporary_dir}/${label}.upstream" "${temporary_dir}/${label}.stripped"
  then
    echo "campaign overlay: ${path} differs outside its Senpai block" >&2
    exit 1
  fi
}

verify_campaign_owned_file() {
  local path="$1" begin="$2" end="$3" label
  label="${path//\//_}"
  if [[ ! -f "${path}" || -L "${path}" ]]; then
    echo "campaign overlay: ${path} must be a regular file" >&2
    exit 1
  fi
  if ! strip_block \
    "${path}" "${temporary_dir}/${label}.outside" "${begin}" "${end}"
  then
    echo "campaign overlay: ${path} has malformed campaign markers" >&2
    exit 1
  fi
  if [[ -s "${temporary_dir}/${label}.outside" ]]; then
    echo "campaign overlay: ${path} must be wholly campaign-owned" >&2
    exit 1
  fi
}

verify_campaign_owned_file \
  AGENTS.md '<!-- SENPAI-CAMPAIGN-BEGIN -->' '<!-- SENPAI-CAMPAIGN-END -->'
verify_file \
  .gitignore '# SENPAI-CAMPAIGN-BEGIN' '# SENPAI-CAMPAIGN-END'

echo "campaign overlay OK: AGENTS.md is campaign-owned and .gitignore matches ${upstream_sha} plus its marked block"
