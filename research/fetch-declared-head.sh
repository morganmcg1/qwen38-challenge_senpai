#!/usr/bin/env bash
# Provision the head that `mtp-head.manifest.json` declares, locally.
#
# `setup-qwen-mtp.sh` only ever provisions the ORGANIZER-PINNED head
# (EigenLabs/Qwen3.8-27B-MTP-bf16, 849 MB); it never reads the declaration.
# The ranked workflow's "resolve the head declaration" step does, and hands the
# resolved tree to the CANDIDATE leg only. So an unmodified local
# `--local-iterate` measures the candidate schedule against the wrong head, and
# any cost constant fitted that way is fitted to a head the ranked run does not
# execute.
#
# This mirrors the workflow's fetch + verify byte for byte: single
# `model.safetensors` for an `hf:` source, then the tree-digest rule (sha256
# over "<file sha256>  <relative path>\n" in LC_ALL=C order, README.md
# excluded) checked against the manifest's `sha256`/`bytes`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

declaration="mtp-head.manifest.json"
dest="${1:-${HOME}/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head-declared}"

source_kind="$(jq -r '.source // "pinned"' "${declaration}")"
[[ "${source_kind}" == "remote" ]] || {
  echo "fetch-declared-head.sh: declaration source is '${source_kind}', not 'remote'" >&2
  exit 1
}
want_sha="$(jq -r '.sha256 // ""' "${declaration}")"
want_bytes="$(jq -r '.bytes // 0' "${declaration}")"
url="$(jq -r '.source_url // ""' "${declaration}")"
repo_rev="${url#hf:}"
repo="${repo_rev%@*}"
rev="${repo_rev##*@}"

mkdir -p "${dest}"
if [[ ! -f "${dest}/model.safetensors" ]]; then
  curl --fail --location --max-time 1800 \
    --output "${dest}/model.safetensors.part" \
    "https://huggingface.co/${repo}/resolve/${rev}/model.safetensors"
  mv "${dest}/model.safetensors.part" "${dest}/model.safetensors"
fi

got_bytes="$(find "${dest}" -type f ! -name README.md -print0 \
  | xargs -0 -n1 wc -c | awk '{ total += $1 } END { print total + 0 }')"
got_sha="$( (cd "${dest}" \
  && find . -type f ! -name README.md \
  | sed 's|^\./||' \
  | LC_ALL=C sort \
  | while read -r f; do
      printf '%s  %s\n' "$(shasum -a 256 "${f}" | awk '{print $1}')" "${f}"
    done) | shasum -a 256 | awk '{print $1}')"

[[ "${got_sha}" == "${want_sha}" ]] || {
  echo "fetch-declared-head.sh: digest mismatch: manifest ${want_sha}, tree ${got_sha}" >&2
  exit 1
}
[[ "${got_bytes}" == "${want_bytes}" ]] || {
  echo "fetch-declared-head.sh: byte mismatch: manifest ${want_bytes}, tree ${got_bytes}" >&2
  exit 1
}
echo "fetch-declared-head.sh: declared head verified at ${dest} (${got_bytes} bytes, sha256 ${got_sha})"

# --- local-wrapper run tree --------------------------------------------------
# The verified tree is ONE file, exactly as the ranked runner stages it, and
# `Qwen36MTPHeadAttachment.verifyHeadTree` supports that shape directly: with no
# `model.safetensors.index.json` present it takes the declared-head branch and
# validates structure from the safetensors header, never reading `config.json`.
# `benchmark-qwen-mtp.sh` still refuses a head directory without a non-empty
# `config.json`, so a run tree needs one. Adding it to ${dest} would change the
# tree digest and destroy the property that makes ${dest} checkable, hence a
# sibling: `model.safetensors` HARDLINKED (same inode, no second 427 MB) plus
# the organizer head-family `config.json`, which is inert on this code path.
run_dest="${dest}-run"
pinned_config="$(dirname "${dest}")/mtp-head/config.json"
[[ -s "${pinned_config}" ]] || {
  echo "fetch-declared-head.sh: ${pinned_config} is missing; run ./setup-qwen-mtp.sh" >&2
  exit 1
}
mkdir -p "${run_dest}"
rm -f "${run_dest}/model.safetensors"
ln "${dest}/model.safetensors" "${run_dest}/model.safetensors"
cp "${pinned_config}" "${run_dest}/config.json"
[[ ! -e "${run_dest}/model.safetensors.index.json" ]] || {
  echo "fetch-declared-head.sh: ${run_dest} carries an index.json; that would send the loader down the pinned-tree branch" >&2
  exit 1
}
echo "fetch-declared-head.sh: run tree staged at ${run_dest} (E11_HEAD_DIR / MLXFAST_QWEN_MTP_HEAD_DIR)"
