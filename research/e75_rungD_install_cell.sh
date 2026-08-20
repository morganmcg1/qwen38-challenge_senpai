#!/usr/bin/env bash
# Install one prebuilt E75 rung D cell into the build roots.
#
#   research/e75_rungD_install_cell.sh .mlxfast-private/e75-rungD/cells/CELL
#
# A 2x2 over {kernel dispatch table} x {depth price} switches the kernel table
# per leg. Rebuilding that in-session would recompile Cmlx and the 158 MiB
# metallib between legs, which is minutes of drift for no scientific gain, so
# every cell is built once before any timing starts and installed by copy here.
#
# The swap is only safe if it cannot put an unrelated build under measurement,
# so this script refuses unless:
#
#   * the three source files in the worktree hash to what the cell was built
#     from, which ties the installed binaries to the bytes the leg committed;
#   * every installed artifact hashes to what the prebuild recorded;
#   * the metallib's published fingerprint equals the fingerprint of the
#     vendored sources now in the tree. That is a CONTENT check and it is
#     strictly stronger than the mtime freshness test benchmark-qwen-mtp.sh
#     runs, which the copy would satisfy trivially.
set -euo pipefail

cell_dir="${1:?usage: e75_rungD_install_cell.sh CELL_DIR}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ "${cell_dir}" == /* ]] || cell_dir="${PWD}/${cell_dir}"
manifest="${cell_dir}/cell.json"
[[ -s "${manifest}" ]] || { echo "install-cell: no manifest at ${manifest}" >&2; exit 2; }

read_json() { python3 -c '
import json, sys
payload = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    payload = payload[key]
print(payload)
' "${manifest}" "$1"; }

digest() { shasum -a 256 "$1" | cut -d' ' -f1; }

# --- the worktree must carry the bytes this cell was built from ---------------
while read -r path want; do
  [[ -n "${path}" ]] || continue
  have="$(digest "${path}")"
  if [[ "${have}" != "${want}" ]]; then
    echo "install-cell: ${path} is ${have}, cell was built from ${want}" >&2
    exit 3
  fi
done < <(python3 -c '
import json, sys
for path, sha in sorted(json.load(open(sys.argv[1]))["source_sha256"].items()):
    print(path, sha)
' "${manifest}")

# --- install ------------------------------------------------------------------
mkdir -p .build/release .build-worker/release
install -m 0755 "${cell_dir}/mlxfast-runtime-worker" \
  .build-worker/release/mlxfast-runtime-worker
install -m 0755 "${cell_dir}/mlxfast-swift" .build/release/mlxfast-swift
for root in .build-worker/release .build/release; do
  install -m 0644 "${cell_dir}/mlx.metallib" "${root}/mlx.metallib"
  install -m 0644 "${cell_dir}/mlx.metallib.fingerprint" \
    "${root}/mlx.metallib.fingerprint"
done

# --- and prove what landed ----------------------------------------------------
check() {
  local path="$1" want="$2" have
  have="$(digest "${path}")"
  [[ "${have}" == "${want}" ]] \
    || { echo "install-cell: ${path} installed as ${have}, want ${want}" >&2; exit 4; }
}
check .build-worker/release/mlxfast-runtime-worker "$(read_json worker_sha256)"
check .build/release/mlxfast-swift "$(read_json cli_sha256)"
check .build-worker/release/mlx.metallib "$(read_json metallib_sha256)"

tree_fp="$(tools/build-mlx-metallib.sh --print-fingerprint)"
published_fp="$(awk '{print $2}' .build-worker/release/mlx.metallib.fingerprint)"
if [[ "${tree_fp}" != "${published_fp}" ]]; then
  echo "install-cell: metallib was built from ${published_fp} but the tree is now ${tree_fp}" >&2
  exit 5
fi

printf 'install-cell: %s installed\n' "$(basename "${cell_dir}")"
printf '  worker   __text    %s\n' "$(read_json worker_text_sha256)"
printf '  worker   __cstring %s\n' "$(read_json worker_cstring_sha256)"
printf '  metallib %s (fingerprint %s)\n' "$(read_json metallib_sha256)" "${tree_fp}"
