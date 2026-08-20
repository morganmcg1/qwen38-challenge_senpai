#!/usr/bin/env bash
# Build every cell of the E75 rung D 2x2 before any leg is timed.
#
#   research/e75_rungD_prebuild.sh [CELL ...]
#
# Cells are `TABLE-PRICE` over TABLE in {ours, crown} and PRICE in
# {ship, pbfit}. The default order builds `ours` first and switches the kernel
# table exactly once, so Cmlx and the metallib are recompiled once rather than
# four times.
#
# Each cell leaves behind its worker, its CLI, its metallib and a manifest of
# every digest an installed leg must reproduce. The last step asserts the
# witness matrix: `__TEXT,__text` must take exactly two values across the four
# cells (the depth price), `__TEXT,__cstring` exactly two (the kernel table),
# and the four pairs must be distinct. That is the reachability evidence for
# the 2x2 and it is checked here, before a single leg waits at the 40C gate.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo_root="${PWD}"

readonly SCORED_FILE="Sources/MLXFastModel/Qwen36MTPBlockSession.swift"
readonly TABLE_FILES=(
  "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h"
  "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp"
)
readonly DEFAULT_CELLS=(ours-ship ours-pbfit crown-ship crown-pbfit)

store="${E75_CELL_STORE:-${repo_root}/.mlxfast-private/e75-rungD/cells}"
rung1="${E68_RUNG1_CURVE:-research/e68-artifacts/e68-rung1.json}"
verify_forward_key="${E68_VERIFY_FORWARD_KEY:-0.060300}"

cells=("$@")
((${#cells[@]})) || cells=("${DEFAULT_CELLS[@]}")

[[ -z "$(git status --porcelain)" ]] \
  || { echo "prebuild: worktree is dirty; refusing to build over uncommitted work" >&2; exit 1; }
pre_sha="$(git rev-parse HEAD)"
restore() { git checkout -q "${pre_sha}" -- "${SCORED_FILE}" "${TABLE_FILES[@]}" 2>/dev/null || true; }
trap restore EXIT

mkdir -p "${store}" .build/clang-module-cache .build-worker/clang-module-cache

for cell in "${cells[@]}"; do
  table="${cell%%-*}"
  price="${cell##*-}"
  out="${store}/${cell}"
  rm -rf "${out}"; mkdir -p "${out}"
  echo "prebuild: === $(date -u +%Y-%m-%dT%H:%M:%SZ) ${cell} (table=${table} price=${price}) ===" >&2

  restore
  python3 research/e75_arms.py "${table}" --out "${out}/table-arm.json" \
    > "${out}/table-arm.log" 2>&1
  price_args=("${price}" --out "${out}/price-arm.json")
  if [[ "${price}" == "pbfit" ]]; then
    price_args+=(--raw-from "${rung1}" --verify-forward-key "${verify_forward_key}")
  fi
  python3 research/e68_swift_arm.py "${price_args[@]}" > "${out}/price-arm.log" 2>&1

  selected="$(grep -c "let depthPriceArm: DepthPriceArm = .${price}\$" "${SCORED_FILE}")"
  ((selected == 1)) || { echo "prebuild: ${cell} selector did not take" >&2; exit 2; }

  tools/build-mlx-metallib.sh --all-build-roots > "${out}/build-metallib.log" 2>&1
  CLANG_MODULE_CACHE_PATH="${repo_root}/.build/clang-module-cache" \
    swift build -c release --force-resolved-versions --product mlxfast-swift \
    > "${out}/build-cli.log" 2>&1
  CLANG_MODULE_CACHE_PATH="${repo_root}/.build-worker/clang-module-cache" \
    swift build -c release --force-resolved-versions \
    --scratch-path .build-worker --product mlxfast-runtime-worker \
    > "${out}/build-worker.log" 2>&1

  cp -f .build-worker/release/mlxfast-runtime-worker "${out}/mlxfast-runtime-worker"
  cp -f .build/release/mlxfast-swift "${out}/mlxfast-swift"
  cp -f .build-worker/release/mlx.metallib "${out}/mlx.metallib"
  cp -f .build-worker/release/mlx.metallib.fingerprint "${out}/mlx.metallib.fingerprint"

  python3 - "${out}" "${cell}" "${table}" "${price}" "${SCORED_FILE}" "${TABLE_FILES[@]}" <<'PY'
import hashlib, json, pathlib, subprocess, sys

out, cell, table, price, *sources = sys.argv[1:]
out = pathlib.Path(out)


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def sections(binary):
    raw = subprocess.run(
        ["python3", "research/e59_worker_digest.py", binary, "--json"],
        check=True, capture_output=True, text=True).stdout
    return json.loads(raw)["sections"]


worker = out / "mlxfast-runtime-worker"
sect = sections(str(worker))
payload = {
    "cell": cell,
    "kernel_table": table,
    "depth_price": price,
    "source_sha256": {path: sha256(path) for path in sources},
    "worker_sha256": sha256(worker),
    "worker_text_sha256": sect["__TEXT,__text"]["sha256"],
    "worker_cstring_sha256": sect["__TEXT,__cstring"]["sha256"],
    "cli_sha256": sha256(out / "mlxfast-swift"),
    "metallib_sha256": sha256(out / "mlx.metallib"),
    "metallib_fingerprint":
        (out / "mlx.metallib.fingerprint").read_text().split()[1],
    "table_arm": json.loads((out / "table-arm.json").read_text()),
    "price_arm": json.loads((out / "price-arm.json").read_text()),
}
(out / "cell.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
print("prebuild: %s worker __text %s __cstring %s"
      % (cell, payload["worker_text_sha256"][:16],
         payload["worker_cstring_sha256"][:16]))
PY
done

restore

python3 - "${store}" "${DEFAULT_CELLS[@]}" <<'PY'
import json, pathlib, sys

store = pathlib.Path(sys.argv[1])
cells = sys.argv[2:]
have = {c: json.loads((store / c / "cell.json").read_text())
        for c in cells if (store / c / "cell.json").exists()}
if len(have) < len(cells):
    print("prebuild: %d of %d cells present; witness matrix not checked yet"
          % (len(have), len(cells)))
    raise SystemExit(0)

text = {c: have[c]["worker_text_sha256"] for c in cells}
cstr = {c: have[c]["worker_cstring_sha256"] for c in cells}
pairs = {c: (text[c], cstr[c]) for c in cells}

print("prebuild: witness matrix")
for c in cells:
    print("  %-12s __text %s  __cstring %s"
          % (c, text[c][:16], cstr[c][:16]))

problems = []
if len(set(text.values())) != 2:
    problems.append("__text takes %d values, want 2 (the depth price)"
                    % len(set(text.values())))
if len(set(cstr.values())) != 2:
    problems.append("__cstring takes %d values, want 2 (the kernel table)"
                    % len(set(cstr.values())))
if len(set(pairs.values())) != 4:
    problems.append("the four cells give %d distinct witness pairs, want 4"
                    % len(set(pairs.values())))
# The depth price must move __text within a table, and the kernel table must
# move __cstring within a price. Either failing means the factor did not reach
# the binary and the corresponding main effect would be unmeasurable.
for table in ("ours", "crown"):
    if text["%s-ship" % table] == text["%s-pbfit" % table]:
        problems.append("depth price did not move __text on the %s table" % table)
for price in ("ship", "pbfit"):
    if cstr["ours-%s" % price] == cstr["crown-%s" % price]:
        problems.append("kernel table did not move __cstring at price %s" % price)

if problems:
    for p in problems:
        print("prebuild: FAIL %s" % p)
    raise SystemExit(6)
print("prebuild: witness matrix OK, 4 distinct cells on 2 independent axes")
PY
