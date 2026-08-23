#!/usr/bin/env bash
# E153 R1 -- the submission gate chain for leaf16 shipped as the compiled
# default. The merged SDPA kernel is NOT in this tree; it moved to E156 after
# the gated ABBA measured it bit-exact but slower.
#
# The contrast itself is NOT re-measured here. E149 Arm A already has twelve
# gated legs on exactly `derivedClusterRowsPerLeaf` 8 -> 16, and the advisor
# directed that it not be re-timed. What this chain proves is that the shipped
# surface carries the change, that the change still has a failing polarity, and
# that a full 512-token leg is exact on the shipped binary.
#
#   phase 0  worker rebuild and Rule 101 selector assertion
#   phase 1  swift test at the documented floor
#   phase 2  twin audit
#   phase 3  Rule 101 leaf-override positive control, on the shipped default
#   phase 4  512-token exactness legs, real 40 C gate, medpair prompts
#
#   env: E153R1_PHASES=0,1,2,3,4
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

phases="${E153R1_PHASES:-0,1,2,3,4}"
out=research/out
mkdir -p "${out}"
has_phase() { [[ ",${phases}," == *",$1,"* ]]; }

dirty="$(git status --porcelain -- Sources Vendor Package.swift \
  Package.resolved mtp-head.manifest.json)"
if [[ -n "${dirty}" ]]; then
  echo "e153_r1: scored surface is dirty; refusing" >&2
  echo "${dirty}" >&2
  exit 1
fi
session_commit="$(git rev-parse HEAD)"
echo "e153_r1: session_commit=${session_commit}"

if has_phase 0; then
  echo "=== e153_r1 phase 0: worker build and selector assertion ==="
  # The merged-kernel selectors must now be ABSENT. Asserting the leaf16
  # selectors present and the E156 selectors gone is the two-sided check that
  # this really is the leaf16-only surface.
  senpai/rebuild-and-assert-worker.sh \
    --require MLX_E141_ROWS_PER_LEAF \
    --require-symbol activeClusterRowsPerLeaf \
    --require-symbol qwen35E141RowsPerLeafOverride \
    || { echo "e153_r1: leaf16 selectors missing" >&2; exit 3; }
  if strings .build-worker/release/mlxfast-runtime-worker 2>/dev/null \
      | grep -q qwen35_merged_sdpa_vector; then
    echo "e153_r1: merged SDPA kernel is still in the worker; not shippable" >&2
    exit 3
  fi
  echo "ok   absent 'qwen35_merged_sdpa_vector' (moved to E156)"
fi

worker_sha="$(shasum -a 256 .build-worker/release/mlxfast-runtime-worker \
  | awk '{print $1}')"
echo "e153_r1: worker_sha256=${worker_sha}"

if has_phase 1; then
  echo "=== e153_r1 phase 1: swift test ==="
  swift test --force-resolved-versions 2>&1 | tee "${out}/e153-r1-swift-test.log"
  echo "swift test exit ${PIPESTATUS[0]}"
  grep -E "Test run with .* passed|Test run with .* failed|[0-9]+ issue" \
    "${out}/e153-r1-swift-test.log" | tail -5
fi

if has_phase 2; then
  echo "=== e153_r1 phase 2: twin audit ==="
  python3 research/twin_audit.py 2>&1 | tee "${out}/e153-r1-twin-audit.log"
  echo "twin_audit exit ${PIPESTATUS[0]}"
fi

if has_phase 3; then
  echo "=== e153_r1 phase 3: leaf override positive control (Rule 101) ==="
  research/e149_leaf_witness.sh 2>&1 | tail -20
  cp research/e149-leaf-witness.json "${out}/e153-r1-leaf-witness.json" \
    2>/dev/null || true
fi

if has_phase 4; then
  runs_parent=".mlxfast-private/e128/runs-e153r1"
  for id in beagle_a essays_montaigne; do
    echo "=== e153_r1 phase 4: cool gate before 512-token leg ${id} ==="
    if ./benchmark.sh --local-cool-gate-only; then
      gate_status=passed
    else
      gate_status=stalled_above_40C
    fi
    echo "=== e153_r1 phase 4: 512-token exactness leg ${id}" \
         "gate=${gate_status} ==="
    env E128_FORCE=1 E128_NO_TRACE=1 E128_TOKENS=512 E128_DEPTH=8 \
        E128_RUNS_DIR="runs-e153r1/${id}" \
      research/e128_session.sh "${id}"
    legout="${runs_parent}/${id}/${id}"
    {
      echo "e153_leg_role=shipped-surface-exactness"
      echo "e153_cool_gate_status=${gate_status}"
      echo "e153_session_commit=${session_commit}"
      echo "e153_session_worker_sha256=${worker_sha}"
      echo "e153_rows_per_leaf=16-compiled-default"
    } >> "${legout}/meta.txt"
    grep -E "all_tokens_matched|residual_divergence_count|gpu_temp_entry_c|gpu_temp_exit_c|effective_mean_draft_len|round_count" \
      "${legout}/meta.txt" 2>/dev/null | sed 's/^/  /'
  done
fi

echo "e153_r1: gate chain complete"
