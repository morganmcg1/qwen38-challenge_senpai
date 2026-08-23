#!/usr/bin/env bash
# E151 R1 -- attribute the 512-token row-digest change to a cause.
#
#   usage: bash /tmp/<copy of this script>          (run the COPY, not this file)
#
# WHY A COPY. This script switches the worktree between two commits. The file
# does not exist at the base commit, so bash would read a truncated script if it
# ran from the checkout. Copy it outside the repository first.
#
# WHAT HAPPENED. The R1 candidate leg produced 1024 `mtp-row:` lines with digest
# d070b397..., against a campaign pin of 1025 lines and 719d82b8.... The first
# 1021 rows are byte-identical to the historical trace. The trace loses one
# trailing row at pos=1025 and moves the top-two evidence at pos=1022 to 1024.
# The wrapper still reports `all_tokens_matched: true`, and `--local-submit`
# still closes its row ledger at 512/512 and 561/561.
#
# THE COMPETING EXPLANATIONS.
#   H1  R1 changed local output. This should be IMPOSSIBLE: `is_nax_available()`
#       is false on this g16s host, so the retiled kernel never executes here.
#   H2  The pin is stale with respect to the current campaign base. Many
#       scheduler commits landed between the pinned base and de8ce44c, and the
#       observed round count moved from 78 to 82, which changes the trailing
#       lookahead row.
#
# THE TEST. Run the identical leg at the campaign base and at the candidate, on
# one host in one session with the same head and fixture. H1 predicts the two
# digests differ. H2 predicts they are identical and both differ from the pin.
set -uo pipefail

ROOT="${E151_ROOT:?set E151_ROOT to the checkout}"
cd "${ROOT}"

BASE=de8ce44c7bc133c3c6c079957240782664afd287
BRANCH=qwen-alphonse/e151-ranked-prefill-channel
PIN=719d82b87c79d26a28ba326676bf144606c947cbbd337ed49347b0c5c61ec16e
out=research/e151-artifacts
mkdir -p "${out}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "e151_arm_attribution: worktree is dirty; refusing to switch commits" >&2
  exit 1
fi

restore() {
  git switch "${BRANCH}" >/dev/null 2>&1 \
    || git switch --force "${BRANCH}" >/dev/null 2>&1
}
trap restore EXIT

leg() {
  local tag="$1"
  echo "=== ${tag}: rebuilding worker and metallib at $(git rev-parse --short HEAD) ==="
  senpai/rebuild-and-assert-worker.sh > "${out}/${tag}-build.log" 2>&1 \
    || { echo "${tag}: build failed"; tail -20 "${out}/${tag}-build.log"; return 1; }
  echo "=== ${tag}: 512-token traced leg ==="
  research/e79_trace_leg.sh "${tag}" 512
}

echo "########## ARM-OFF BASE ##########"
git switch --detach "${BASE}" >/dev/null 2>&1 || exit 1
leg e151x512basectl || echo "e151_arm_attribution: base leg returned nonzero"

echo
echo "########## ARM-ON CANDIDATE ##########"
restore
leg e151x512cand2 || echo "e151_arm_attribution: candidate repeat leg returned nonzero"

echo
echo "########## DIGESTS ##########"
python3 - "$PIN" <<'PY'
import hashlib, pathlib, sys, json
pin = sys.argv[1]
tags = ["e151x512cand", "e151x512basectl", "e151x512cand2"]
res = {}
for tag in tags:
    p = pathlib.Path("research/out") / tag / "trace.txt"
    if not p.is_file():
        res[tag] = None
        continue
    rows = [l.strip() for l in p.read_text().splitlines()
            if l.startswith("mtp-row:")]
    meta = (pathlib.Path("research/out") / tag / "meta.txt")
    fields = {}
    if meta.is_file():
        for line in meta.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                fields[k] = v
    res[tag] = {
        "rows": len(rows),
        "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest(),
        "base_sha": fields.get("base_sha"),
        "worker_sha256": fields.get("worker_sha256"),
        "metallib_source_fingerprint": fields.get("metallib_source_fingerprint"),
        "trace_rounds": fields.get("trace_rounds"),
        "dirty_candidate_paths": fields.get("dirty_candidate_paths"),
    }
    print(f"{tag:20} rows={res[tag]['rows']:5} sha256={res[tag]['sha256'][:16]} "
          f"rounds={res[tag]['trace_rounds']} base={str(res[tag]['base_sha'])[:8]}")

cand, base = res.get("e151x512cand"), res.get("e151x512basectl")
rep = res.get("e151x512cand2")
verdict = {
    "experiment": "E151",
    "rung": "R1",
    "harness": "local",
    "pinned_sha256": pin,
    "legs": res,
    "e151_arm_is_local_output_neutral":
        bool(cand and base and cand["sha256"] == base["sha256"]),
    "e151_leg_is_deterministic":
        bool(cand and rep and cand["sha256"] == rep["sha256"]),
    "e151_base_also_misses_pin":
        bool(base and base["sha256"] != pin),
}
print()
for k, v in verdict.items():
    if k.startswith("e151_"):
        print(f"{k:40} {v}")
pathlib.Path("research/e151-arm-attribution.json").write_text(
    json.dumps(verdict, indent=1, sort_keys=True) + "\n")
print("wrote research/e151-arm-attribution.json")
PY
