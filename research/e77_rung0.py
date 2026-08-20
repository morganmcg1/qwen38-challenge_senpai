#!/usr/bin/env python3
"""E77 rung 0: respecify the cost model with an occupancy factor and price the flip.

E73 proved that `argmin` over IPG is invariant to any pure rescaling of the
per-byte rate, so a bandwidth-headroom difference between hosts cannot move a
partition. The ranked receipt says a partition moved. This rung writes down the
one extra term that can move it, shows algebraically that it can, and converts
"can" into the exact numbers rung 1 has to beat.

No GPU. Reads the E73 fit and the E77 register census, writes the rung-0
pre-registration record.

  python3 research/e77_rung0.py --out research/e77-artifacts/rung0.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from e73_pairs import CROWN, SHIPPED, pairs  # noqa: E402

# Bytes of register file consumed by one simdgroup per allocated register:
# 32 threads x 4 bytes.
BYTES_PER_REGISTER_PER_SIMDGROUP = 128

# Register-file bytes per core. The local value is the hypothesis rung 1 tests
# by locating the steps; the ranked value is an EXTRAPOLATION derived from the
# ranked allocator ceiling of 124 under the same "stop at 32 resident
# simdgroups" rule the local ceiling of 96 obeys at 384 KiB.
REGISTER_FILE_BYTES = {"applegpu_g16s": 384 * 1024, "applegpu_g17s": 124 * 128 * 32}

CORES = {"applegpu_g16s": 20, "applegpu_g17s": 40}


def resident(arch: str, registers: int) -> int:
    """Resident simdgroups per core at a given per-thread register count."""
    return REGISTER_FILE_BYTES[arch] // (BYTES_PER_REGISTER_PER_SIMDGROUP * registers)


def steps(arch: str, low: int, high: int) -> list[int]:
    """Register counts at which resident simdgroups per core drops."""
    return [r for r in range(low + 1, high + 1)
            if resident(arch, r) < resident(arch, r - 1)]


def register_by_ipg(census: dict, arch: str) -> dict[int, dict]:
    """Per-IPG register count, and proof that it is a function of IPG alone."""
    found: dict[int, dict] = {}
    for m, ipg in pairs():
        record = census["cells"][arch][f"e77_cell_m{m}_ipg{ipg}"]
        seen = found.setdefault(ipg, dict(registers=record["registers"],
                                          frame_bytes=record["spill_bytes"],
                                          cells=[], collinear=True))
        seen["cells"].append(f"m{m}_ipg{ipg}")
        if (record["registers"] != seen["registers"]
                or record["spill_bytes"] != seen["frame_bytes"]):
            seen["collinear"] = False
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e73", type=pathlib.Path,
                        default=pathlib.Path("research/e73-artifacts/rung2.json"))
    parser.add_argument("--regs", type=pathlib.Path,
                        default=pathlib.Path("research/e77-artifacts/rung0-regs.json"))
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("research/e77-artifacts/rung0.json"))
    args = parser.parse_args()

    e73 = json.loads(args.e73.read_text())
    census = json.loads(args.regs.read_text())
    local, ranked = "applegpu_g16s", "applegpu_g17s"

    by_ipg = {arch: register_by_ipg(census, arch) for arch in (local, ranked)}
    collinear = {arch: all(v["collinear"] for v in by_ipg[arch].values())
                 for arch in (local, ranked)}

    # The threshold each shipped-versus-crown pair has to clear. Replacing the
    # E73 level term `q(IPG)` with `c(IPG) * Omega_h(R_h(IPG))` multiplies every
    # E73 ranked cost by the transfer gain G(IPG) = Omega_R(R_R) / Omega_L(R_L),
    # so the crown's cell wins exactly when G(ours) / G(crown) exceeds the E73
    # ranked cost ratio. Everything else in the model cancels.
    thresholds = {}
    for m in sorted(SHIPPED):
        ours, crown = SHIPPED[m], CROWN[m]
        if ours == crown:
            continue
        ranking = {int(i): c for i, c in e73["ranked_prediction"][str(m)]["ranking"]}
        thresholds[m] = dict(
            ours=ours, crown=crown,
            e73_ranked_cost_ours=ranking[ours],
            e73_ranked_cost_crown=ranking[crown],
            required_gain_ratio=ranking[crown] / ranking[ours],
            registers_local_ours=by_ipg[local][ours]["registers"],
            registers_local_crown=by_ipg[local][crown]["registers"],
            registers_ranked_ours=by_ipg[ranked][ours]["registers"],
            registers_ranked_crown=by_ipg[ranked][crown]["registers"],
        )

    # The pre-registered prior: the penalty is inversely proportional to
    # resident simdgroups per core, Omega(S) = S_ref / S. Under it the transfer
    # gain is fully determined by the two register files, with no free
    # parameter, so it predicts each reorder before rung 1 runs.
    for m, row in thresholds.items():
        gain_ours = (resident(local, row["registers_local_ours"])
                     / resident(ranked, row["registers_ranked_ours"]))
        gain_crown = (resident(local, row["registers_local_crown"])
                      / resident(ranked, row["registers_ranked_crown"]))
        row["prior_gain_ratio"] = gain_ours / gain_crown
        row["prior_reorders"] = row["prior_gain_ratio"] > row["required_gain_ratio"]

    payload = dict(
        experiment="e77", rung=0, harness="local",
        register_file_bytes=REGISTER_FILE_BYTES,
        register_file_ranked_is_extrapolated=True,
        cores=CORES,
        registers_by_ipg={arch: by_ipg[arch] for arch in (local, ranked)},
        registers_are_a_function_of_ipg_alone=collinear,
        resident_simdgroups={
            arch: {ipg: resident(arch, v["registers"])
                   for ipg, v in by_ipg[arch].items()} for arch in (local, ranked)},
        predicted_steps_local_70_to_96=steps(local, 70, 96),
        sweep_registers_local=sorted({
            census["sweep"][local][f"e77_{a}"]["registers"]
            for a, s in census["arms"].items() if s["kind"] == "p"
            and census["sweep"][local][f"e77_{a}"]["spill_bytes"] == 0}),
        thresholds=thresholds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))

    print("registers are a function of IPG alone:", collinear)
    for arch in (local, ranked):
        print(f"{arch}  file={REGISTER_FILE_BYTES[arch]}B cores={CORES[arch]}")
        for ipg in sorted(by_ipg[arch]):
            v = by_ipg[arch][ipg]
            print(f"  IPG={ipg} registers={v['registers']:<4} "
                  f"frame={v['frame_bytes']:<4} "
                  f"resident_simdgroups_per_core={resident(arch, v['registers'])} "
                  f"cells={len(v['cells'])} collinear={v['collinear']}")
    print(f"predicted local steps in [70, 96]: {steps(local, 70, 96)}")
    print(f"{'M':>2} {'ours':>4} {'crown':>5} {'required':>9} {'prior':>7} reorders")
    for m, row in thresholds.items():
        print(f"{m:2d} {row['ours']:4d} {row['crown']:5d} "
              f"{row['required_gain_ratio']:9.4f} {row['prior_gain_ratio']:7.4f} "
              f"{row['prior_reorders']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
