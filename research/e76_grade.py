"""E76 rung 3, graded against the E77-corrected occupancy law.

Feedback `e76-retract-the-occupancy-premise` retracts the earlier 208 KiB
register file, `eps = 0.111`, `kappa = 0.0600` and the graded tier table. The
surviving measured law from E77 is

    S_local(R)  = floor(384 KiB / (128 * R))
    S_ranked(R) = floor(496 KiB / (128 * R))
    Omega(S)    = (32 / S) ** gamma,  gamma = 0.01346 +/- 0.00065

Occupancy is a smooth, very weak function of R, not a staircase. This script
prices each arm's occupancy gain under that law and sets it against the measured
local cost, so the register route is judged on the corrected constants.

It also answers deliverable B: which NA=6 arms remove the 16-byte local spill
frame without raising the ranked register count.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e76_report  # noqa: E402  (needs the research directory on the path)

LOCAL_FILE_KIB = 384
RANKED_FILE_KIB = 496
BYTES_PER_REGISTER_PER_SIMDGROUP = 128
GAMMA = 0.01346
GAMMA_STDERR = 0.00065
SHIPPED_G16S_FRAME_BYTES = 16
ARTIFACTS = pathlib.Path("research/e76-artifacts")


def simdgroups(registers: int, file_kib: int) -> int:
    return (file_kib * 1024) // (BYTES_PER_REGISTER_PER_SIMDGROUP * registers)


def omega(simdgroups_per_core: int, gamma: float = GAMMA) -> float:
    return (32.0 / simdgroups_per_core) ** gamma


def check_against_e77() -> None:
    """The two register files must reproduce Alphonse's published S columns."""
    published = [(70, 43, 83, 47), (93, 33, 90, 44), (94, 32, 91, 43),
                 (95, 32, 98, 40), (96, 32, 111, 35)]
    print("=== register-file validation against E77 ===")
    ok = True
    for local_r, local_s, ranked_r, ranked_s in published:
        got_local = simdgroups(local_r, LOCAL_FILE_KIB)
        got_ranked = simdgroups(ranked_r, RANKED_FILE_KIB)
        good = got_local == local_s and got_ranked == ranked_s
        ok = ok and good
        print(f"{'OK' if good else 'MISMATCH':>9}  local R={local_r} S={got_local}"
              f" (E77 {local_s});  ranked R={ranked_r} S={got_ranked} (E77 {ranked_s})")
    print(f"{'all rows reproduced' if ok else 'DISAGREEMENT'}\n")


def main() -> None:
    check_against_e77()
    rows = json.loads((ARTIFACTS / "rung1-table.json").read_text())
    timed = e76_report.timings()
    checked = e76_report.parity()

    def per_round(na: int, arm: str) -> float | None:
        shapes = timed.get((na, arm), {})
        if not shapes:
            return None
        return sum(e76_report.DISPATCHES_PER_ROUND[s] * v
                   for s, v in shapes.items())

    print("=== the register route under the corrected law ===")
    print("Occupancy gain is priced against the shipped cell at the same width.")
    for na in (5, 6):
        shipped = next(r for r in rows if r["na"] == na and r["arm"] == "plain")
        base_s = simdgroups(shipped["g17s_registers"], RANKED_FILE_KIB)
        base_omega = omega(base_s)
        base_sec = per_round(na, "plain")
        print(f"\nNA={na}: shipped {shipped['g17s_registers']} regs,"
              f" ranked S={base_s}")
        print(f"{'arm':<14}{'g17s':>6}{'S':>5}{'occ gain':>10}"
              f"{'measured cost':>15}{'net':>10}  verdict")
        sel = sorted((r for r in rows if r["na"] == na),
                     key=lambda r: r["g17s_registers"])
        for r in sel:
            if r["g17s_spill_bytes"] or r["arm"] == "plain":
                continue
            sec = per_round(na, r["arm"])
            if sec is None or base_sec is None:
                continue
            s = simdgroups(r["g17s_registers"], RANKED_FILE_KIB)
            gain = 100.0 * (omega(s) / base_omega - 1.0)
            cost = 100.0 * (sec / base_sec - 1.0)
            net = 100.0 * ((omega(s) / base_omega) * (1 + cost / 100) - 1.0)
            print(f"{r['arm']:<14}{r['g17s_registers']:>6}{s:>5}"
                  f"{gain:>+9.3f} %{cost:>+14.2f} %{net:>+9.2f} %"
                  f"  {'wins' if net < 0 else 'loses'}")

    best = min(r["g17s_registers"] for r in rows if not r["g17s_spill_bytes"])
    worst = max(r["g17s_registers"] for r in rows if not r["g17s_spill_bytes"])
    span = 100.0 * (omega(simdgroups(best, RANKED_FILE_KIB))
                    / omega(simdgroups(worst, RANKED_FILE_KIB)) - 1.0)
    print(f"\nWhole-grid occupancy span, {worst} -> {best} ranked registers:"
          f" {span:+.3f} %")
    for label, g in (("gamma -1 sd", GAMMA - GAMMA_STDERR),
                     ("gamma +1 sd", GAMMA + GAMMA_STDERR)):
        s_hi = omega(simdgroups(best, RANKED_FILE_KIB), g)
        s_lo = omega(simdgroups(worst, RANKED_FILE_KIB), g)
        print(f"  {label}: {100.0 * (s_hi / s_lo - 1.0):+.3f} %")

    print("\n=== deliverable B: remove the 16-byte local frame at NA=6 ===")
    print("Requirement: g16s frame gone, ranked g17s count not raised above the")
    print("shipped 111, output bit-identical on all seven priced shapes.")
    print(f"{'arm':<14}{'rps':>4}{'g16s':>6}{'frame':>7}{'g17s':>6}"
          f"{'vs shipped':>12}{'parity':>18}")
    shipped6 = next(r for r in rows if r["na"] == 6 and r["arm"] == "plain")
    cands = [r for r in rows if r["na"] == 6
             and r["g16s_spill_bytes"] == 0
             and r["g17s_spill_bytes"] == 0
             and r["g17s_registers"] <= shipped6["g17s_registers"]]
    for r in sorted(cands, key=lambda r: (-r["rows_per_simd"], r["g17s_registers"])):
        check = checked.get((6, r["arm"]))
        par = ("NOT CHECKED" if check is None
               else f"clean {len(check['shapes'])} shapes" if check["differing"] == 0
               else f"DIFFERS {check['differing']}")
        print(f"{r['arm']:<14}{r['rows_per_simd']:>4}{r['g16s_registers']:>6}"
              f"{r['g16s_spill_bytes']:>7}{r['g17s_registers']:>6}"
              f"{r['g17s_registers'] - shipped6['g17s_registers']:>+11}"
              f"{par:>18}")


if __name__ == "__main__":
    main()
