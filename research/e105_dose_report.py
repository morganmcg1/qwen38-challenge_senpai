#!/usr/bin/env python3
"""Reduce the E105 in-situ dispatch-dose ladder to F and to the fusion ceiling.

    usage: research/e105_dose_report.py TAG [TAG ...] [--json OUT]

Each TAG is `research/out/TAG/` holding `score.json` and `meta.txt` from
`research/e105_dose_leg.sh`.

WHAT THIS MEASURES. E105 rung 0 established that the GDN prework, q/k norm +
RoPE and KV cache write families are already one dispatch per layer, 96
dispatches per round in total. A fusion therefore cannot win more than

    dT  =  N x F        N = dispatches removed,  F = marginal cost of one

so the whole experiment reduces to F. The dose ladder inserts a known number
of extra dependent dispatches per decoder layer and reads the slope, which is
F in the in-situ frame. No isolation discount is applied or needed, because
nothing here is measured in the isolated census frame.

TWO INDEPENDENT SLOPES. The `--local-iterate` leg times a serial pass and an
MTP pass with the same build. The serial pass runs exactly one target forward
per token, so its slope is the cleanest estimate of F. The MTP pass runs one
target forward per round and needs the round's token yield to convert.

FRAME. Advisor feedback e105-f1 retired the Finding 22 LATENCY multiplier of
2.40x. Launch-overhead-bound dispatch work transfers at about 0.95x, so a
local percent of the local round is also the ranked percent. This reducer
reports local percent only. It applies no multiplier and no isolation
discount. Two denominators are reported because the campaign uses both:

  census GPU-busy round   GPU-busy time in one w5 round from the E58/E80
                          census. This is the frame of the advisor's family
                          table, where the three families total 782.34 us.
  leg wall round          `mtp_spt x (1 + mean_draft)` from the leg itself.
                          This is the frame of the matched-ABBA promotion bar.

Promotion bar, e105-f1: a bit-exact arm that moves the matched-ABBA local
round by at least 0.20 %. Everything here is `harness=local` directional
evidence and no leg is a score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

DECODER_LAYERS = 64
# Measured on this base, tag e105r0-d4-ops0: GPU-busy time in one w5 round.
CENSUS_ROUND_US = 102013.3
# The advisor's family table is anchored on the older E96 census round.
ADVISOR_ROUND_US = 127533.0
BAR_LOCAL_PCT = 0.20  # e105-f1 promotion bar, matched-ABBA local round
SF_FLOOR_PCT = 0.160

# Rung 0 census, tag e105r0-d4-ops0, w5. Dispatches per round that a fusion
# could remove. The q/k kernel survives as the host of any FA-side fusion, so
# 80 is the real maximum and 96 is the physically impossible upper bound.
N_REAL = 80
N_ABSOLUTE = 96


def load(tag: str) -> dict:
    d = pathlib.Path("research/out") / tag
    score = json.loads((d / "score.json").read_text())["metrics"]
    meta = {}
    for line in (d / "meta.txt").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    dose = int(meta["dose_per_layer"])
    draft = score["effective_mean_draft_len"]
    return {
        "tag": tag,
        "dose": dose,
        "shape": meta["dose_shape"],
        "tokens": int(meta["decode_tokens"]),
        "added_dispatches_per_forward": dose * DECODER_LAYERS,
        "mtp_spt": score["mtp_seconds_per_token"],
        "serial_spt": score["serial_seconds_per_token"],
        "speedup": score["mtp_decode_speedup"],
        "mean_draft": draft,
        "tokens_per_round": 1.0 + draft,
        "mtp_round_us": score["mtp_seconds_per_token"] * (1.0 + draft) * 1e6,
        "serial_round_us": score["serial_seconds_per_token"] * 1e6,
        "matched": score["all_tokens_matched"],
        "entry_c": meta.get("gpu_temp_entry_c", ""),
        "exit_c": meta.get("gpu_temp_exit_c", ""),
        "worker_sha256": meta.get("worker_sha256_pre", ""),
        "dose_probe_in_worker": meta.get("worker_dose_probe", ""),
        "git_head": meta.get("git_head", ""),
        "cool_gate_passed_real_gate": meta.get("cool_gate_passed_real_gate", ""),
        "gate_qualified_for_timing": meta.get("gate_qualified_for_timing", ""),
    }


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def slope(legs: list[dict], key: str) -> dict | None:
    """Least-squares slope of round us against added dispatches per forward."""
    by_dose: dict[int, list[float]] = {}
    for leg in legs:
        by_dose.setdefault(leg["added_dispatches_per_forward"], []).append(leg[key])
    if len(by_dose) < 2:
        return None
    xs = sorted(by_dose)
    ys = [mean(by_dose[x]) for x in xs]
    xbar, ybar = mean([float(x) for x in xs]), mean(ys)
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    den = sum((x - xbar) ** 2 for x in xs)
    return {
        "points": {str(x): round(y, 1) for x, y in zip(xs, ys)},
        "replicates": {str(x): len(by_dose[x]) for x in xs},
        "F_us_per_dispatch": num / den,
        "pairwise": {
            f"{xs[i]}->{xs[i + 1]}": (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
            for i in range(len(xs) - 1)
        },
    }


DENOM_LABELS = (
    "census_gpu_busy_round",
    "advisor_e96_round",
    "leg_wall_round",
    "decode_only_round",
)


def decode_only_round(legs: list[dict]) -> dict | None:
    """Strip the fixed seed and warmup share out of the reported round.

    `spt(n) = P / n + D`. Two dose-0 token counts give P and D exactly, and
    the honest local round is `D x (1 + mean_draft)`.
    """
    zero = [leg for leg in legs if leg["dose"] == 0]
    by_n: dict[int, list[dict]] = {}
    for leg in zero:
        by_n.setdefault(leg["tokens"], []).append(leg)
    if len(by_n) < 2:
        return None
    ns = sorted(by_n)
    n_lo, n_hi = ns[0], ns[-1]
    out: dict[str, object] = {"tokens": [n_lo, n_hi]}
    for key, label in (("mtp_spt", "mtp"), ("serial_spt", "serial")):
        s_lo = mean([leg[key] for leg in by_n[n_lo]])
        s_hi = mean([leg[key] for leg in by_n[n_hi]])
        # s = P/n + D  =>  D = (n_hi*s_hi - n_lo*s_lo) / (n_hi - n_lo)
        d = (n_hi * s_hi - n_lo * s_lo) / (n_hi - n_lo)
        p = (s_lo - d) * n_lo
        out[label] = {
            "spt_at_n_lo": s_lo,
            "spt_at_n_hi": s_hi,
            "fixed_seed_and_warmup_s": p,
            "marginal_spt_s": d,
            "fixed_share_of_reported_at_n_hi": 1.0 - d / s_hi,
        }
    draft = mean([leg["mean_draft"] for leg in by_n[n_hi]])
    out["mean_draft_at_n_hi"] = draft
    out["decode_only_round_us"] = out["mtp"]["marginal_spt_s"] * (1 + draft) * 1e6
    return out


def price(f_us: float, wall_round_us: float, decode_round_us: float) -> dict:
    denominators = dict(
        zip(
            DENOM_LABELS,
            (CENSUS_ROUND_US, ADVISOR_ROUND_US, wall_round_us, decode_round_us),
        )
    )
    out: dict = {"F_us_per_dispatch": f_us, "denominators_us": denominators}
    for name, n in (("N80_real_max", N_REAL), ("N96_absolute", N_ABSOLUTE)):
        dt = n * f_us
        entry: dict = {"dispatches_removed": n, "ceiling_us_per_round": dt}
        for dname, dval in denominators.items():
            pct = 100.0 * dt / dval
            entry[dname] = {
                "local_pct": pct,
                "clears_0p20_bar": pct >= BAR_LOCAL_PCT,
                "multiple_of_bar": pct / BAR_LOCAL_PCT,
            }
        out[name] = entry
    out["required_F_us"] = {
        f"{dname}_at_N{n}": BAR_LOCAL_PCT * dval / 100.0 / n
        for dname, dval in denominators.items()
        for n in (N_REAL, N_ABSOLUTE)
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json")
    args = ap.parse_args()

    legs = [load(t) for t in args.tags]

    print("=== E105 in-situ dispatch-dose ladder ===")
    hdr = (f'{"tag":<34}{"shape":>9}{"dose":>6}{"n":>5}{"+disp":>7}'
           f'{"mtp us/rnd":>12}{"serial us/rnd":>15}{"speedup":>9}{"draft":>7}'
           f'{"match":>7}{"entryC":>8}{"exitC":>7}')
    print(hdr)
    print("-" * len(hdr))
    for leg in legs:
        print(f'{leg["tag"]:<34}{leg["shape"]:>9}{leg["dose"]:>6}'
              f'{leg["tokens"]:>5}'
              f'{leg["added_dispatches_per_forward"]:>7}{leg["mtp_round_us"]:>12.1f}'
              f'{leg["serial_round_us"]:>15.1f}{leg["speedup"]:>9.4f}'
              f'{leg["mean_draft"]:>7.3f}{str(leg["matched"]):>7}'
              f'{float(leg["entry_c"] or "nan"):>8.2f}'
              f'{float(leg["exit_c"] or "nan"):>7.2f}')

    voided = [leg["tag"] for leg in legs if not leg["matched"]]
    if voided:
        print(f"\nVOID: token mismatch in {voided}. The dose adds exactly zero, "
              f"so a mismatch means the instrument is wrong, not the model.")

    temps = [float(leg["entry_c"]) for leg in legs if leg["entry_c"]]
    if temps:
        print(f"\nentry temperature spread {min(temps):.2f} to {max(temps):.2f} C "
              f"({max(temps) - min(temps):.2f} C)")
    print("cool_gate_passed_real_gate=false  gate_qualified_for_timing=false  "
          "official_or_ranked_score=false")

    report: dict[str, object] = {"legs": legs}

    dec = decode_only_round(legs)
    if dec is None:
        print("\nno second dose-0 token count: cannot separate the fixed seed "
              "and warmup share, so `decode_only_round` falls back to the "
              "reported wall round")
        decode_round_us = mean(
            [leg["mtp_round_us"] for leg in legs if leg["dose"] == 0]
        )
    else:
        report["decode_only"] = dec
        decode_round_us = dec["decode_only_round_us"]
        print(f'\n--- fixed seed and warmup share, dose 0, n={dec["tokens"]} ---')
        for label in ("mtp", "serial"):
            b = dec[label]
            print(f'  {label:<7} spt {b["spt_at_n_lo"] * 1e3:8.2f} ms at n='
                  f'{dec["tokens"][0]}, {b["spt_at_n_hi"] * 1e3:8.2f} ms at n='
                  f'{dec["tokens"][1]}  ->  fixed '
                  f'{b["fixed_seed_and_warmup_s"]:.3f} s, marginal '
                  f'{b["marginal_spt_s"] * 1e3:.2f} ms/token, fixed share of '
                  f'the reported number {100 * b["fixed_share_of_reported_at_n_hi"]:.1f} %')
        print(f'  decode-only local round      : {decode_round_us:,.1f} us')

    # The scored round must be timed at one token count, so the slope uses the
    # main token count only and the short probe legs stay out of it.
    main_n = max(leg["tokens"] for leg in legs)
    ladder = [leg for leg in legs if leg["tokens"] == main_n]

    for shape in sorted({leg["shape"] for leg in ladder if leg["dose"] > 0}):
        # dose 0 belongs to every shape: it is the shared zero-dose reference.
        sel = [leg for leg in ladder if leg["shape"] == shape or leg["dose"] == 0]
        block: dict[str, object] = {}
        for key, label in (("serial_round_us", "serial"), ("mtp_round_us", "mtp")):
            s = slope(sel, key)
            if s is None:
                continue
            wall = mean([leg["mtp_round_us"] for leg in sel if leg["dose"] == 0])
            block[label] = {
                **s,
                **price(s["F_us_per_dispatch"], wall, decode_round_us),
            }
            print(f"\n--- shape={shape}  pass={label} ---")
            print(f'  round us by added dispatches : {s["points"]}')
            print(f'  replicates                   : {s["replicates"]}')
            print(f'  pairwise us/dispatch         : '
                  f'{ {k: round(v, 3) for k, v in s["pairwise"].items()} }')
            f_us = s["F_us_per_dispatch"]
            print(f'  F (least squares)            : {f_us:.3f} us/dispatch')
            p = block[label]
            for name in ("N80_real_max", "N96_absolute"):
                c = p[name]
                print(f'  ceiling at N={c["dispatches_removed"]:<3}          : '
                      f'{c["ceiling_us_per_round"]:8.1f} us/round')
                for dname in DENOM_LABELS:
                    q = c[dname]
                    denom = p["denominators_us"][dname]
                    print(f'    vs {dname:<22}{denom:>10.0f} us : '
                          f'{q["local_pct"]:7.3f} % local   '
                          f'{q["multiple_of_bar"]:5.2f}x the 0.20 % bar   '
                          f'{"CLEARS" if q["clears_0p20_bar"] else "FAILS"}')
            print('  F required to clear the 0.20 % bar:')
            for k, v in p["required_F_us"].items():
                print(f'    {k:<40}: {v:6.2f} us')
        report[shape] = block

    if args.json:
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
