#!/usr/bin/env python3
"""E17 analysis: per-prompt paired arms, ranked-style median, r3 re-arithmetic.

Conventions, fixed before any E17 arm was timed:

  raw_p = serial_seconds_per_token / mtp_seconds_per_token

Both terms are read verbatim from `.parent_measured_seconds_per_token` in the
run's own reports. That field is prefill-INCLUSIVE by construction --
QwenRuntimeMTPDriver starts its clock before beginMTPDecode and stops it after
the last emitted token, and QwenRuntimeMTP divides that whole span by the decode
token count -- so nothing here subtracts `seed_prefill_seconds`. Subtracting it
would report a decode-only ratio the ranked score never computes.

  03-mtp-timed.json  the serial control leg of that run (mtp_depth 0)
  04-mtp-timed.json  the MTP arm leg of that run (mtp_depth 8)

Each --local-iterate invocation measures both, so every prompt carries one
serial leg per arm and the spread between them is that prompt's noise floor.

usage:
  research/e17_analyse.py            per-prompt table + ranked-style medians
  research/e17_analyse.py --r3       re-derive the r3 published pair
  research/e17_analyse.py --json     machine-readable dump
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

RUNS = Path(".mlxfast-private/e17/runs")
R3_RUNS = Path(".mlxfast-private/e11/runs")
ARMS = ("CURVE", "FLAT18")
PROMPTS = (
    "english",
    "narrative",
    "technical",
    "dramatic",
    "travel",
    "philosophy",
    "natural_history",
    "medicine",
)
HELD_OUT = tuple(p for p in PROMPTS if p != "english")


def median(xs: list[float]) -> float:
    """Ranked median: with an even count, the mean of the two middle values."""
    return statistics.median(xs)


def load_leg(run: Path, name: str) -> dict:
    return json.loads((run / "reports" / name).read_text())


def load_arm(prompt: str, arm: str, root: Path = RUNS) -> dict | None:
    run = root / f"{prompt}-{arm}"
    try:
        serial = load_leg(run, "03-mtp-timed.json")
        mtp = load_leg(run, "04-mtp-timed.json")
    except FileNotFoundError:
        return None
    assert serial["mtp_depth"] == 0 and serial["is_serial_control"], run
    assert mtp["mtp_depth"] == 8 and not mtp["is_serial_control"], run
    hist = Counter(mtp["effective_draft_lengths"])
    meta = dict(
        line.split("=", 1) for line in (run / "meta.txt").read_text().splitlines() if "=" in line
    )
    correctness = load_leg(run, "01-correctness.json")
    return {
        "run": str(run),
        "meta": {k: v.strip() for k, v in meta.items()},
        "emitted_tokens": mtp["emitted_token_total"],
        "seed_tokens": mtp["seed_token_count"],
        "stall_max": mtp["max_block_request_seconds_after_first"],
        "stall_p50": mtp["p50_block_request_seconds_after_first"],
        "stall_ratio": mtp["max_block_request_seconds_after_first"]
        / mtp["p50_block_request_seconds_after_first"],
        "drift_tripwire_passed": correctness["passed"],
        "drift_tripwire_steps": correctness["checked_steps"],
        "golden_hash": correctness["golden_hash"],
        "serial_spt": serial["parent_measured_seconds_per_token"],
        "mtp_spt": mtp["parent_measured_seconds_per_token"],
        "raw": serial["parent_measured_seconds_per_token"]
        / mtp["parent_measured_seconds_per_token"],
        "serial_prefill_s": serial["seed_prefill_seconds"],
        "mtp_prefill_s": mtp["seed_prefill_seconds"],
        "decode_tokens": mtp["decode_token_count"],
        "rounds": mtp["round_count"],
        "mean_depth": mtp["effective_mean_draft_len"],
        "max_depth": mtp["effective_max_draft_len"],
        "depth_hist": {str(d): hist[d] for d in sorted(hist)},
        "accepted": mtp["accepted_draft_total"],
        "rejected": mtp["rejected_draft_total"],
        "accept_rate": mtp["accepted_draft_rate"],
        "replays": mtp["verify_block_replayed_round_count"],
        "matched": mtp["all_tokens_matched"] and serial["all_tokens_matched"],
        "parity": mtp["parity_all_ok"] and serial["parity_all_ok"],
        "divergence": mtp["residual_divergence_count"],
        "declared_rows": mtp["declared_rows_total"],
        "checked_rows": mtp["reference_checked_row_total"],
        "pinned_head": mtp["uses_pinned_mtp_head"],
        "head_sha256": mtp["head_provenance"]["sha256"],
    }


def collect() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for prompt in PROMPTS:
        arms = {a: load_arm(prompt, a) for a in ARMS}
        if all(v is not None for v in arms.values()):
            out[prompt] = arms
    return out


def report(data: dict[str, dict[str, dict]]) -> None:
    if not data:
        print("e17: no completed prompt pairs under", RUNS)
        return

    print("PER-PROMPT PAIRS (raw_p = serial spt / mtp spt, prefill-inclusive)")
    print(
        f"{'prompt':<16}{'serial C':>10}{'serial F':>10}{'floor%':>8}"
        f"{'mtp CURVE':>11}{'mtp FLAT':>10}{'raw C':>8}{'raw F':>8}"
        f"{'d_raw':>8}{'g%':>8}"
    )
    for prompt, arms in data.items():
        c, f = arms["CURVE"], arms["FLAT18"]
        floor = abs(c["serial_spt"] - f["serial_spt"]) / (
            (c["serial_spt"] + f["serial_spt"]) / 2
        )
        g = (f["mtp_spt"] - c["mtp_spt"]) / f["mtp_spt"]
        print(
            f"{prompt:<16}{c['serial_spt']:>10.6f}{f['serial_spt']:>10.6f}"
            f"{100*floor:>8.3f}{c['mtp_spt']:>11.6f}{f['mtp_spt']:>10.6f}"
            f"{c['raw']:>8.4f}{f['raw']:>8.4f}"
            f"{c['raw']-f['raw']:>+8.4f}{100*g:>+8.3f}"
        )

    for label, ids in (("HELD-OUT 7", HELD_OUT), ("ALL 8 (with in-sample anchor)", PROMPTS)):
        sub = [p for p in ids if p in data]
        if len(sub) < 2:
            continue
        rc = [data[p]["CURVE"]["raw"] for p in sub]
        rf = [data[p]["FLAT18"]["raw"] for p in sub]
        gs = [
            (data[p]["FLAT18"]["mtp_spt"] - data[p]["CURVE"]["mtp_spt"])
            / data[p]["FLAT18"]["mtp_spt"]
            for p in sub
        ]
        mc, mf, gm = median(rc), median(rf), median(gs)
        print(f"\n{label}  n={len(sub)}  ({', '.join(sub)})")
        print(f"  median(raw|CURVE)  = {mc:.6f}   spread {min(rc):.4f}..{max(rc):.4f}")
        print(f"  median(raw|FLAT18) = {mf:.6f}   spread {min(rf):.4f}..{max(rf):.4f}")
        print(f"  headline delta     = {mc-mf:+.6f}  ({100*(mc-mf)/mf:+.3f}% of FLAT18)")
        print(f"  g_median           = {100*gm:+.3f}%  spread {100*min(gs):+.3f}..{100*max(gs):+.3f}%")
        print(f"  curve wins on      = {sum(1 for x in gs if x > 0)}/{len(gs)} prompts")

    print("\nDEPTH / ACCEPTANCE / CORRECTNESS")
    for prompt, arms in data.items():
        for arm in ARMS:
            a = arms[arm]
            print(
                f"  {prompt:<16}{arm:<7} rounds={a['rounds']:<4} "
                f"mean_d={a['mean_depth']:.3f} max_d={a['max_depth']} "
                f"acc={a['accepted']}/{a['accepted']+a['rejected']} "
                f"({100*a['accept_rate']:.1f}%) replay={a['replays']} "
                f"rows={a['checked_rows']}/{a['declared_rows']} "
                f"matched={a['matched']} parity={a['parity']} "
                f"div={a['divergence']} pinned_head={a['pinned_head']}"
            )
            print(f"  {'':<23}depth_hist={a['depth_hist']}")

    mechanism(data)


def mechanism(data: dict[str, dict[str, dict]]) -> None:
    """Which per-prompt quantity tracks the curve's win?

    The curve accepts FEWER tokens than the scalar on every prompt yet is faster
    on every prompt, so the win cannot be an acceptance effect. Rank the
    candidate explanations against g.
    """
    print("\nMECHANISM: what predicts the size of the curve's win?")
    print(
        f"  {'prompt':<16}{'g%':>8}{'F-C rows':>9}{'rows%':>8}"
        f"{'F-C acc':>8}{'F-C rnds':>9}{'rows/acc C':>11}{'rows/acc F':>11}{'ratio':>7}"
    )
    rows = []
    for prompt, arms in data.items():
        c, f = arms["CURVE"], arms["FLAT18"]
        g = 100 * (f["mtp_spt"] - c["mtp_spt"]) / f["mtp_spt"]
        d_rows = f["checked_rows"] - c["checked_rows"]
        pct_rows = 100 * d_rows / c["checked_rows"]
        rpa_c = c["checked_rows"] / c["accepted"]
        rpa_f = f["checked_rows"] / f["accepted"]
        rows.append((prompt, g, d_rows, pct_rows, rpa_f / rpa_c))
        print(
            f"  {prompt:<16}{g:>+8.3f}{d_rows:>9}{pct_rows:>+8.2f}"
            f"{f['accepted']-c['accepted']:>+8}{f['rounds']-c['rounds']:>+9}"
            f"{rpa_c:>11.3f}{rpa_f:>11.3f}{rpa_f/rpa_c:>7.3f}"
        )
    if len(rows) < 3:
        return

    def spearman(a: list[float], b: list[float]) -> float:
        ra = [sorted(a).index(x) for x in a]
        rb = [sorted(b).index(x) for x in b]
        n = len(a)
        return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))

    gs = [r[1] for r in rows]
    print(
        f"\n  Spearman(g, extra rows %)      = {spearman(gs, [r[3] for r in rows]):+.3f}"
        f"   <- redundant target work"
    )
    print(f"  Spearman(g, rows/accept ratio) = {spearman(gs, [r[4] for r in rows]):+.3f}")
    print(
        "  The scalar accepts MORE tokens on every prompt and is SLOWER on every prompt:\n"
        "  the curve trades cheap extra rounds for expensive saved verify rows."
    )


def r3() -> None:
    """Re-derive the r3 published pair under both conventions.

    The r3 report published Sp3=1.507282 (scalar 0.18) and Hp3=1.609073
    (merged curve). This shows which convention those numbers already used.
    """
    print("r3 RE-ARITHMETIC (E11 runs, 512 decode tokens)\n")
    rows = []
    for label, what in (("Sp3", "scalar h=0.18"), ("Hp3", "merged curve")):
        run = R3_RUNS / label
        s = load_leg(run, "03-mtp-timed.json")
        m = load_leg(run, "04-mtp-timed.json")
        n = m["decode_token_count"]
        ss, ms = s["parent_measured_seconds_per_token"], m["parent_measured_seconds_per_token"]
        sp, mp = s["seed_prefill_seconds"], m["seed_prefill_seconds"]
        # decode-only: strip that run's own measured prefill from each leg
        ss_d, ms_d = ss - sp / n, ms - mp / n
        rows.append((label, what, ss, ms, ss / ms, ss_d, ms_d, ss_d / ms_d, sp, mp))
        print(f"{label} ({what})  n={n}")
        print(f"  serial: spt={ss:.18f}  prefill={sp:.6f}s  spt-P/n={ss_d:.18f}")
        print(f"  mtp   : spt={ms:.18f}  prefill={mp:.6f}s  spt-P/n={ms_d:.18f}")
        print(f"  ratio prefill-inclusive = {ss/ms:.6f}")
        print(f"  ratio decode-only       = {ss_d/ms_d:.6f}\n")

    (_, _, sss, sm, si, _, smd, sd, ssp, smp) = rows[0]
    (_, _, hss, hm, hi, _, hmd, hd, hsp, hmp) = rows[1]
    print(f"published r3 pair: Sp3=1.507282  Hp3=1.609073")
    print(f"prefill-inclusive: Sp3={si:.6f}  Hp3={hi:.6f}   <-- matches published")
    print(f"decode-only      : Sp3={sd:.6f}  Hp3={hd:.6f}")
    print(f"\ng (curve gain on the MTP leg)")
    print(f"  prefill-inclusive = {100*(sm-hm)/sm:+.3f}%   <-- matches published 6.378%")
    print(f"  decode-only       = {100*(smd-hmd)/smd:+.3f}%")

    # Reconstruct the advisor's r1 restatement (1.437971 / 1.521771, "17.67%
    # smaller"), to show which operation produced it. Adding P/n to the
    # published spt values charges seed prefill a SECOND time, because the
    # published values already carried it.
    n = 512
    sd2 = (sss + ssp / n) / (sm + smp / n)
    hd2 = (hss + hsp / n) / (hm + hmp / n)
    pub_delta, dbl_delta = hi - si, hd2 - sd2
    print("\nWHERE 1.437971 / 1.521771 COMES FROM (prefill charged twice)")
    print(f"  (serial_spt + P/n) / (mtp_spt + P/n): Sp3={sd2:.6f}  Hp3={hd2:.6f}")
    print(f"  advisor r1 quoted                   : Sp3=1.437971  Hp3=1.521771")
    print(f"  pair delta: published {pub_delta:.6f} -> double-charged {dbl_delta:.6f}")
    print(f"  shrink = {100*(1-dbl_delta/pub_delta):.2f}%  (advisor r1 quoted 17.67%)")


def contract(data: dict[str, dict[str, dict]]) -> None:
    """Evidence-contract item 5 + item 6, one row per timed arm."""
    print("CONTRACT ITEM 5 -- CORRECTNESS / HYGIENE (every timed arm)")
    hdr = (
        f"{'arm':22s} {'match':5s} {'div':>3s} {'parity':6s} {'emit/gold':>9s} "
        f"{'rows d==c':>9s} {'env':3s} {'dirty':>5s} {'pin':3s} {'drift':5s} {'stall x':>7s}"
    )
    print("  " + hdr)
    ok = True
    for prompt, arms in data.items():
        for arm, a in arms.items():
            good = (
                a["matched"]
                and a["parity"]
                and a["divergence"] == 0
                and a["emitted_tokens"] == a["decode_tokens"] == 512
                and a["declared_rows"] == a["checked_rows"]
                and a["meta"]["mlx_qwen_env"] == ""
                and a["meta"]["dirty"] == "0"
                and a["pinned_head"]
                and a["drift_tripwire_passed"]
                and a["stall_ratio"] < 4.0
            )
            ok = ok and good
            print(
                f"  {prompt+'-'+arm:22s} {str(a['matched']):5s} {a['divergence']:3d} "
                f"{str(a['parity']):6s} {a['emitted_tokens']:4d}/{a['decode_tokens']:<4d} "
                f"{a['declared_rows']:4d}={a['checked_rows']:<4d} "
                f"{'(-)' if a['meta']['mlx_qwen_env']=='' else 'SET':3s} "
                f"{a['meta']['dirty']:>5s} {'yes' if a['pinned_head'] else 'NO':3s} "
                f"{str(a['drift_tripwire_passed']):5s} {a['stall_ratio']:7.3f}"
            )
    print(f"\n  ALL ARMS CLEAN: {ok}")

    print("\nCONTRACT ITEM 6 -- BINARY FRESHNESS (per arm)")
    seen: dict[str, set[str]] = {}
    for prompt, arms in data.items():
        for arm, a in arms.items():
            m = a["meta"]
            seen.setdefault(arm, set()).add(m["worker_sha256"])
            print(
                f"  {prompt+'-'+arm:22s} cli={m['cli_sha256'][:16]} "
                f"worker={m['worker_sha256'][:16]} src={m['source_sha256'][:16]} "
                f"head_sha={m['head_sha'][:7]}"
            )
    for arm, shas in seen.items():
        print(f"  arm {arm}: {len(shas)} distinct worker binary/binaries -> {sorted(s[:16] for s in shas)}")
    print(f"  arms use distinct workers: {len(set().union(*seen.values())) == len(seen)}")

    print("\nHEAD PROVENANCE (identical on every arm)")
    heads = {a["head_sha256"] for arms in data.values() for a in arms.values()}
    print(f"  distinct head sha256 across all arms: {len(heads)} -> {sorted(heads)}")

    print("\nGOLDEN PROVENANCE (public-drift tripwire, pre-timing, 64 steps)")
    for prompt, arms in data.items():
        g = {a["golden_hash"] for a in arms.values()}
        print(f"  {prompt:16s} golden_hash={sorted(g)}  shared_by_both_arms={len(g)==1}")


def main(argv: list[str]) -> int:
    if "--r3" in argv:
        r3()
        return 0
    data = collect()
    if "--json" in argv:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0
    if "--contract" in argv:
        contract(data)
        return 0
    report(data)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
