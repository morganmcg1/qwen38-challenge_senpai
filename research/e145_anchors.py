#!/usr/bin/env python3
"""E145 R0b: the width-1 intercept anchors, and the seed prefill inside them.

`harness=ranked` for every board quantity, `harness=local` for every leg
quantity. Zero GPU. Answers F3 section 3.

WHAT THIS ADDS TO F219. F219 converts a board row into a per-prompt round cost
with `round_us = mtp_seconds_per_token_mean * 512 / R`. That identity is exact,
but `mtp_seconds_per_token_mean` is a SEED-INCLUSIVE quantity. Two authorities
say so:

  fixtures/qwen3_8_27b_mtp_track.json
      proposed_scoring.prefill_component
      "none; seed prefill is charged inside the decode measurement,
       identically on both legs"
  senpai/program.md
      "Both seed processing and decoding are included in the same timed leg"

and the local report reproduces the same construction exactly:
`parent_measured_seconds_per_token == (seed_prefill_seconds + sum(blocks)) /
512`, which `research/e145_read.py` checks on every leg through
`basis_residual_s`.

So the F219 round cost is not per-round work. It is

    round_us(prompt) = P / R(prompt)  +  mean per-round work(prompt)

with `P` the whole leg's seed prefill, a constant that no schedule can move.
`R = 512 / (1 + a*d)`, so `P / R = P * (1 + a*d) / 512`, which is LINEAR in
tokens per round and therefore approximately linear in realised width. A width
cost curve fitted to F219-style round costs absorbs that term as a spurious
SLOPE. It leaves the intercept almost untouched, because the prefill term at
`w = 1` is only `P * (1 - a) / 512`.

The arithmetic still closes for a seconds-per-token forecast, because
`P * tokens_per_round / 512` divided by `tokens_per_round` returns the constant
`P / 512`. It does NOT close for any physical reading of the slope: "one more
verify row costs 3,446 us" is not what that number measures. R2 measures the
blocks-only curve, which is prefill-free, so the difference between the two
shapes identifies `P` in the ranked frame.

The `prefill_seconds_per_token` field on the board is a DIFFERENT measurement -
the standalone prefill-throughput leg - and is about 0.00103 on every prompt of
every row. It is not the seed prefill charged inside the decode leg and must
not be substituted for it.
"""

import argparse
import json
import os
import pathlib
import statistics
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = "https://api.yukon.org/api"
BENCHMARK_ID = "5d1ee4d7-80bd-4555-b182-6505f26ef495"
CACHE = pathlib.Path("/tmp/yukon-board/full.json")

# `research/board_per_prompt.py:41-49` is the canonical mapping.
PROMPT = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}

# The replayed curve under test, `per_round` form, widths 1..9, microseconds.
REPLAYED = {
    1: 31173.2, 2: 34619.3, 3: 38065.4, 4: 41511.4, 5: 44957.5,
    6: 61198.8, 7: 62824.9, 8: 70315.4, 9: 75638.9,
}

# F92's pinned ranked round counts, the object F219's identity was validated
# against. Used here only to price the drafting contamination in an anchor.
F92_ROUNDS = {
    "beagle": 110, "botany": 81, "drama": 252, "essays": 92,
    "medicine": 90, "plutarch": 487, "republic": 93, "travel": 212,
}

# One local `beagle_a` depth-0 leg, `r1-benchfixture-p3-serial`, is the only
# place where the seed prefill and the per-round work are separately visible.
LOCAL_SERIAL = {
    "leg": "r1-benchfixture-p3-serial",
    "seed_prefill_s": 4.004016041755676,
    "spt_seed_inclusive": 0.073703887,
    "spt_blocks_only": 0.065897046,
    "rounds": 512,
}


def unwrap(payload) -> list:
    if isinstance(payload, list):
        return payload
    for key in ("submissions", "data", "items", "results"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise SystemExit(f"unrecognised board payload keys: {sorted(payload)}")


def fetch(refresh: bool) -> list:
    if CACHE.exists() and not refresh:
        return unwrap(json.loads(CACHE.read_text()))
    token = os.environ["YUKON_API_TOKEN"]
    url = f"{BASE}/benchmarks/{BENCHMARK_ID}/submissions?all=true"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=180) as fh:
        payload = json.load(fh)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(payload))
    return unwrap(payload)


def row(rows: list, prefix: str) -> dict:
    hit = [r for r in rows if str(r.get("id", "")).startswith(prefix)]
    if len(hit) != 1:
        raise SystemExit(f"{prefix}: expected 1 row, found {len(hit)}")
    return hit[0]


def per_prompt(r: dict) -> dict:
    return {PROMPT[p["prompt_sha256"][:8]]: p
            for p in r["officialMetrics"]["per_prompt"]}


def curve_at(width: float) -> float:
    """Linear interpolation inside the replayed curve."""
    lo = max(1, min(8, int(width)))
    frac = width - lo
    return REPLAYED[lo] + frac * (REPLAYED[lo + 1] - REPLAYED[lo])


def width1_anchor(spt: float, dlen: float, nondraft: int,
                  rounds: float) -> dict:
    """Width-1 round cost implied by one prompt that barely drafts.

    The rounds that DO draft ran at a width well above 1, so their excess is
    priced on the replayed curve and removed. That correction uses the object
    under test, so it is reported separately; it is under 2 % here.
    """
    round_us = spt * 1e6 * 512.0 / rounds
    drafting = max(0.0, rounds - nondraft)
    width = 1.0 + (dlen * rounds / drafting) if drafting > 0 else 1.0
    excess = (drafting / rounds) * (curve_at(width) - REPLAYED[1])
    return {
        "mtp_spt": spt, "draft_len": dlen, "non_drafting_rounds": nondraft,
        "rounds_used": rounds, "round_us_raw": round_us,
        "drafting_rounds": drafting, "mean_drafting_width": width,
        "drafting_excess_us": excess, "round_us_width1": round_us - excess,
    }


def near_serial_anchor(r: dict, name: str) -> dict:
    """Aggregate the width-1 anchor over every prompt of a near-serial row.

    The accept rate is not published. Here it barely matters: the whole row
    drafts about 0.04 tokens per round, so accepted drafts cannot exceed
    `512 - non_drafting_round_count`, which bounds `R` inside `[502, 512]`.
    `R = 512` is the base case and is also the conservative one, because a
    smaller `R` raises the implied round cost.
    """
    out = {"name": name, "submission": r["id"][:8],
           "solver": r["solverUsername"], "status": r["status"],
           "score": r.get("officialScore"), "prompts": {}}
    raw, corrected, hi = [], [], []
    for p, e in sorted(per_prompt(r).items()):
        spt = e["mtp_seconds_per_token_mean"]
        dlen = e["effective_mean_draft_len"]
        nondraft = e["non_drafting_round_count"]
        entry = width1_anchor(spt, dlen, nondraft, 512.0)
        entry["rounds_lower_bound"] = float(nondraft)
        entry["round_us_at_rounds_lower_bound"] = spt * 1e6 * 512.0 / nondraft
        out["prompts"][p] = entry
        raw.append(entry["round_us_raw"])
        corrected.append(entry["round_us_width1"])
        hi.append(entry["round_us_at_rounds_lower_bound"])
    out["round_us_raw_mean"] = statistics.fmean(raw)
    out["round_us_width1_mean"] = statistics.fmean(corrected)
    out["round_us_upper_bound_mean"] = statistics.fmean(hi)
    out["round_us_width1_spread_pct"] = (
        100.0 * (max(corrected) - min(corrected))
        / statistics.fmean(corrected))
    return out


def prefill_decomposition(f219: dict, prefills_s) -> dict:
    """How much of each F219 round cost is amortised seed prefill.

    `P / R` is charged to every round by the identity, and it is larger on
    exactly the prompts that draft deepest, because they run fewer rounds.
    """
    out = {}
    for p_total in prefills_s:
        per_prompt_share = {}
        for p, e in f219.items():
            term = p_total * 1e6 / e["rounds"]
            per_prompt_share[p] = {
                "prefill_us_per_round": term,
                "share_of_round_pct": 100.0 * term / e["round_us"],
                "work_us_per_round": e["round_us"] - term,
            }
        median_pair = ("beagle", "essays")
        out[f"{p_total:.3f}"] = {
            "prefill_seconds_total": p_total,
            "per_prompt": per_prompt_share,
            "mean_share_pct": statistics.fmean(
                v["share_of_round_pct"] for v in per_prompt_share.values()),
            "median_pair_share_pct": statistics.fmean(
                per_prompt_share[p]["share_of_round_pct"]
                for p in median_pair),
            # d(P/R)/dw = P * a / 512 with R = 512 / (1 + a(w-1)).
            "slope_us_per_width_at_accept_0.83":
                p_total * 1e6 * 0.83 / 512.0,
            "share_of_replayed_slope_pct":
                100.0 * (p_total * 1e6 * 0.83 / 512.0)
                / (REPLAYED[2] - REPLAYED[1]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--json", default="research/e145-artifacts/r0b-anchors.json")
    args = ap.parse_args()

    rows = fetch(args.refresh)
    r_1760 = row(rows, "1760479a")
    r_623 = row(rows, "623e77af")
    r_c915 = row(rows, "c91581eb")

    # The accept rate is not published and it is NOT common across prompts:
    # F219's own `tokens per round` column implies 0.833 on beagle and 0.333
    # on plutarch. F92's pinned round counts are therefore used directly
    # instead of reconstructing `R` from one assumed accept rate.
    f219 = {}
    for p, e in per_prompt(r_1760).items():
        rounds = float(F92_ROUNDS[p])
        dlen = e["effective_mean_draft_len"]
        f219[p] = {
            "mtp_spt": e["mtp_seconds_per_token_mean"],
            "draft_len": dlen,
            "rounds": rounds,
            "rounds_source": "F92 pinned ranked round count",
            "implied_accept_rate": (512.0 / rounds - 1.0) / dlen if dlen else 0,
            "non_drafting_rounds": e["non_drafting_round_count"],
            "round_us": e["mtp_seconds_per_token_mean"] * 1e6 * 512.0 / rounds,
        }

    # A near-serial row is the only board object with almost no width mass, so
    # its round cost is almost the intercept. `c91581eb` was reported as a row
    # with zero accepted drafts; the live payload says otherwise, so the claim
    # is checked here rather than assumed.
    c915_zero_draft = all(
        e["effective_mean_draft_len"] == 0.0
        and e["non_drafting_round_count"] == 512
        for e in per_prompt(r_c915).values())

    plut = per_prompt(r_1760)["plutarch"]
    anchors = {
        "replayed_curve_w1": {
            "name": "replayed E134 curve, w=1",
            "round_us_width1": REPLAYED[1],
            "anchor_receipt": "623e77af",
            "note": "the object under test",
        },
        "f219_plutarch": {
            "name": "F219 plutarch @1760479a",
            "submission": r_1760["id"][:8],
            "solver": r_1760["solverUsername"],
            **width1_anchor(plut["mtp_seconds_per_token_mean"],
                            plut["effective_mean_draft_len"],
                            plut["non_drafting_round_count"],
                            float(F92_ROUNDS["plutarch"])),
        },
        "c91581eb": near_serial_anchor(r_c915, "near-serial board row"),
    }
    three = [
        anchors["replayed_curve_w1"]["round_us_width1"],
        anchors["f219_plutarch"]["round_us_width1"],
        anchors["c91581eb"]["round_us_width1_mean"],
    ]
    spread_pct = 100.0 * (max(three) - min(three)) / statistics.fmean(three)

    # Uniform-level-factor estimate of the ranked seed prefill. The local
    # depth-0 leg separates P from work exactly; the near-serial board row
    # gives the same total in the ranked frame. Assuming ONE factor scales
    # both parts is the weak step, and it is labelled as such: prefill is
    # compute bound and decode is bandwidth bound, so M5 need not move them
    # together. R2's blocks-only curve replaces this estimate with a fit.
    local_total_s = LOCAL_SERIAL["spt_seed_inclusive"] * 512.0
    local_work_s = LOCAL_SERIAL["spt_blocks_only"] * 512.0
    ranked_total_s = statistics.fmean(
        e["mtp_seconds_per_token_mean"]
        for e in per_prompt(r_c915).values()) * 512.0
    level = local_total_s / ranked_total_s
    prefill_ranked_s = LOCAL_SERIAL["seed_prefill_s"] / level

    decomposition = prefill_decomposition(
        f219, [0.5, 1.0, prefill_ranked_s, 2.0])

    result = {
        "harness": "ranked board rows plus one local depth-0 leg",
        "gpu_used": False,
        "prefill_is_inside_the_scored_leg": True,
        "prefill_authority": [
            "fixtures/qwen3_8_27b_mtp_track.json"
            " proposed_scoring.prefill_component",
            "senpai/program.md, Goal And Score",
        ],
        "board_prefill_seconds_per_token_is_a_different_measurement": True,
        "c91581eb_is_a_zero_accepted_draft_row": c915_zero_draft,
        "f219_round_costs": f219,
        "anchors": anchors,
        "three_anchor_spread_pct": spread_pct,
        "three_anchor_agree_within_1p5_pct": spread_pct <= 1.5,
        "local_serial_leg": LOCAL_SERIAL,
        "ranked_prefill_estimate": {
            "local_total_seconds": local_total_s,
            "local_work_seconds": local_work_s,
            "local_prefill_seconds": LOCAL_SERIAL["seed_prefill_s"],
            "ranked_near_serial_total_seconds": ranked_total_s,
            "uniform_level_factor": level,
            "prefill_seconds_ranked": prefill_ranked_s,
            "assumption": "one level factor scales prefill and decode alike",
        },
        "prefill_decomposition": decomposition,
    }
    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"e145 anchors -> {out}\n")

    print("1. Is `c91581eb` a zero-accepted-draft row?"
          f"  {c915_zero_draft}")
    for p, e in sorted(anchors["c91581eb"]["prompts"].items()):
        print(f"   {p:<9} dlen {e['draft_len']:.6f}"
              f"   non-drafting {e['non_drafting_rounds']:>4} of 512"
              f"   drafting rounds {e['drafting_rounds']:>5.1f}"
              f"   width {e['mean_drafting_width']:.2f}")

    print("\n2. Three width-1 anchors, drafting contamination removed")
    a = anchors
    print(f"   replayed curve w=1        {REPLAYED[1]:>9.1f} us"
          f"   (built on 623e77af)")
    print(f"   F219 plutarch @1760479a   "
          f"{a['f219_plutarch']['round_us_width1']:>9.1f} us"
          f"   raw {a['f219_plutarch']['round_us_raw']:.1f},"
          f" contamination {a['f219_plutarch']['drafting_excess_us']:.0f}")
    print(f"   c91581eb 8-prompt mean    "
          f"{a['c91581eb']['round_us_width1_mean']:>9.1f} us"
          f"   raw {a['c91581eb']['round_us_raw_mean']:.1f},"
          f" spread {a['c91581eb']['round_us_width1_spread_pct']:.2f} %")
    print(f"   spread across the three   {spread_pct:>9.2f} %"
          f"   within 1.5 %: {spread_pct <= 1.5}")

    print("\n3. Seed prefill is INSIDE the scored leg, so it is inside every")
    print("   F219 round cost as `P / R`, and R is small on deep prompts.")
    print(f"   uniform-level-factor estimate of ranked P:"
          f" {prefill_ranked_s:.3f} s"
          f"   (level {level:.3f})")
    print(f"   {'P (s)':>7} {'mean share':>11} {'medpair share':>14}"
          f" {'slope us/width':>15} {'of replayed slope':>18}")
    for key, d in decomposition.items():
        print(f"   {float(key):>7.3f} {d['mean_share_pct']:>10.2f} %"
              f" {d['median_pair_share_pct']:>13.2f} %"
              f" {d['slope_us_per_width_at_accept_0.83']:>15.0f}"
              f" {d['share_of_replayed_slope_pct']:>17.1f} %")

    key = f"{prefill_ranked_s:.3f}"
    print(f"\n   per-prompt at P = {prefill_ranked_s:.3f} s")
    print(f"   {'prompt':<9} {'R':>7} {'round us':>10} {'prefill us':>11}"
          f" {'share':>8} {'work us':>10}")
    for p in sorted(f219):
        e = f219[p]
        d = decomposition[key]["per_prompt"][p]
        print(f"   {p:<9} {e['rounds']:>7.1f} {e['round_us']:>10.1f}"
              f" {d['prefill_us_per_round']:>11.1f}"
              f" {d['share_of_round_pct']:>7.2f} %"
              f" {d['work_us_per_round']:>10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
