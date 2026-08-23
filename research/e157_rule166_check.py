"""E157 R0: test the FINDING 286 ranked row law without recovering round counts.

The published `effective_mean_draft_len` bounds the tokens emitted per round.
Every round emits one primary token plus its accepted drafts, and the accepted
drafts cannot exceed the drafted ones, so

    1 <= tokens_per_round <= 1 + edl

for every prompt, with no fitted parameter and no recovered round count. The
clean decode time per round is therefore bracketed by

    decode_us_per_token <= clean_us_per_round <= decode_us_per_token * (1 + edl)

where `decode_us_per_token = 1e6 * (mtp_spt - prefill_spt)`.

The bracket is tight exactly where a prompt barely drafts. `plutarch` runs at
`edl = 0.1557`, so its bracket is only 16 % wide, and its verify width is
published as `edl + 1 = 1.156`. That single prompt tests any proposed ranked
round-cost law at a known width.

Sign convention in words: a positive `miss` means the law predicts LESS time
than the receipt can possibly have spent.

    python3 research/e157_rule166_check.py [RECEIPT_ID8]
"""
from __future__ import annotations

import json
import sys

CACHE = "/tmp/yukon-board/full.json"
DECODE_TOKENS = 512
PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}
# Candidate laws for ranked clean us per round.
LAWS = {
    # FINDING 286 as quoted in E157 F3.
    "finding_286_rows": lambda width, tokens: 8434.0 + 5657.0 * width,
    # FINDING 281 as recorded in the ledger, tokens per round frame.
    "finding_281_tokens": lambda width, tokens: 25409.0 + 4291.0 * tokens,
    # RULE 166 applied to the pooled local law: local / 2.75.
    "rule_166_from_local": lambda width, tokens: (24365.0 + 15754.0 * width) / 2.75,
}


LOCAL_A_US = 23421.07  # E154 R2, harness=local, M4 Pro 48 GiB
LOCAL_B_US = 15708.50  # E154 R2 within-d contrast; E155 R1 pooled gives 15754
SHIPPED_H = 0.18
# E157 R0 within-prompt ranked sweep on `drama`, 21 uniquely recovered cells,
# widths 1.000 to 8.914, R^2 0.9957. harness=ranked.
RANKED_SWEEP = (28896.0, 1082.0, 598.0)


def ranked_round_us(width: float) -> float:
    a, b, c = RANKED_SWEEP
    return a + b * width + c * width * width


def ranked_marginal_us(width: float) -> float:
    _, b, c = RANKED_SWEEP
    return b + c * (2.0 * width - 1.0)


def local_round_us(width: float) -> float:
    return LOCAL_A_US + LOCAL_B_US * width


def transfer_summary() -> dict:
    """What the depth rule actually consumes is `h = marginal / (fixed +
    marginal)`. A single-divisor host transfer cannot change `h` at all, so it
    cannot explain a depth policy that reverses between hosts.

    The ranked side uses the within-prompt width sweep, which measures the
    marginal price at each width instead of averaging it across prompts.
    """
    local_h = LOCAL_B_US / local_round_us(1.0)
    ratios = {
        str(w): local_round_us(float(w)) / ranked_round_us(float(w))
        for w in range(1, 10)
    }
    mid = [ratios[str(w)] for w in range(3, 9)]
    return {
        "local_h_marginal": local_h,
        "shipped_h": SHIPPED_H,
        "single_divisor_h": local_h,
        "single_divisor_preserves_h": True,
        "ranked_h_by_width": {
            str(w): ranked_marginal_us(float(w)) / ranked_round_us(1.0)
            for w in range(1, 10)
        },
        "round_total_local_over_ranked_by_width": ratios,
        "rule_166_claim": "2.75 +- 0.03 for M in [3, 8]",
        "measured_ratio_over_m_three_to_eight": [min(mid), max(mid)],
        "reading": (
            "the measured round-total ratio runs 1.28 at width one and 1.88 "
            "to 2.07 over M in [3, 8], not a constant 2.75; and the ranked "
            "depth price h rises with width from 0.055 to 0.33, while a "
            "single divisor would leave the local h of 0.40 unchanged at "
            "every width"
        ),
    }


def load_rows() -> list[dict]:
    with open(CACHE) as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else payload["submissions"]


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "0cf1637e"
    rows = load_rows()
    hits = [r for r in rows if str(r.get("id", "")).startswith(target)]
    if len(hits) != 1:
        raise SystemExit(f"{target}: {len(hits)} matches, need exactly one")
    row = hits[0]

    entries = sorted(
        (PROMPT_NAMES.get(e["prompt_sha256"][:8], e["prompt_sha256"][:8]), e)
        for e in row["officialMetrics"]["per_prompt"]
    )

    out = {
        "experiment": "e157-r0-ranked-law-bracket-test",
        "harness": "ranked",
        "official_or_ranked_score": True,
        "receipt": row["id"][:8],
        "official_score": row["officialScore"],
        "method": (
            "tokens per round lies in [1, 1 + edl] by construction, so the "
            "clean time per round is bracketed with no fitted parameter"
        ),
        "per_prompt": [],
    }

    print(f"=== {row['id'][:8]}  score {row['officialScore']:.8f} ===")
    header = (
        f"{'prompt':9s} {'width':>6s} {'lo us/rnd':>10s} {'hi us/rnd':>10s} "
    )
    for name in LAWS:
        header += f"{name[:14]:>15s}"
    print(header)

    for name, entry in entries:
        edl = entry["effective_mean_draft_len"]
        width = edl + 1.0
        decode_us_per_token = 1e6 * (
            entry["mtp_seconds_per_token_mean"] - entry["prefill_seconds_per_token"]
        )
        low = decode_us_per_token
        high = decode_us_per_token * (1.0 + edl)
        record = {
            "prompt": name,
            "verify_width_mean": width,
            "clean_us_per_round_lower_bound": low,
            "clean_us_per_round_upper_bound": high,
            "bracket_width_pct": 100.0 * (high / low - 1.0),
            "laws": {},
        }
        line = f"{name:9s} {width:6.3f} {low:10.0f} {high:10.0f} "
        for label, law in LAWS.items():
            # Evaluate each law at the tokens per round that best suits it, so
            # the test cannot fail merely because tokens per round is unknown.
            best = None
            for step in range(0, 101):
                tokens = 1.0 + edl * step / 100.0
                predicted = law(width, tokens)
                observed = decode_us_per_token * tokens
                miss = (observed - predicted) / observed
                if best is None or abs(miss) < abs(best[0]):
                    best = (miss, predicted, observed, tokens)
            miss, predicted, observed, tokens = best
            record["laws"][label] = {
                "best_case_relative_miss": miss,
                "predicted_us_per_round": predicted,
                "observed_us_per_round": observed,
                "tokens_per_round_used": tokens,
            }
            line += f"{100 * miss:+14.1f}%"
        print(line)
        out["per_prompt"].append(record)

    print()
    for label in LAWS:
        misses = [r["laws"][label]["best_case_relative_miss"] for r in out["per_prompt"]]
        worst = max(misses, key=abs)
        rms = (sum(m * m for m in misses) / len(misses)) ** 0.5
        out.setdefault("law_summary", {})[label] = {
            "worst_best_case_relative_miss": worst,
            "rms_best_case_relative_miss": rms,
        }
        print(
            f"{label:22s} worst best-case miss {100 * worst:+7.1f} %   "
            f"rms {100 * rms:6.1f} %"
        )

    out["transfer"] = transfer_summary()
    for key, value in out["transfer"].items():
        if isinstance(value, float):
            print(f"{key:44s} {value:9.4f}")
        elif isinstance(value, str):
            print(f"{key:44s} {value}")

    path = "research/e157-artifacts/e157_rule166_check.json"
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
