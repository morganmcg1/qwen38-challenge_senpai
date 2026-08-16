"""Score the E9 Part B default-flip arms against the advisor's decision table.

  python3 research/e9_partb.py

Part B asks one question: with MLX_QWEN_MTP_DRAFT_BITS unset, does the in-tree
default deliver the 3-bit draft readout in scored time? On the cap-7 base the
same arm doubles as the first direct measurement of the composed candidate
(segmentedVerifyDepthCap 7 + 3-bit draft head) with no environment cooperation.
"""

import json
import pathlib

R4_MS = 1.0480  # per-readout 4-bit cost solved from the English cap-8 arms
KERNEL_SPEEDUP_3B = 0.242
ROOT = pathlib.Path(__file__).resolve().parent.parent / ".mlxfast-private" / "draft-bits"


def leg(tag):
    return json.loads((ROOT / tag / "amdahl.json").read_text())["mtp_leg"]


def ms(l):
    return l["parent_measured_seconds_per_token"] * 1000


def ms_per_round(l):
    return ms(l) * l["emitted_token_total"] / l["round_count"]


def main():
    ctl, dflt = leg("e9-cap7-b4"), leg("e9-cap7-bdefault")
    old4, old3, olddef = leg("e9-flip-b4"), leg("e9-flip-b3"), leg("e9-flip-bdefault")

    print("=== Part B on cap-7 base 8970d775, one binary 3a62b25ce753 ===")
    for name, l in (("control(env=4)", ctl), ("default(unset)", dflt)):
        print(
            f"  {name:<15} {ms(l):8.4f} ms/token  a={l['accepted_draft_rate']:.16f}"
            f"  rounds={l['round_count']:3d}  D={l['effective_mean_draft_len']:.4f}"
            f"  match={l['all_tokens_matched']}  div={l['residual_divergence_count']}"
        )
    print(f"  delta {(ms(dflt) / ms(ctl) - 1) * 100:+.4f}%")
    print("\n=== advisor decision table ===")
    for label, target, l in (
        ("cap-7 control  ~33.99", 33.99, ctl),
        ("BANKED         ~33.34", 33.34, dflt),
        ("pre-merge stale~35.09", 35.09, ctl),
    ):
        print(f"  {label}: measured {ms(l):8.4f}  offset {(ms(l) / target - 1) * 100:+.4f}%")

    print("\n=== decomposition of the default arm ===")
    share = ctl["effective_mean_draft_len"] * R4_MS / ms_per_round(ctl)
    print(f"  ctl ms/round {ms_per_round(ctl):.3f}   D {ctl['effective_mean_draft_len']:.4f}"
          f"   readout share {share * 100:.2f}%")
    print(f"  mechanism ceiling at 24.2% faster readout: {-KERNEL_SPEEDUP_3B * share * 100:.3f}% ms/token")
    print(
        f"  measured {(ms(dflt) / ms(ctl) - 1) * 100:+.3f}%"
        f"  =  per-round {(ms_per_round(dflt) / ms_per_round(ctl) - 1) * 100:+.3f}%"
        f"  +  round-count {(dflt['round_count'] / ctl['round_count'] - 1) * 100:+.3f}%"
        f"  ({ctl['round_count']}->{dflt['round_count']})"
    )
    saving = KERNEL_SPEEDUP_3B * R4_MS * ctl["effective_mean_draft_len"]
    tpr_be = ctl["tokens_per_round"] * (1 - saving / ms_per_round(ctl))
    floor = (tpr_be - 1) / ctl["effective_mean_draft_len"]
    margin = dflt["accepted_draft_rate"] - floor
    print(
        f"  break-even floor {floor:.6f}   default a {dflt['accepted_draft_rate']:.6f}"
        f"   margin {margin:+.6f}  -> {'PASS' if margin > 0 else 'FAIL'}"
    )

    print("\n=== composition, same host, arms 20 minutes apart ===")
    print(f"  cap-8 control {ms(old4):.4f} -> cap-7 control {ms(ctl):.4f}"
          f"  = {(ms(ctl) / ms(old4) - 1) * 100:+.4f}%   (Alphonse reported -3.1215%)")
    print(f"  cap-8 control {ms(old4):.4f} -> cap-7 default {ms(dflt):.4f}"
          f"  = {(ms(dflt) / ms(old4) - 1) * 100:+.4f}%   full stack")
    predicted = (ms(ctl) / ms(old4)) * (ms(old3) / ms(old4))
    measured = ms(dflt) / ms(old4)
    print(f"  multiplicative prediction {predicted:.6f} vs measured {measured:.6f}"
          f"  -> interaction {(measured / predicted - 1) * 100:+.3f}%")

    print("\n=== conversion check (cap-8 arms, three arms one binary) ===")
    print(f"  a(default) == a(env=3) exactly? {olddef['accepted_draft_rate'] == old3['accepted_draft_rate']}")
    print(f"  a(default) != a(env=4) exactly? {olddef['accepted_draft_rate'] != old4['accepted_draft_rate']}")


if __name__ == "__main__":
    main()
