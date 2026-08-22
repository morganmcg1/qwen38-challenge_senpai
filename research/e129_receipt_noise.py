"""Per-prompt schedule invariance and the serial-leg noise scale for one receipt pair.

The serial leg is produced by the runner-owned prebuilt baseline workspace, so
candidate-editable code cannot move it (program.md, ranked causal boundary).
Its per-prompt spread is therefore a measured null distribution, and it sets the
scale against which any candidate-leg effect must be judged.

harness=ranked. Read-only over the cached Yukon board.
"""

import json
import math
import sys

from e129_schedule_invariance import NAMES, per_prompt, signature

BOARD = "/tmp/yukon-board/full.json"


def load(prefix):
    for r in json.load(open(BOARD)):
        if r["id"].startswith(prefix):
            return r
    raise SystemExit(f"no board row for {prefix}")


def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(var)
    return m, sd, sd / math.sqrt(n)


def main(new_id, old_id):
    new, old = load(new_id), load(old_id)
    a, b = per_prompt(new), per_prompt(old)

    print(f"harness=ranked   new {new['id'][:8]}  old {old['id'][:8]}")
    print()
    print("--- schedule invariance, both lines, all eight prompts ---")
    print(f"{'prompt':<10}{'draftlen_new':>14}{'draftlen_old':>14}"
          f"{'nondraft_new':>14}{'nondraft_old':>14}  same")
    all_same = True
    for n in NAMES.values():
        dn = a[n]["effective_mean_draft_len"]
        do = b[n]["effective_mean_draft_len"]
        rn = a[n]["non_drafting_round_count"]
        ro = b[n]["non_drafting_round_count"]
        same = round(dn, 9) == round(do, 9) and rn == ro
        all_same &= same
        print(f"{n:<10}{dn:>14.9f}{do:>14.9f}{rn:>14}{ro:>14}  "
              f"{'yes' if same else 'NO'}")
    print(f"signatures equal over all {len(NAMES)} prompts: "
          f"{signature(a) == signature(b)}  (elementwise {all_same})")
    print()

    cand, ser = [], []
    for n in NAMES.values():
        cand.append((1.0 - a[n]["mtp_seconds_per_token_mean"]
                     / b[n]["mtp_seconds_per_token_mean"]) * 100)
        ser.append((a[n]["serial_seconds_per_token_mean"]
                    / b[n]["serial_seconds_per_token_mean"] - 1) * 100)

    cm, csd, cse = stats(cand)
    sm, ssd, sse = stats(ser)
    print("--- leg change, positive candidate value = candidate faster ---")
    print(f"candidate leg  mean {cm:+.4f} %  sd {csd:.4f}  se {cse:.4f}  "
          f"2se band [{cm - 2 * cse:+.4f}, {cm + 2 * cse:+.4f}] %")
    print(f"serial leg     mean {sm:+.4f} %  sd {ssd:.4f}  se {sse:.4f}  "
          f"2se band [{sm - 2 * sse:+.4f}, {sm + 2 * sse:+.4f}] %")
    print("the serial row is a pure null: candidate code cannot move it.")
    print()

    print("--- where the published score actually moved ---")
    sn = sorted((a[n]["raw_ratio_of_means"], n) for n in NAMES.values())
    so = sorted((b[n]["raw_ratio_of_means"], n) for n in NAMES.values())
    cn, co = sn[3:5], so[3:5]
    score_n = (cn[0][0] + cn[1][0]) / 2
    score_o = (co[0][0] + co[1][0]) / 2
    print(f"new central two {cn[0][1]} {cn[0][0]:.6f} | {cn[1][1]} {cn[1][0]:.6f}"
          f"  -> median {score_n:.8f}")
    print(f"old central two {co[0][1]} {co[0][0]:.6f} | {co[1][1]} {co[1][0]:.6f}"
          f"  -> median {score_o:.8f}")
    d = score_n - score_o
    print(f"score delta {d:+.8f}")
    if {x[1] for x in cn} == {x[1] for x in co}:
        for (_, n) in cn:
            part = (a[n]["raw_ratio_of_means"] - b[n]["raw_ratio_of_means"]) / 2
            cg = (1.0 - a[n]["mtp_seconds_per_token_mean"]
                  / b[n]["mtp_seconds_per_token_mean"]) * 100
            sg = (a[n]["serial_seconds_per_token_mean"]
                  / b[n]["serial_seconds_per_token_mean"] - 1) * 100
            print(f"  {n:<10} contributes {part:+.8f} "
                  f"({100 * part / d:5.1f} % of the move); "
                  f"candidate {cg:+.3f} %, serial {sg:+.3f} % "
                  f"(serial share {100 * sg / (cg + sg):5.1f} %)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["623e77af", "0c6191b7"]))
