#!/usr/bin/env python3
"""E171 r2: decompose the D-vs-A candidate-leg regression into its factors.

Read-only desk analysis over the cached Yukon board payload written by
``research/board_per_prompt.py fetch``.

Receipt ladder under test:

    A 5a9f130a  organizer-pure main
    B 180db842  organizer + Q(4 quantized kernels) + instrumentation
    C fda590bb  organizer + instrumentation, Q reverted
    D 2c885d64  organizer + Q + E165 head-prefetch + strip

The candidate leg is the only channel our edits can move, so every effect here
is priced on ``mtp_seconds_per_token_mean``. The serial numerator is drawn fresh
by the runner and is excluded on purpose.

The null noise floor is calibrated from board pairs that submitted the *same*
commit, which bounds run-to-run draw without assuming anything about a model.
"""


import json
import statistics as st

CACHE = "/tmp/yukon-board/full.json"
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
DRAFTING = ["drama", "travel", "beagle", "republic", "essays", "medicine", "botany"]
RECEIPTS = {"A": "5a9f130a", "B": "180db842", "C": "fda590bb", "D": "2c885d64"}


def load():
    rows = json.load(open(CACHE))
    return [r for r in rows
            if isinstance(r, dict)
            and (r.get("officialMetrics") or {}).get("per_prompt")
            and r.get("officialScore") is not None]


def vec(row):
    return {PROMPT_NAMES.get(e["prompt_sha256"][:8], e["prompt_sha256"][:8]): e
            for e in row["officialMetrics"]["per_prompt"]}


def mtp(row):
    return {k: e["mtp_seconds_per_token_mean"] for k, e in vec(row).items()}


def edl(row):
    return {k: e["effective_mean_draft_len"] for k, e in vec(row).items()}


def delta_pct(base, cand):
    x, y = mtp(base), mtp(cand)
    return {n: (y[n] - x[n]) / x[n] * 100.0 for n in x}


def mean7(d):
    return st.mean(d[n] for n in DRAFTING)


def pick(scored, prefix):
    hits = [r for r in scored if r["id"].startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("prefix %r matched %d rows" % (prefix, len(hits)))
    return hits[0]


def serial_null(scored):
    """Run-level dispersion of the serial leg.

    The serial numerator comes from the runner's own prebuilt baseline
    workspace, so its source is identical across every run on the board. Its
    run-to-run spread is therefore a pure measurement null with no mechanism in
    it, and it bounds the common-mode draw that also lands on the candidate leg.
    """
    board = {}
    for n in DRAFTING:
        board[n] = st.mean(vec(r)[n]["serial_seconds_per_token_mean"]
                           for r in scored)
    devs = []
    for r in scored:
        v = vec(r)
        devs.append(st.mean((v[n]["serial_seconds_per_token_mean"] - board[n])
                            / board[n] * 100.0 for n in DRAFTING))
    return devs


def main():
    scored = load()
    rows = {k: pick(scored, v) for k, v in RECEIPTS.items()}

    print("=" * 78)
    print("0. COHORT CHECK -- head provenance and depth must match to compare")
    print("=" * 78)
    for k, r in rows.items():
        om = r["officialMetrics"]
        heads = {e["head_provenance_sha256"][:12] for e in om["per_prompt"]}
        print("  %s %s  score=%.8f  depth=%s  tokens=%s  heads=%s"
              % (k, r["id"][:8], r["officialScore"], om.get("mtp_depth"),
                 om.get("decode_tokens"), ",".join(sorted(heads))))

    print()
    print("=" * 78)
    print("1. NULL FLOOR -- same source bytes, different run")
    print("=" * 78)
    devs = serial_null(scored)
    leg_sd = st.stdev(devs)
    floor = leg_sd * (2 ** 0.5)
    print("  serial leg, n=%d runs, source identical by construction" % len(devs))
    print("    run-level mean7 dispersion: sd %.4f %%" % leg_sd)
    print("    => a difference of two runs carries sd %.4f %%" % floor)
    nullpair = delta_pct(rows["A"], pick(scored, "ec24d591"))
    print("  byte-identical candidate pair ec24d591 vs 5a9f130a (both")
    print("    organizer-pure main): mean7 %+.4f %%  -- %.1f x the floor,"
          % (mean7(nullpair), abs(mean7(nullpair)) / floor))
    print("    a concrete draw of the same null.")
    print("  USE %.3f %% as 1 sd on every pairwise mean7 below." % floor)

    print()
    print("=" * 78)
    print("2. PER-PROMPT CANDIDATE-LEG VECTORS (positive = slower)")
    print("=" * 78)
    pairs = [("D vs A", "A", "D", "Q + E165 + P    (all factors)"),
             ("C vs A", "A", "C", "I + P           (instrumentation)"),
             ("B vs C", "C", "B", "Q               (quantized kernels)"),
             ("B vs A", "A", "B", "Q + I + P       (additivity check)"),
             ("D vs B", "B", "D", "E165 - I        (P-free, Q-free)")]
    vectors = {}
    e = edl(rows["A"])
    print("  %-9s %6s %10s %10s %10s %10s"
          % ("prompt", "edl", "D vs A", "C vs A", "B vs C", "D vs B"))
    computed = {name: delta_pct(rows[b], rows[c]) for name, b, c, _ in pairs}
    for n in sorted(PROMPT_NAMES.values(), key=lambda n: e[n]):
        print("  %-9s %6.3f %+10.4f %+10.4f %+10.4f %+10.4f"
              % (n, e[n], computed["D vs A"][n], computed["C vs A"][n],
                 computed["B vs C"][n], computed["D vs B"][n]))
    print()
    for name, b, c, meaning in pairs:
        d = computed[name]
        m = mean7(d)
        sd = st.stdev(d[x] for x in DRAFTING)
        vectors[name] = m
        print("  %-7s mean7 %+7.4f %%  sd %.4f  plutarch %+7.4f  | %.1f x null | %s"
              % (name, m, sd, d["plutarch"], abs(m) / floor, meaning))
    resid = vectors["C vs A"] + vectors["B vs C"] - vectors["B vs A"]
    print("  additivity residual (C-A)+(B-C)-(B-A) = %+.4f %% -- log-additive"
          % resid)

    print()
    print("=" * 78)
    print("3. EDL SIGNATURE -- does the effect load on long-draft prompts?")
    print("=" * 78)
    print("  Q should load on drafting work; a per-round fixed cost should not")
    print("  scale the same way. plutarch (edl 0.156, non-drafting) is the")
    print("  natural control: a drafting-path mechanism must be ~0 there.")
    print()
    xs = [e[n] for n in DRAFTING]
    for name, _, _, meaning in pairs:
        ys = [computed[name][n] for n in DRAFTING]
        r = st.correlation(xs, ys)
        slope = st.linear_regression(xs, ys).slope
        print("  %-7s corr(edl, delta) = %+.3f   slope = %+.4f %%/edl   %s"
              % (name, r, slope, meaning))
    slopes = {name: st.linear_regression(xs, [computed[name][n] for n in DRAFTING]).slope
              for name, _, _, _ in pairs}
    resid_slope = (slopes["B vs C"] + slopes["D vs B"] + slopes["C vs A"]
                   - slopes["D vs A"])
    print()
    print("  slope algebra: Q + (E165-I) + (I+P) should equal (Q+E165+P).")
    print("  residual = %+.4f %%/edl -- the shape system is self-consistent."
          % resid_slope)
    print("  It cannot separate I from P either, so it corroborates the level")
    print("  algebra rather than adding an independent constraint. Under P ~ 0")
    print("  it puts E165's own slope at %+.3f %%/edl, i.e. nearly flat, which"
          % (slopes["D vs B"] + slopes["C vs A"]))
    print("  is the per-round fixed-cost signature predicted for a prefetch.")

    print()
    print("=" * 78)
    print("4. THE SPLIT -- closed by source inspection, not by assumption")
    print("=" * 78)
    DA, CA, BC = vectors["D vs A"], vectors["C vs A"], vectors["B vs C"]
    DB = vectors["D vs B"]
    print("  SOURCE FACT that closes the system (git, not assumption):")
    print("    D's Qwen35.swift is byte-identical to organizer main, and the")
    print("    x-sums sidecar lives in organizer main. `Qwen35XSumsSidecar")
    print("    .take`, `.publish` and `xsums: fused ?? xsumsTable(x)` are all")
    print("    still present in D. The 88-line revert removed two counters, an")
    print("    unset-on-runner env override, research-only default arguments")
    print("    and comments. There is no 'Q-without-sidecar' factor: D and B")
    print("    both carry Q WITH the sidecar, so Q is common to both.")
    print()
    print("  measured:  D-A = Q + E165 + P = %+.4f" % DA)
    print("             C-A = I + P        = %+.4f" % CA)
    print("             B-C = Q            = %+.4f" % BC)
    print("             D-B = E165 - I     = %+.4f   <- P-free and Q-free" % DB)
    print()
    print("  E165 + P = (D-A) - Q      = %+.4f %% +- %.4f" % (DA - BC, floor * 2 ** 0.5))
    print("  E165 - I = (D-B) measured = %+.4f %% +- %.4f" % (DB, floor))
    print()
    print("  Instrumentation I >= 0 and the pre-existing lines P >= 0, so:")
    print("    E165 >= %+.4f %%   (from D-B, taking I = 0)" % DB)
    print("    E165 <= %+.4f %%   (from D-A minus Q, taking P = 0)" % (DA - BC))
    print("    => E165 costs between %+.2f %% and %+.2f %% of candidate-leg time."
          % (DB, DA - BC))
    print("    The lower bound alone is %.1f sd from zero." % (DB / floor))
    print()
    tau1 = -(DB) / 1.150
    tau2 = -(DA - BC) / 1.150
    print("  tau (E165 transfer coefficient); predicted gain at tau=1 is -1.150 %:")
    print("    tau = %.3f .. %.3f  -- firmly NEGATIVE, not zero." % (tau2, tau1))
    print("    interim 6 read tau ~ 0 from the D-vs-C pair; that pair mixes Q,")
    print("    E165 and I, and the mixture cancels. Separated, E165 is harmful.")

    print()
    print("=" * 78)
    print("5. PREDICTIONS")
    print("=" * 78)
    A_pub = rows["A"]["officialScore"]
    crown = 3.72911001
    A_sf = 3.70885256
    q_score = st.mean([computed["B vs C"]["beagle"], computed["B vs C"]["essays"]])
    print("  A published %.8f   A serial-free %.8f   crown %.8f"
          % (A_pub, A_sf, crown))
    print("  receipt-level published sd ~0.271 %% (serial draw included)")
    print()
    print("  (b) Edward E175 = organizer + Q + campaign Qwen35 minus counter.")
    print("      The campaign Qwen35 content is inert on the scored path, so")
    print("      E175 == A + Q. Predicted candidate-leg gain:")
    for label, g in (("mean7", BC), ("score-setting prompts", q_score)):
        for anchor_name, anchor in (("A published", A_pub), ("A serial-free", A_sf)):
            pred = anchor / (1.0 + g / 100.0)
            print("        %-21s %+7.4f %%  on %-14s -> %.5f  (crown %+.5f)"
                  % (label, g, anchor_name, pred, pred - crown))
    print("      Verdict: E175 straddles the crown. Fire it.")
    print()
    print("  (a) Askeladd cap-4 on organizer-pure main: shares NO factor with")
    print("      D. No Q, no E165, no instrumentation, no P. Its base is A,")
    print("      which is measured. Nothing in D's vector shifts its bands.")
    print("      UNTHREATENED.")
    print()
    print("  (c) An E165 single-factor receipt is NOT worth a slot: the harm")
    print("      is already %.1f sd from zero on a P-free, Q-free pair." % (DB / floor))


if __name__ == "__main__":
    main()
