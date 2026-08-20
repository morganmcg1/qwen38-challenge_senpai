#!/usr/bin/env python3
"""Reproduce every rung-3 number in research/e68-results.md.

Reads the per-leg artifacts written by research/e68_run_leg.sh and prints the
arm table, the null bar, the additive position estimator, the realised
verify-width histograms, the pooled in-situ latency curve, the arm-independence
check, the marginal token price ladder, and the calibrated ranked projection.

    python3 research/e68_rung3_analysis.py \
        --leg ship:e68-r3c-warmup:discard --leg ship:e68-r3c-c1 \
        --leg pb5:e68-r3c-c2 --leg pb7:e68-r3c-c3 --leg pbfit:e68-r3c-c4 \
        --leg pbfit:e68-r3c-c5 --leg pb7:e68-r3c-c6 --leg pb5:e68-r3c-c7 \
        --leg ship:e68-r3c-c8
"""

import argparse
import hashlib
import json
import math
import os
import statistics as st

# Rung-1 whole-table isolated QMV cost, milliseconds, indexed by verify width.
# Source: research/e68-artifacts/e68-rung1.json, median of three shipped legs.
ISOLATED_MS = {1: 60.372, 2: 65.377, 3: 72.128, 4: 82.163, 5: 95.568,
               6: 122.876, 7: 138.314, 8: 148.841, 9: 163.621, 10: 271.147}

# Per-draft acceptance, ledger 184(B). The local figure is advisor ruling 2.
OPERATING_POINTS = [("local fixture", 0.8808), ("ranked beagle", 0.8351),
                    ("ranked medicine", 0.8750), ("ranked republic", 0.9019)]

SHIP_MARGINAL = [0.18] * 8


def load(runs_dir, tag):
    base = os.path.join(runs_dir, tag)
    return {
        "tag": tag,
        "score": json.load(open(os.path.join(base, "score.json")))["metrics"],
        "timed": json.load(open(os.path.join(base, "reports/04-mtp-timed.json"))),
        "out": json.load(open(os.path.join(base, "reports/02-mtp-verify-output.json"))),
        "arm_cfg": json.load(open(os.path.join(base, "arm.json"))),
        "meta": dict(l.strip().split("=", 1)
                     for l in open(os.path.join(base, "meta.txt")) if "=" in l),
    }


def widths(leg):
    return [d + 1 for d in leg["timed"]["effective_draft_lengths"]]


def least_squares(X, y):
    n, k = len(X), len(X[0])
    A = [[sum(X[i][p] * X[i][q] for i in range(n)) for q in range(k)] +
         [sum(X[i][p] * y[i] for i in range(n))] for p in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(A[r][c]))
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]
        A[c] = [v / pv for v in A[c]]
        for r in range(k):
            if r != c and A[r][c]:
                f = A[r][c]
                A[r] = [A[r][j] - f * A[c][j] for j in range(k + 1)]
    return [A[i][k] for i in range(k)]


def walk(marginal, p, cap=8):
    """The shipped greedy walk, generalised to a per-position price."""
    reach, expected, cumulative, depth = 1.0, 0.0, 1.0, 0
    while depth < cap:
        reach *= p
        if not reach > marginal[depth] * (1.0 + expected) / cumulative:
            break
        expected += reach
        cumulative += marginal[depth]
        depth += 1
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", action="append", required=True,
                    metavar="ARM:TAG[:discard]")
    ap.add_argument("--runs-dir", default=".mlxfast-private/e68-e2e/runs")
    ap.add_argument("--out")
    args = ap.parse_args()

    legs = []
    for i, spec in enumerate(args.leg):
        parts = spec.split(":")
        arm, tag = parts[0], parts[1]
        leg = load(args.runs_dir, tag)
        leg.update(arm=arm, pos=i, discard=len(parts) > 2 and parts[2] == "discard")
        recorded = leg["arm_cfg"]["arm"]
        if recorded != arm:
            raise SystemExit(f"{tag}: arm.json says {recorded!r}, not {arm!r}")
        legs.append(leg)
    body = [L for L in legs if not L["discard"]]
    arms = sorted({L["arm"] for L in body})
    by = {a: [L for L in body if L["arm"] == a] for a in arms}
    report = {}

    print("== emitted token stream identity ==")
    digests = {}
    for L in legs:
        h = hashlib.sha256(json.dumps(L["out"]["emitted_tokens"]).encode()).hexdigest()
        digests.setdefault(h, []).append(L["tag"])
        assert L["score"]["all_tokens_matched"], L["tag"]
        assert L["score"]["residual_divergence_count"] == 0, L["tag"]
    for h, tags in digests.items():
        print("  %s  %d leg(s): %s" % (h[:16], len(tags), " ".join(tags)))
    identical = len(digests) == 1
    print("  identical across all legs: %s" % identical)
    report["emitted_identical"] = identical
    report["emitted_sha256"] = list(digests)

    print()
    print("== row ledger ==")
    print("  %-18s%-7s%8s%10s%10s%9s%9s%7s" %
          ("tag", "arm", "rounds", "declared", "checked", "accept", "reject", "div"))
    for L in legs:
        t = L["timed"]
        assert t["declared_rows_total"] == t["reference_checked_row_total"]
        assert (t["declared_rows_total"] ==
                t["accepted_draft_total"] + t["rejected_draft_total"] + t["round_count"])
        assert t["rejected_draft_total"] == t["rejected_rows_reference_checked"]
        assert t["parity_all_ok"] and t["uses_pinned_mtp_head"]
        print("  %-18s%-7s%8d%10d%10d%9d%9d%7d" %
              (L["tag"], L["arm"], t["round_count"], t["declared_rows_total"],
               t["reference_checked_row_total"], t["accepted_draft_total"],
               t["rejected_draft_total"], t["residual_divergence_count"]))
    print("  ledger closes and parity holds on every leg")

    print()
    print("== arm table ==")
    ship = st.mean([L["score"]["mtp_seconds_per_token"] for L in by["ship"]])
    print("  %-7s%3s%15s%10s%11s%10s%10s%10s" %
          ("arm", "n", "mtp s/tok", "spread%", "vs ship%", "serial", "ratio", "meanW"))
    spreads = {}
    report["arms"] = {}
    for a in arms:
        v = [L["score"]["mtp_seconds_per_token"] for L in by[a]]
        mu = st.mean(v)
        spreads[a] = (max(v) - min(v)) / min(v) * 100.0
        mw = st.mean([sum(widths(L)) / len(widths(L)) for L in by[a]])
        report["arms"][a] = dict(
            n=len(v), mtp=mu, spread_pct=spreads[a], vs_ship_pct=(mu - ship) / ship * 100.0,
            serial=st.mean([L["score"]["serial_seconds_per_token"] for L in by[a]]),
            ratio=st.mean([L["score"]["mtp_decode_speedup"] for L in by[a]]), mean_width=mw)
        r = report["arms"][a]
        print("  %-7s%3d%15.9f%10.3f%11.3f%10.6f%10.5f%10.4f" %
              (a, r["n"], mu, r["spread_pct"], r["vs_ship_pct"], r["serial"], r["ratio"], mw))
    null_bar = max(spreads.values())
    report["null_bar_pct"] = null_bar
    print("  null bar (largest same-arm spread) = %.3f %%" % null_bar)

    print()
    print("== additive estimator  mtp ~ arm + centred leg position ==")
    others = [a for a in arms if a != "ship"]
    mean_pos = st.mean([L["pos"] for L in body])
    X = [[1.0] + [1.0 if L["arm"] == a else 0.0 for a in others] + [L["pos"] - mean_pos]
         for L in body]
    y = [L["score"]["mtp_seconds_per_token"] for L in body]
    beta = least_squares(X, y)
    names = ["intercept"] + others + ["pos_slope"]
    report["estimator"] = dict(zip(names, beta))
    for nm, b in zip(names, beta):
        print("  %-12s %+.9f   (%+.3f %% of intercept)" % (nm, b, b / beta[0] * 100.0))
    resid = [y[i] - sum(X[i][p] * beta[p] for p in range(len(beta))) for i in range(len(body))]
    rms = math.sqrt(sum(r * r for r in resid) / len(resid))
    report["estimator"]["residual_rms"] = rms
    print("  residual rms %.9f   (%.3f %%)" % (rms, rms / beta[0] * 100.0))

    print()
    print("== realised verify-width histogram, first leg of each arm ==")
    ws = list(range(1, 10))
    print("  %-7s%8s" % ("arm", "rounds") + "".join("%6s" % ("W%d" % w) for w in ws) +
          "%9s%10s" % ("meanW", ">=7"))
    report["histogram"] = {}
    for a in arms:
        W = widths(by[a][0])
        h = [W.count(w) for w in ws]
        report["histogram"][a] = dict(zip(map(str, ws), h))
        print("  %-7s%8d" % (a, len(W)) + "".join("%6d" % v for v in h) +
              "%9.4f%9.1f%%" % (sum(W) / len(W),
                                100.0 * sum(v for w, v in zip(ws, h) if w >= 7) / len(W)))

    print()
    print("== pooled in-situ round latency by realised verify width ==")
    pool, per_arm = {}, {}
    for L in body:
        for w, t in zip(widths(L), L["timed"]["block_request_seconds"]):
            pool.setdefault(w, []).append(t * 1000.0)
            per_arm.setdefault((L["arm"], w), []).append(t * 1000.0)
    latency, prev = {}, None
    print("  %6s%7s%12s%12s%12s" % ("width", "n", "median ms", "step ms", "vs isolated"))
    for w in sorted(pool):
        latency[w] = st.median(pool[w])
        step = "" if prev is None else "%+.4f" % (latency[w] - latency[w - 1])
        ratio = ""
        if prev is not None and w in ISOLATED_MS and (w - 1) in ISOLATED_MS:
            iso = ISOLATED_MS[w] - ISOLATED_MS[w - 1]
            ratio = "%.3fx" % ((latency[w] - latency[w - 1]) / iso)
        print("  %6d%7d%12.4f%12s%12s" % (w, len(pool[w]), latency[w], step, ratio))
        prev = w
    report["insitu_latency_ms"] = latency

    print()
    print("== arm independence of round latency ==")
    print("  %6s" % "width" + "".join("%11s" % a for a in arms) + "%12s" % "max spread")
    for w in sorted(pool):
        cells, vals = [], []
        for a in arms:
            v = per_arm.get((a, w))
            if v and len(v) >= 2:
                vals.append(st.median(v))
                cells.append("%11.3f" % vals[-1])
            else:
                cells.append("%11s" % "-")
        tail = ("%11.2f%%" % ((max(vals) - min(vals)) / min(vals) * 100.0)) if len(vals) > 1 else "%12s" % "-"
        print("  %6d" % w + "".join(cells) + tail)

    print()
    print("== marginal token price ladder ==")
    ladder = sorted(((512.0 / st.mean([L["timed"]["round_count"] for L in by[a]]),
                      st.mean([L["timed"]["decode_seconds"] for L in by[a]]) * 1000.0 /
                      st.mean([L["timed"]["round_count"] for L in by[a]]), a) for a in arms))
    print("  %-7s%14s%12s%12s%22s" %
          ("arm", "tokens/round", "ms/round", "ms/token", "marginal ms/token"))
    report["ladder"] = []
    prev_pt = None
    for tok, msr, a in ladder:
        marg = "-" if prev_pt is None else "%.1f" % ((msr - prev_pt[1]) / (tok - prev_pt[0]))
        report["ladder"].append(dict(arm=a, tokens_per_round=tok, ms_per_round=msr,
                                     ms_per_token=msr / tok, marginal=marg))
        print("  %-7s%14.3f%12.2f%12.2f%22s" % (a, tok, msr, msr / tok, marg))
        prev_pt = (tok, msr)

    print()
    print("== flat-p projection, priced with the pooled in-situ latency ==")
    pbfit = next((L["arm_cfg"]["marginal"] for L in body if L["arm"] == "pbfit"), None)
    if pbfit is None:
        print("  no pbfit arm in this session; skipping")
    else:
        def price(marginal, p):
            d = walk(marginal, p)
            w = d + 1
            return d, w, latency[w], sum(p ** k for k in range(d + 1))

        raw = {}
        print("  %-18s%8s%7s%7s%11s%9s%11s" %
              ("point", "arm", "depth", "width", "round ms", "E[tok]", "ms/token"))
        for name, p in OPERATING_POINTS:
            for an, marginal in (("ship", SHIP_MARGINAL), ("pbfit", pbfit)):
                d, w, lat, toks = price(marginal, p)
                raw[(name, an)] = lat / toks
                print("  %-18s%8s%7d%7d%11.3f%9.4f%11.4f" %
                      (name, an, d, w, lat, toks, lat / toks))
        k = {}
        for a in ("ship", "pbfit"):
            measured = st.mean([L["timed"]["decode_seconds"] for L in by[a]])
            k[a] = measured / (raw[("local fixture", a)] * 512 / 1000.0)
        print()
        print("  local calibration factor:  ship %.4f   pbfit %.4f" % (k["ship"], k["pbfit"]))
        print("  %-18s%14s%14s" % ("point", "raw %", "calibrated %"))
        report["projection"] = {}
        for name, _ in OPERATING_POINTS:
            a, b = raw[(name, "ship")], raw[(name, "pbfit")]
            cal = (b * k["pbfit"] - a * k["ship"]) / (a * k["ship"]) * 100.0
            report["projection"][name] = dict(raw_pct=(b - a) / a * 100.0, calibrated_pct=cal)
            print("  %-18s%+14.3f%+14.3f" % (name, (b - a) / a * 100.0, cal))
        report["calibration"] = k

    print()
    print("== identity tuple ==")
    for key in ("base_sha", "branch_commit", "fixture_sha256", "head_run_dir_tree_sha256",
                "metallib_source_fingerprint", "tokens", "offered_depth",
                "cool_gate_passed_real_gate", "gate_qualified_for_timing"):
        vals = sorted({L["meta"][key] for L in legs})
        flag = "OK" if len(vals) == 1 else "VARIES"
        print("  %-30s %-7s %s" % (key, flag, vals[0] if len(vals) == 1 else vals))
    print("  %-30s %s" % ("scored_source_sha256",
                          {L["arm"]: L["meta"]["scored_source_sha256"][:12] for L in legs}))
    print("  %-30s %s" % ("worker_text_sha256",
                          {L["arm"]: L["meta"]["worker_text_sha256"][:12] for L in legs}))
    temps = [(float(L["meta"]["gpu_temp_entry_c"]), float(L["meta"]["gpu_temp_exit_c"]))
             for L in body]
    print("  entry C %.2f to %.2f (spread %.2f)   exit C %.2f to %.2f" %
          (min(t[0] for t in temps), max(t[0] for t in temps),
           max(t[0] for t in temps) - min(t[0] for t in temps),
           min(t[1] for t in temps), max(t[1] for t in temps)))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1, sort_keys=True)
        print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
