#!/usr/bin/env python3
"""E75 rung D: the within-session 2x2 over {kernel table} x {depth price}.

    python3 research/e75_rungD_analysis.py \
        --leg ours-ship:e75-rD-d1 --leg ours-pbfit:e75-rD-d2 \
        --leg crown-ship:e75-rD-d3 --leg crown-pbfit:e75-rD-d4 \
        --leg crown-pbfit:e75-rD-d5 --leg crown-ship:e75-rD-d6 \
        --leg ours-pbfit:e75-rD-d7 --leg ours-ship:e75-rD-d8

Every cell appears twice in a mirrored palindrome, so the within-cell spread is
a measured session null rather than an assumed one, and monotone thermal drift
cancels to first order.

Three quantities come out of the four cells, all inside one thermal session:

    main effect of table  = mean(C-*) - mean(O-*)
    main effect of pbfit  = mean(*-pbfit) - mean(*-ship)
    interaction           = (C-pbfit - C-ship) - (O-pbfit - O-ship)

`O-pbfit - O-ship` is a built-in positive control with a known answer from E68,
so the design cannot silently succeed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import statistics as st

REPO = pathlib.Path(__file__).resolve().parent.parent
PREREG = REPO / "research/e75-artifacts/e75-rungD-prereg.json"
TOKENS = 512

E68_PBFIT_EFFECT_PCT = -3.500
E68_SESSION_NULL_PCT = 0.143
RANKED_TABLE_EFFECT_PCT = -0.298
CELLS = ("ours-ship", "ours-pbfit", "crown-ship", "crown-pbfit")
SHORT = {"ours-ship": "O-ship", "ours-pbfit": "O-pbfit",
         "crown-ship": "C-ship", "crown-pbfit": "C-pbfit"}


def load(runs_dir, cell, tag):
    base = pathlib.Path(runs_dir) / tag
    out = json.loads((base / "reports/02-mtp-verify-output.json").read_text())
    tokens = out["emitted_tokens"]
    meta = dict(line.strip().split("=", 1)
                for line in (base / "meta.txt").read_text().splitlines()
                if "=" in line)
    return {
        "cell": cell,
        "tag": tag,
        "score": json.loads((base / "score.json").read_text())["metrics"],
        "timed": json.loads((base / "reports/04-mtp-timed.json").read_text()),
        "meta": meta,
        "stream_sha256": hashlib.sha256(
            ",".join(str(int(t)) for t in tokens).encode()).hexdigest(),
        "emitted_token_count": len(tokens),
    }


def widths(leg):
    return [d + 1 for d in leg["timed"]["effective_draft_lengths"]]


def histogram(leg):
    return collections.Counter(widths(leg))


def per_width_round_seconds(legs):
    """Round latency grouped by realised verify width, pooled over legs."""
    acc = collections.defaultdict(list)
    for leg in legs:
        for w, s in zip(widths(leg), leg["timed"]["block_request_seconds"]):
            acc[w].append(s)
    return acc


def pct(new, old):
    return (new - old) / old * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", action="append", required=True,
                    help="CELL:TAG, optionally CELL:TAG:discard")
    ap.add_argument("--runs-dir", default=str(REPO / ".mlxfast-private/e75-e2e/runs"))
    args = ap.parse_args()

    legs, discarded = [], []
    for spec in args.leg:
        parts = spec.split(":")
        cell, tag = parts[0], parts[1]
        (discarded if len(parts) > 2 and parts[2] == "discard" else legs).append(
            load(args.runs_dir, cell, tag))

    print("=" * 86)
    print("E75 RUNG D  harness=local  2x2 {kernel table} x {depth price}, "
          "mirrored palindrome")
    print("=" * 86)

    print("\nper-leg provenance and exactness")
    hdr = ("  %-14s %-12s %10s %8s %6s %6s %5s %6s  %-16s" %
           ("tag", "cell", "s/token", "ratio", "in C", "out C", "div", "match",
            "stream sha256"))
    print(hdr)
    for leg in legs + discarded:
        m, meta = leg["score"], leg["meta"]
        print("  %-14s %-12s %10.7f %8.5f %6s %6s %5d %6s  %-16s%s" % (
            leg["tag"], leg["cell"], m["mtp_seconds_per_token"],
            m["mtp_decode_speedup"], meta.get("gpu_temp_entry_c", "?"),
            meta.get("gpu_temp_exit_c", "?"), m["residual_divergence_count"],
            m["all_tokens_matched"], leg["stream_sha256"][:16],
            "  DISCARDED" if leg in discarded else ""))

    print("\ngate flags, verbatim, every timed leg")
    for leg in legs:
        meta = leg["meta"]
        print("  %-14s cool_gate_passed_real_gate=%s gate_qualified_for_timing=%s "
              "stale_metallib_warnings=%s parity_all_ok=%s rows %d/%d" % (
                  leg["tag"], meta.get("cool_gate_passed_real_gate"),
                  meta.get("gate_qualified_for_timing"),
                  meta.get("stale_metallib_warnings"),
                  leg["timed"]["parity_all_ok"],
                  leg["timed"]["reference_checked_row_total"],
                  leg["timed"]["declared_rows_total"]))

    print("\nbinary witness matrix, read back from the leg that ran")
    seen = {}
    for leg in legs:
        seen.setdefault(leg["cell"], leg["meta"])
    for cell in CELLS:
        meta = seen.get(cell)
        if meta:
            print("  %-12s __text %s  __cstring %s" % (
                cell, meta.get("worker_text_sha256", "?")[:16],
                meta.get("worker_cstring_sha256", "?")[:16]))
    texts = {c: seen[c].get("worker_text_sha256") for c in seen}
    cstrs = {c: seen[c].get("worker_cstring_sha256") for c in seen}
    print("  distinct __text %d (want 2)   distinct __cstring %d (want 2)   "
          "distinct pairs %d (want 4)" % (
              len(set(texts.values())), len(set(cstrs.values())),
              len(set(zip(texts.values(), cstrs.values())))))

    by_cell = collections.defaultdict(list)
    for leg in legs:
        by_cell[leg["cell"]].append(leg)

    print("\nCELL MEANS, primary quantity is absolute candidate MTP seconds "
          "per token")
    print("  %-12s %3s %12s %10s %10s %9s" %
          ("cell", "n", "s/token", "spread %", "ratio", "decode s"))
    mean_spt, mean_ratio, spreads = {}, {}, []
    for cell in CELLS:
        cl = by_cell.get(cell) or []
        if not cl:
            continue
        spt = [l["score"]["mtp_seconds_per_token"] for l in cl]
        ratio = [l["score"]["mtp_decode_speedup"] for l in cl]
        dec = [l["timed"]["decode_seconds"] for l in cl]
        mean_spt[cell] = st.fmean(spt)
        mean_ratio[cell] = st.fmean(ratio)
        spread = (max(spt) - min(spt)) / st.fmean(spt) * 100 if len(spt) > 1 else 0.0
        if len(spt) > 1:
            spreads.append(spread)
        print("  %-12s %3d %12.9f %9.3f%% %10.5f %9.3f" %
              (cell, len(cl), mean_spt[cell], spread, mean_ratio[cell], st.fmean(dec)))

    session_null = max(spreads) if spreads else float("nan")
    print("\n  MEASURED SESSION NULL, worst within-cell spread : %.3f %%"
          "   (E68 session null %.3f %%)" % (session_null, E68_SESSION_NULL_PCT))

    if len(mean_spt) == 4:
        o_ship, o_pbfit = mean_spt["ours-ship"], mean_spt["ours-pbfit"]
        c_ship, c_pbfit = mean_spt["crown-ship"], mean_spt["crown-pbfit"]
        pbfit_ours = pct(o_pbfit, o_ship)
        pbfit_crown = pct(c_pbfit, c_ship)
        table_ship = pct(c_ship, o_ship)
        table_pbfit = pct(c_pbfit, o_pbfit)
        main_table = pct((c_ship + c_pbfit) / 2, (o_ship + o_pbfit) / 2)
        main_pbfit = pct((o_pbfit + c_pbfit) / 2, (o_ship + c_ship) / 2)

        print("\nTHE THREE DERIVED QUANTITIES, each against the session null "
              "of %.3f %%" % session_null)
        print("  main effect of kernel table  %+8.3f %%" % main_table)
        print("  main effect of pbfit         %+8.3f %%" % main_pbfit)
        print("  INTERACTION                  %+8.3f pp" % (pbfit_crown - pbfit_ours))

        print("\nSIMPLE EFFECTS")
        print("  pbfit on OUR table    %+8.3f %%   POSITIVE CONTROL, "
              "E68 measured %+.3f %%" % (pbfit_ours, E68_PBFIT_EFFECT_PCT))
        print("  pbfit on CROWN table  %+8.3f %%" % pbfit_crown)
        print("  table at ship         %+8.3f %%" % table_ship)
        print("  table at pbfit        %+8.3f %%" % table_pbfit)

        control_error = pbfit_ours - E68_PBFIT_EFFECT_PCT
        print("\n  positive control error %+.3f pp against E68  -> %s"
              % (control_error,
                 "REPRODUCED" if abs(control_error) <= 1.0 else
                 "FAILED, the session is not comparable to E68"))

        print("\nTHE HARNESS PAIR, one identical eight-line dispatch-table diff")
        print("  %-10s %-56s %10s" % ("harness", "measurement", "value"))
        print("  %-10s %-56s %+9.3f %%" %
              ("ranked", "receipt 9b241879, plutarch-corrected scoring mean",
               RANKED_TABLE_EFFECT_PCT))
        print("  %-10s %-56s %+9.3f %%" %
              ("local", "this session, C-ship against O-ship, n=%d per cell"
               % len(by_cell["ours-ship"]), table_ship))
        print("  divergence %+.3f points. NOT converted; the two harnesses are "
              "reported side by side." % (table_ship - RANKED_TABLE_EFFECT_PCT))

        if PREREG.exists():
            pre = json.loads(PREREG.read_text())
            print("\nPREDICTED vs MEASURED, prediction committed before the session")
            print("  %-10s %12s %12s %9s" %
                  ("cell", "predicted", "measured", "error %"))
            for cell in CELLS:
                p = pre["predicted_candidate_mtp_seconds_per_token"][SHORT[cell]]
                got = mean_spt[cell]
                print("  %-10s %12.9f %12.9f %+8.2f%%" %
                      (SHORT[cell], p, got, pct(got, p)))
            print("  %-22s %10s %10s %9s" % ("effect", "predicted", "measured", "error pp"))
            for name, predicted, measured in (
                    ("pbfit on ours", pre["predicted_effects_pct"]["pbfit_on_ours"], pbfit_ours),
                    ("pbfit on crown", pre["predicted_effects_pct"]["pbfit_on_crown"], pbfit_crown),
                    ("table at ship", pre["predicted_effects_pct"]["table_at_ship"], table_ship),
                    ("interaction", pre["predicted_effects_pct"]["interaction_pp"],
                     pbfit_crown - pbfit_ours)):
                sign = "same" if (predicted >= 0) == (measured >= 0) else "OPPOSITE"
                print("  %-22s %+9.3f %+10.3f %+9.3f  sign %s" %
                      (name, predicted, measured, measured - predicted, sign))

    print("\nVERIFY-WIDTH HISTOGRAM, all four cells")
    all_widths = sorted({w for leg in legs for w in histogram(leg)})
    print("  %-12s %s  %6s %6s" % ("cell", "".join("%6d" % w for w in all_widths),
                                   "rounds", "rows"))
    hists = {}
    for cell in CELLS:
        cl = by_cell.get(cell) or []
        if not cl:
            continue
        pooled = collections.Counter()
        for leg in cl:
            pooled.update(histogram(leg))
        per_leg = {w: pooled[w] / len(cl) for w in all_widths}
        hists[cell] = per_leg
        rows = st.fmean([l["timed"]["declared_rows_total"] for l in cl])
        print("  %-12s %s  %6.1f %6.1f" %
              (cell, "".join("%6.1f" % per_leg[w] for w in all_widths),
               sum(per_leg.values()), rows))

    if "ours-ship" in hists and "crown-ship" in hists:
        diff = max(abs(hists["crown-ship"][w] - hists["ours-ship"][w])
                   for w in all_widths)
        print("\n  C-ship against O-ship, largest per-width round difference "
              ": %.1f rounds" % diff)
        print("  -> %s" % (
            "same histogram: the whole local table effect is per-cell cost, "
            "no schedule reaction, so the pair is a clean single-mechanism "
            "calibration" if diff <= 2.0 else
            "the histograms differ: the table also changes the schedule, so "
            "the main effect is not a single mechanism"))

    print("\nFIXED-WIDTH ROUND LATENCY, the predictor's assumption, per table")
    print("  %-6s %3s %11s %11s %9s" % ("table", "M", "ship ms", "pbfit ms", "delta %"))
    worst = {}
    for table in ("ours", "crown"):
        ship_legs = by_cell.get("%s-ship" % table) or []
        pbfit_legs = by_cell.get("%s-pbfit" % table) or []
        if not ship_legs or not pbfit_legs:
            continue
        a = per_width_round_seconds(ship_legs)
        b = per_width_round_seconds(pbfit_legs)
        worst[table] = 0.0
        for w in sorted(set(a) & set(b)):
            if len(a[w]) < 3 or len(b[w]) < 3:
                continue
            ma, mb = st.median(a[w]) * 1000, st.median(b[w]) * 1000
            d = pct(mb, ma)
            worst[table] = max(worst[table], abs(d))
            print("  %-6s %3d %11.3f %11.3f %+8.2f%%" % (table, w, ma, mb, d))
        print("  %-6s worst |delta| at a shared width with n>=3 : %.2f %%  -> %s"
              % (table, worst[table],
                 "arm-independent, predictor valid" if worst[table] <= 1.0 else
                 "NOT arm-independent: the histogram is not the only channel "
                 "and the prediction is void, not wrong"))

    digests = {leg["stream_sha256"] for leg in legs}
    print("\nEXACTNESS ACROSS THE SESSION")
    print("  distinct emitted-stream digests : %d" % len(digests))
    for d in sorted(digests):
        tags = [l["tag"] for l in legs if l["stream_sha256"] == d]
        cells = sorted({l["cell"] for l in legs if l["stream_sha256"] == d})
        print("    %s  %s  %s" % (d[:16], ",".join(cells), " ".join(tags)))
    print("  -> %s" % ("one stream on every cell: neither factor changes the "
                       "tokens" if len(digests) == 1 else
                       "MORE THAN ONE STREAM: a factor changed the emitted "
                       "tokens and that is invalid, not slow"))


if __name__ == "__main__":
    main()
