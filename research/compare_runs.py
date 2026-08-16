#!/usr/bin/env python3
"""Side-by-side comparison of traced local-iterate runs (schedule + fidelity)."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=analysis.json")
    args = ap.parse_args()
    runs = [(s.split("=", 1)[0], json.load(open(s.split("=", 1)[1]))) for s in args.runs]

    print("=== ROW GATE (fidelity) ===")
    print("| run | rows compared | unmatched | value mismatch | id mismatch | widths seen | verdict |")
    print("|:--|---:|---:|---:|---:|:--|:--|")
    for label, A in runs:
        rg = A["row_gate"]
        pw = rg.get("per_width", {})
        widths = ", ".join("w%s:%s" % (w, pw[w].get("rows", "?")) for w in sorted(pw, key=int))
        allok = all(pw[w].get("bit_exact") for w in pw)
        print("| %s | %s | %s | %s | %s | %s | %s |" % (
            label, rg["compared_rows"], rg["unmatched_positions"], rg["value_mismatches"],
            rg.get("id_mismatches", 0), widths,
            "ALL BIT-EXACT" if allok else "**MISMATCH**"))

    print("\n=== SCHEDULE OCCUPANCY: fraction of drafting rounds at each depth ===")
    hs = [(l, A["phases"]["mtp"]["depth_histogram"]) for l, A in runs]
    alld = sorted({int(k) for _, h in hs for k in h if h[k]})
    print("| depth | " + " | ".join("%s n | %s occ" % (l, l) for l, _ in hs) + " |")
    print("|---:|" + "---:|" * (2 * len(hs)))
    for d in alld:
        cells = []
        for _, h in hs:
            n = sum(h.values())
            c = h.get(str(d), 0)
            cells += ["%d" % c, "%.1f%%" % (100.0 * c / n)]
        print("| %d | %s |" % (d, " | ".join(cells)))
    print("| **total rounds** | " + " | ".join("%d |" % sum(h.values()) for _, h in hs) + " |")

    print("\n=== THROUGHPUT / GATE COUNTERS ===")
    keys = ["rounds", "mean_depth", "drafted_tokens", "accepted_tokens", "rejected_tokens",
            "accept_rate", "reject_round_rate", "accepted_tokens_per_round",
            "tokens_per_round", "rounds_per_token", "mean_round_us", "mean_eval_wall_us",
            "cap_histogram", "deep_gate_open_rate"]
    print("| metric | " + " | ".join(l for l, _ in runs) + " |")
    print("|:--|" + "---:|" * len(runs))
    for k in keys:
        vals = []
        for _, A in runs:
            v = A["phases"]["mtp"].get(k)
            vals.append("%.4f" % v if isinstance(v, float) else str(v))
        print("| %s | %s |" % (k, " | ".join(vals)))

    print("\n=== SCORE ===")
    sk = ["serial_seconds_per_token", "mtp_seconds_per_token", "mtp_decode_speedup",
          "accepted_draft_rate", "effective_mean_draft_len", "all_tokens_matched",
          "residual_divergence_count", "public_drift_tripwire_passed"]
    print("| metric | " + " | ".join(l for l, _ in runs) + " |")
    print("|:--|" + "---:|" * len(runs))
    for k in sk:
        print("| %s | %s |" % (k, " | ".join(str(A["metrics"].get(k)) for _, A in runs)))

    print("\n=== EMA[4] CONDITIONED ON THE STREAK GATE ===")
    for label, A in runs:
        m = A["phases"]["mtp"]
        print("  %-16s open  %s" % (label, m.get("ema4_when_gate_open")))
        print("  %-16s close %s" % (label, m.get("ema4_when_gate_closed")))


if __name__ == "__main__":
    main()
