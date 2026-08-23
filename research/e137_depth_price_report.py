#!/usr/bin/env python3
"""Read the E137 depth-price ladder and price `pb6` against `ship`.

The headline is absolute candidate seconds per token. The serial leg is
reported beside it as a null diagnostic: a depth price cannot reach the serial
leg, so any serial movement is session noise and bounds the reading.

Each replicate is the palindrome ``pb6 ship ship pb6``, so the within-replicate
contrast of the two arm means cancels a monotone drift in leg index to first
order. The per-replicate contrasts are the unit of evidence, not the legs.
"""

import glob
import json
import os
import statistics
import sys


def read_meta(path):
    out = {}
    with open(path) as handle:
        for line in handle:
            if "=" in line:
                key, _, value = line.partition("=")
                out[key.strip()] = value.strip()
    return out


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "D1"
    legs = []
    for meta_path in sorted(glob.glob("research/out/e137%sk*/meta.txt" % label)):
        directory = os.path.dirname(meta_path)
        score_path = os.path.join(directory, "score.json")
        if not os.path.exists(score_path):
            continue
        meta = read_meta(meta_path)
        metrics = json.load(open(score_path))["metrics"]
        legs.append(
            {
                "tag": meta.get("tag", os.path.basename(directory)),
                "arm": meta.get("e137_arm", "?"),
                "replicate": int(meta.get("e137_replicate", 0)),
                "position": int(meta.get("e137_position", 0)),
                "mtp": metrics["mtp_seconds_per_token"],
                "serial": metrics["serial_seconds_per_token"],
                "edl": metrics["effective_mean_draft_len"],
                "matched": metrics["all_tokens_matched"],
                "divergences": metrics["residual_divergence_count"],
                "entry_c": float(meta.get("gpu_temp_entry_c", "nan")),
                "exit_c": float(meta.get("gpu_temp_exit_c", "nan")),
                "worker": meta.get("worker_sha256", "?"),
                "commit": meta.get("e137_session_commit", "?"),
                "gated": meta.get("gate_qualified_for_timing", "?"),
            }
        )

    if not legs:
        sys.exit("e137_depth_price_report: no legs found for label %s" % label)

    workers = {leg["worker"] for leg in legs}
    commits = {leg["commit"] for leg in legs}
    if len(workers) != 1 or len(commits) != 1:
        sys.exit(
            "e137_depth_price_report: session is not one binary; "
            "workers=%s commits=%s" % (workers, commits)
        )

    print("E137 depth-price ladder, label %s" % label)
    print("  commit %s" % commits.pop())
    print("  worker %s" % workers.pop())
    print("  legs   %d" % len(legs))
    print("  gate_qualified_for_timing %s" % sorted({l["gated"] for l in legs}))
    print()

    bad = [l for l in legs if not l["matched"] or l["divergences"]]
    if bad:
        print("!! EXACTNESS FAILURE on %d legs: %s" % (bad, [l["tag"] for l in bad]))
        print()

    print(
        "%-28s %4s %3s %3s %13s %13s %9s %7s %7s"
        % ("tag", "arm", "rep", "pos", "mtp s/tok", "serial s/tok", "edl",
           "in C", "out C")
    )
    for leg in sorted(legs, key=lambda l: (l["replicate"], l["position"])):
        print(
            "%-28s %4s %3d %3d %13.9f %13.9f %9.5f %7.2f %7.2f"
            % (
                leg["tag"], leg["arm"], leg["replicate"], leg["position"],
                leg["mtp"], leg["serial"], leg["edl"],
                leg["entry_c"], leg["exit_c"],
            )
        )
    print()

    for field, name in (("mtp", "candidate mtp s/tok"),
                        ("serial", "serial s/tok"),
                        ("edl", "effective_mean_draft_len")):
        print("== %s ==" % name)
        for arm in ("pb6", "ship"):
            vals = [l[field] for l in legs if l["arm"] == arm]
            if not vals:
                continue
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(
                "  %-5s n=%d mean=%.9f sd=%.9f (%.3f %%)"
                % (arm, len(vals), statistics.mean(vals), sd,
                   100.0 * sd / statistics.mean(vals))
            )
        print()

    print("== within-replicate contrast, ship against pb6 ==")
    print("   negative means ship is FASTER, which is the pb6-is-costly direction")
    contrasts = {}
    for field in ("mtp", "serial"):
        rows = []
        for rep in sorted({l["replicate"] for l in legs}):
            arms = {}
            for arm in ("pb6", "ship"):
                vals = [
                    l[field] for l in legs
                    if l["replicate"] == rep and l["arm"] == arm
                ]
                if vals:
                    arms[arm] = statistics.mean(vals)
            if len(arms) == 2:
                rows.append(100.0 * (arms["ship"] - arms["pb6"]) / arms["pb6"])
        contrasts[field] = rows
        if rows:
            sd = statistics.stdev(rows) if len(rows) > 1 else 0.0
            print(
                "  %-7s per-replicate %s"
                % (field, ["%+.4f %%" % r for r in rows])
            )
            print(
                "  %-7s mean %+.4f %%  sd %.4f %%  negative in %d of %d"
                % (field, statistics.mean(rows), sd,
                   sum(1 for r in rows if r < 0), len(rows))
            )
    print()

    if contrasts.get("mtp") and contrasts.get("serial"):
        m = statistics.mean(contrasts["mtp"])
        s = statistics.mean(contrasts["serial"])
        print("== reading ==")
        print("  candidate leg %+.4f %%, serial leg %+.4f %% (null diagnostic)" % (m, s))
        print("  drift-corrected candidate effect %+.4f %%" % (m - s))
        print()
        print("  This session is ungated. cool_gate_passed_real_gate is false on")
        print("  every leg, so this is directional evidence inside its own")
        print("  counterbalanced session and is not a gate-qualified result.")


if __name__ == "__main__":
    main()
