#!/usr/bin/env python3
"""E188 stage 2: analyse the commit-phase bisection legs and log them to W&B.

harness=local. One W&B run per timed leg, plus one roll-up run that carries the
arm comparison and the per-round regime profile.

Each leg run carries its full experiment identity tuple in `config` so a leg can
never be compared with a leg that differs in more than the bisected dimension,
and carries every phase median in `summary`. The per-round series is logged as a
step series so the round-3 regime transition is visible in the W&B UI.

    python3 research/e188_stage2_wandb.py --legs e188-head-a e188-pre165-a ...

`--dry-run` prints the same records without contacting W&B.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics as st
import subprocess

PHASES = ["commit_us", "readout_us", "upkeep_us", "eval_wall_us",
          "verify_build_us", "draft_build_us", "round_us"]

# The four distinct contents of the commit-phase window between 59b67f50 and
# the current base, from research/e188_commit_phase_bisect.py.
ARM_OF_REV = {
    "59b67f50": "e90-anchor",
    "7a427dfa": "e134-restore-v1",
    "806181de": "pre-e165",
    "HEAD": "post-e165",
}

# E90 (Edward, PR 92, ledger 234.5): 512 tokens, declared head, depth 8,
# ungated, ship arm, 78 rounds with 8 warmup skipped.
E90 = {"commit_us": 183.9, "readout_us": 14.8, "upkeep_us": 66.7,
       "eval_wall_us": 76434.2, "round_us": 165183.9}


def read_rounds(tag: str) -> list[dict]:
    path = pathlib.Path("research/out") / tag / "trace.txt"
    rounds = []
    for line in path.read_text().splitlines():
        if not line.startswith("mtp-trace:"):
            continue
        kv = dict(re.findall(r"(\w+)=([-\w.]+)", line))
        if "round" not in kv:
            continue
        try:
            kv["round"] = int(kv["round"])
        except ValueError:
            continue
        rounds.append(kv)
    rounds.sort(key=lambda r: r["round"])
    return rounds


def read_meta(tag: str) -> dict:
    path = pathlib.Path("research/out") / tag / "meta.txt"
    meta = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k] = v
    return meta


def summarise(tag: str) -> dict:
    rounds = read_rounds(tag)
    meta = read_meta(tag)
    rev = meta.get("e188_bisect_rev", "HEAD")
    body = rounds[1:]  # round 1 pays the cold caches and first shape warmups

    rec = {
        "tag": tag,
        "arm": ARM_OF_REV.get(rev, rev),
        "rev": rev,
        "blockSessionBlob": meta.get("e188_bisect_blob", "")[:12],
        "nRoundsTotal": len(rounds),
        "nRoundsScored": len(body),
        "rounds": rounds,
    }
    for ph in PHASES:
        vals = [float(r[ph]) for r in body if ph in r]
        if not vals:
            continue
        rec[ph] = st.median(vals)
        if len(vals) > 3:
            q = st.quantiles(vals, n=4)
            rec[ph + "_iqr"] = q[2] - q[0]
    return rec


def identity(tag: str, base_sha: str) -> dict:
    """The experiment identity tuple the program requires before timing."""
    meta = read_meta(tag)
    return {
        "harness": "local",
        "leg": tag,
        "bisectRev": meta.get("e188_bisect_rev", "HEAD"),
        "bisectedFile": "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
        "blockSessionBlob": meta.get("e188_bisect_blob", ""),
        "blockSessionBlobAtHead": meta.get("e188_bisect_head_blob", ""),
        "baseSha": meta.get("base_sha", base_sha),
        "workerSha256": meta.get("worker_sha256", ""),
        "host": meta.get("host", os.uname().nodename),
        "chip": meta.get("chip", ""),
        "memoryBytes": meta.get("memory_bytes", ""),
        "tokens": meta.get("tokens", ""),
        "localMode": meta.get("local_mode", ""),
        "headDir": pathlib.Path(meta.get("head_dir", "")).name,
        "sandbox": meta.get("sandbox", ""),
        "coolGate": meta.get("cool_gate", ""),
        "coolGatePassedRealGate": meta.get("cool_gate_passed_real_gate", ""),
        "gateQualifiedForTiming": meta.get("gate_qualified_for_timing", ""),
        "officialOrRankedScore": meta.get("official_or_ranked_score", ""),
        "gpuTempEntryC": meta.get("gpu_temp_entry_c", ""),
        "gpuTempExitC": meta.get("gpu_temp_exit_c", ""),
        "metallibSourceFingerprint": meta.get("metallib_source_fingerprint", ""),
        "startedUtc": meta.get("started", ""),
        "referenceSource": "candidate-generated local rows",
    }


# The periodic multi-millisecond commit spikes are the FINDING 482 ledger-hole
# artifact. They recur about every seven rounds at every depth, so they are
# excluded from the depth fit and reported separately.
SPIKE_US = 1500.0


def depth_profile(legs: list[dict]) -> dict:
    """Host-phase cost against the realized draft depth `d` in the same build.

    The trace reports the depth the scheduler actually chose, which ramps from
    4 to the cap of 7 while the acceptance EMAs warm up. That gives a
    within-leg, same-build, same-host contrast in `d` at no extra GPU cost.
    """
    by_depth: dict[int, list[dict]] = {}
    spikes = []
    for leg in legs:
        for r in leg["rounds"]:
            if "d" not in r or "commit_us" not in r:
                continue
            rec = {"leg": leg["tag"], "round": r["round"], "d": int(r["d"]),
                   "acc": int(r.get("acc", -1)),
                   "commit_us": float(r["commit_us"]),
                   "readout_us": float(r.get("readout_us", "nan")),
                   "upkeep_us": float(r.get("upkeep_us", "nan"))}
            if rec["commit_us"] >= SPIKE_US:
                spikes.append(rec)
            else:
                by_depth.setdefault(rec["d"], []).append(rec)

    rows = {}
    xs, ys = [], []
    for d in sorted(by_depth):
        v = by_depth[d]
        rows[d] = {"n": len(v),
                   "commit_us": st.median([x["commit_us"] for x in v]),
                   "readout_us": st.median([x["readout_us"] for x in v]),
                   "upkeep_us": st.median([x["upkeep_us"] for x in v])}
        xs += [d] * len(v)
        ys += [x["commit_us"] for x in v]

    fit = {}
    if len(set(xs)) > 1:
        mx, my = st.mean(xs), st.mean(ys)
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                 / sum((x - mx) ** 2 for x in xs))
        intercept = my - slope * mx
        fit = {"n": len(xs), "interceptUs": intercept, "slopeUsPerDraft": slope,
               "impliedDepthAtE90":
               (E90["commit_us"] - intercept) / slope if slope else None}
        for d in sorted(set(xs)):
            fit[f"predictedAtD{d}"] = intercept + slope * d
    return {"byDepth": rows, "fit": fit, "nSpikes": len(spikes),
            "spikeRounds": [(x["leg"], x["round"], x["commit_us"]) for x in spikes]}


def split_rounds(leg: dict) -> tuple[list[float], list[float]]:
    """Commit cost on full-acceptance rounds and on prefix-rejection rounds.

    `restoreAfterPrefixReject` is the only commit-window callee the bisection
    moves, and it runs ONLY when `acc < d`. A median over all rounds is
    dominated by full-acceptance rounds and cannot see it, so the arms must be
    compared on the rejection rounds separately.
    """
    full, reject = [], []
    for r in leg["rounds"]:
        if "acc" not in r or "d" not in r or "commit_us" not in r:
            continue
        rec = (float(r["commit_us"]), float(r.get("round_us", "nan")))
        (reject if int(r["acc"]) < int(r["d"]) else full).append(rec)
    return full[1:], reject  # round 1 is cold and is always a full-acceptance round


def arm_table(legs: list[dict]) -> dict:
    """Per-arm commit cost, split by acceptance outcome, in palindromic order."""
    arms: dict[str, dict[str, list[float]]] = {}
    for leg in legs:
        full, reject = split_rounds(leg)
        a = arms.setdefault(leg["arm"], {"legMedians": [], "full": [], "reject": []})
        a["legMedians"].append(leg["commit_us"])
        a["full"] += full
        a["reject"] += reject
    out = {}
    for arm, a in arms.items():
        out[arm] = {
            "legs": a["legMedians"],
            "mean": st.mean(a["legMedians"]),
            "spread": (max(a["legMedians"]) - min(a["legMedians"])
                       if len(a["legMedians"]) > 1 else 0.0),
            "nFull": len(a["full"]),
            "fullAcceptCommitUs": st.median([c for c, _ in a["full"]]),
            "fullAcceptRoundUs": st.median([r for _, r in a["full"]]),
            "nReject": len(a["reject"]),
            "rejectCommitUs": st.median([c for c, _ in a["reject"]]),
            "rejectRoundUs": st.median([r for _, r in a["reject"]]),
            "rejectCommitValues": sorted(c for c, _ in a["reject"]),
        }
        out[arm]["repairCostUs"] = (out[arm]["rejectCommitUs"]
                                    - out[arm]["fullAcceptCommitUs"])
        out[arm]["rejectRoundExcessUs"] = (out[arm]["rejectRoundUs"]
                                           - out[arm]["fullAcceptRoundUs"])
    return out


# The source bisection puts the only repair-path change at merge eec2c14b, so
# the two arms whose window content predates it are pooled against the two that
# follow it.
PRE_EEC = ("e90-anchor", "e134-restore-v1")
POST_EEC = ("pre-e165", "post-e165")


def rank_test(table: dict) -> dict:
    """Exact Mann-Whitney U on rejection-round commit cost across eec2c14b.

    The rejection cells are small (4 to 6 per arm) and the values are not
    normal, so a rank test is the honest statistic. U counts the pairs in which
    a post-merge round is slower than a pre-merge round.
    """
    pre = [v for a in PRE_EEC for v in table.get(a, {}).get("rejectCommitValues", [])]
    post = [v for a in POST_EEC for v in table.get(a, {}).get("rejectCommitValues", [])]
    if not pre or not post:
        return {}
    wins = sum(1 for b in post for a in pre if b > a)
    ties = sum(1 for b in post for a in pre if b == a)
    u = wins + 0.5 * ties
    n = len(pre) * len(post)
    # Normal approximation with a continuity correction; exact enumeration is
    # unnecessary because U sits at the extreme of its range.
    mu = n / 2
    sigma = (len(pre) * len(post) * (len(pre) + len(post) + 1) / 12) ** 0.5
    z = (u - mu - 0.5) / sigma if sigma else float("nan")
    return {"nPre": len(pre), "nPost": len(post), "U": u, "UMax": n,
            "medianPreUs": st.median(pre), "medianPostUs": st.median(post),
            "stepUs": st.median(post) - st.median(pre), "z": z}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", nargs="+", required=True)
    ap.add_argument("--group", default="qwen38-r1-e188-commit-phase-bisection")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()

    legs = [summarise(t) for t in args.legs]
    table = arm_table(legs)
    depth = depth_profile(legs)

    print(f"{'leg':22s} {'arm':18s} {'rev':10s} {'n':>3s} " +
          " ".join(f"{p.replace('_us',''):>11s}" for p in PHASES))
    for leg in legs:
        print(f"{leg['tag']:22s} {leg['arm']:18s} {leg['rev']:10s} "
              f"{leg['nRoundsScored']:3d} " +
              " ".join(f"{leg.get(p, float('nan')):11.1f}" for p in PHASES))

    print("\narm commit_us, split by acceptance outcome "
          "(palindromic order, harness=local):")
    for arm, rec in table.items():
        print(f"  {arm:18s} legMed mean={rec['mean']:6.1f} spread={rec['spread']:5.1f}"
              f" | full n={rec['nFull']:2d} commit={rec['fullAcceptCommitUs']:6.1f}"
              f" round={rec['fullAcceptRoundUs']:9.1f}"
              f" | rej n={rec['nReject']:2d} commit={rec['rejectCommitUs']:7.1f}"
              f" round={rec['rejectRoundUs']:9.1f}"
              f" | repair={rec['repairCostUs']:7.1f}"
              f" roundExcess={rec['rejectRoundExcessUs']:8.1f}")

    rank = rank_test(table)
    if rank:
        print(f"\nrejection-round commit across merge eec2c14b (rank test): "
              f"U={rank['U']:.0f}/{rank['UMax']} "
              f"(n_pre={rank['nPre']}, n_post={rank['nPost']}), "
              f"median {rank['medianPreUs']:.1f} -> {rank['medianPostUs']:.1f} us, "
              f"step +{rank['stepUs']:.1f} us, z={rank['z']:.2f}")

    print("\nE90 reference (harness=local, 512 tokens, depth 8, Edward host):")
    for ph, v in E90.items():
        here = st.mean([leg[ph] for leg in legs if ph in leg])
        print(f"  {ph:16s} E90={v:10.1f}  E188={here:10.1f}  ratio={here / v:5.2f}")

    print("\nhost phases against realized draft depth d (spike-free rounds, "
          "same build, same host, harness=local):")
    print(f"  {'d':>2s} {'n':>3s} {'commit':>9s} {'readout':>9s} {'upkeep':>9s}")
    for d, r in depth["byDepth"].items():
        print(f"  {d:2d} {r['n']:3d} {r['commit_us']:9.1f} "
              f"{r['readout_us']:9.1f} {r['upkeep_us']:9.1f}")
    if depth["fit"]:
        f = depth["fit"]
        print(f"  OLS over {f['n']} rounds: commit_us = {f['interceptUs']:.1f} "
              f"+ {f['slopeUsPerDraft']:.1f} * d")
        print(f"  E90's 183.9 us implies realized depth "
              f"{f['impliedDepthAtE90']:.2f}")
    print(f"  excluded {depth['nSpikes']} spike rounds >= {SPIKE_US:.0f} us: "
          f"{[(t, rd) for t, rd, _ in depth['spikeRounds']]}")

    out = {"baseSha": base_sha, "arms": table, "depth": depth, "rankTest": rank,
           "legs": [{k: v for k, v in leg.items() if k != "rounds"} for leg in legs],
           "e90Reference": E90}
    pathlib.Path("/tmp/e188").mkdir(exist_ok=True)
    pathlib.Path("/tmp/e188/stage2_arms.json").write_text(json.dumps(out, indent=1))
    print("\nwrote /tmp/e188/stage2_arms.json")

    if args.dry_run:
        return

    import wandb

    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "qwen38-mlx-challenge-senpai")
    urls = []

    for leg in legs:
        run = wandb.init(entity=entity, project=project, group=args.group,
                         name=f"e188-{leg['tag']}", reinit=True,
                         config=identity(leg["tag"], base_sha) |
                         {"arm": leg["arm"], "experiment": "e188",
                          "stage": "stage2-commit-phase-bisection"})
        for r in leg["rounds"]:
            wandb.log({p: float(r[p]) for p in PHASES if p in r} |
                      {"round": r["round"]}, step=r["round"])
        run.summary.update(
            {p: leg[p] for p in PHASES if p in leg} |
            {p + "_iqr": leg[p + "_iqr"] for p in PHASES if p + "_iqr" in leg} |
            {"nRoundsScored": leg["nRoundsScored"]})
        urls.append((leg["tag"], run.id, run.url))
        run.finish()

    roll = wandb.init(entity=entity, project=project, group=args.group,
                      name="e188-stage2-rollup", reinit=True,
                      config={"experiment": "e188", "harness": "local",
                              "stage": "stage2-commit-phase-bisection",
                              "baseSha": base_sha,
                              "bisectedFile":
                              "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
                              "armOrder": [leg["arm"] for leg in legs]})
    roll.summary.update(
        {f"arm/{arm}/{k}": v for arm, rec in table.items()
         for k, v in rec.items() if not isinstance(v, list)} |
        {f"e90/{k}": v for k, v in E90.items()} |
        {f"depth/d{d}/{k}": v for d, r in depth["byDepth"].items()
         for k, v in r.items()} |
        {f"depthFit/{k}": v for k, v in depth["fit"].items()} |
        {f"rankTest/{k}": v for k, v in rank.items()} |
        {"depth/nSpikesExcluded": depth["nSpikes"]})
    urls.append(("rollup", roll.id, roll.url))
    roll.finish()

    print("\nW&B runs:")
    for tag, rid, url in urls:
        print(f"  {tag:22s} {rid}  {url}")
    pathlib.Path("/tmp/e188/stage2_wandb.json").write_text(
        json.dumps([{"leg": t, "runId": i, "url": u} for t, i, u in urls], indent=1))
    print("wrote /tmp/e188/stage2_wandb.json")


if __name__ == "__main__":
    main()
