"""Assemble the full E9 evidence table from every recorded draft-bits arm.

  python3 research/e9_evidence.py [TAG_PREFIX ...]

One row per arm directory under .mlxfast-private/draft-bits/, carrying the
provenance an arm needs to be trusted (worker digest, head, base, dirty state,
W&B run) alongside its metrics. `draft_readouts_total` is not reported by the
harness; it is D * round_count, which is exact because D is defined as drafts
proposed per round.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / ".mlxfast-private" / "draft-bits"
IDENT = re.compile(r"^(?:run-draft-bits-arm: )?([a-z_0-9]+)=(.*)$")


def arm(d):
    leg = json.loads((d / "amdahl.json").read_text())["mtp_leg"]
    ident = {}
    for line in (d / "identity.txt").read_text().splitlines():
        for field in line.split("run-draft-bits-arm: ")[-1].split(" "):
            m = IDENT.match(field)
            if m:
                ident.setdefault(m.group(1), m.group(2))
    rusage = (d / "rusage.txt").read_text() if (d / "rusage.txt").exists() else ""
    run = re.search(r"/runs/([a-z0-9]{8})", rusage)
    rss = re.search(r"\s+(\d+)\s+maximum resident set size", rusage)
    spt = leg["parent_measured_seconds_per_token"] * 1000
    return {
        "tag": d.name,
        "bits": ident.get("bits", "?"),
        "env": ident.get("env_draft_bits", "?"),
        "head": ident.get("head", "?")[:7],
        "dirty": ident.get("dirty", "?"),
        "base": ident.get("base_sha", "?")[:8],
        "worker": ident.get("worker_sha256", "?")[:12],
        "wandb": run.group(1) if run else "-",
        "peak_gb": round(int(rss.group(1)) / 2**30, 2) if rss else None,
        "spt_ms": spt,
        "accept": leg["accepted_draft_rate"],
        "rounds": leg["round_count"],
        "tpr": leg["tokens_per_round"],
        "D": leg["effective_mean_draft_len"],
        "readouts": round(leg["effective_mean_draft_len"] * leg["round_count"]),
        "depth": leg["mtp_depth"],
        "match": leg["all_tokens_matched"],
        "div": leg["residual_divergence_count"],
        "first_block_ms": leg["first_block_seconds"] * 1000,
        "ms_per_round": spt * leg["emitted_token_total"] / leg["round_count"],
    }


def main(prefixes):
    dirs = sorted(p for p in ROOT.iterdir() if (p / "amdahl.json").exists())
    if prefixes:
        dirs = [p for p in dirs if any(p.name.startswith(x) for x in prefixes)]
    rows = [arm(p) for p in dirs]
    hdr = (
        f"| {'arm':<18} | {'env':>6} | {'ms/token':>9} | {'accepted_draft_rate':<20} | {'rnds':>4} "
        f"| {'D':>6} | {'rdouts':>6} | {'dep':>3} | {'match':>5} | {'div':>3} "
        f"| {'worker_sha256':<12} | {'wandb':<8} | {'base':<8} | {'head':<7} | {'dirty':>5} |"
    )
    print(hdr)
    print("|" + "|".join("-" * (len(c) + 2) for c in hdr.split("|")[1:-1]) + "|")
    for r in rows:
        print(
            f"| {r['tag']:<18} | {r['env']:>6} | {r['spt_ms']:9.4f} | {r['accept']:<20.16f} | {r['rounds']:4d} "
            f"| {r['D']:6.4f} | {r['readouts']:6d} | {r['depth']:3d} | {str(r['match']):>5} | {r['div']:3d} "
            f"| {r['worker']:<12} | {r['wandb']:<8} | {r['base']:<8} | {r['head']:<7} | {r['dirty']:>5} |"
        )
    print()
    for r in rows:
        print(
            f"{r['tag']:<18} ms/round {r['ms_per_round']:8.3f}  first_block {r['first_block_ms']:7.2f} ms"
            f"  peak_rss {r['peak_gb']} GiB"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
