#!/usr/bin/env python3
"""E178 step 1. Content-addressed identical-tree groups for the ranked board.

Identity criterion (exact, not heuristic): every board submission id has an
`upstream refs/heads/submissions/<id>` ref. A treeless fetch (`--filter=tree:0
--depth=1`) into a scratch bare repo yields each snapshot commit object, whose
`tree` field is the content address of the whole submitted snapshot. Two rows
with the same tree sha are byte-identical trees by construction, across solvers
and across differing commit shas.

Writes /tmp/e178/groups.json.

    git ls-remote https://github.com/Layr-Labs/qwen-3.8-mtp-challenge > /tmp/upstream_refs.txt
    git init --bare -q /tmp/subtrees
    (cd /tmp/subtrees && git fetch -q --filter=tree:0 --depth=1 <url> <refspecs>)
    (cd /tmp/subtrees && git for-each-ref --format='%(refname) %(objectname) %(tree)' \
        'refs/heads/submissions/*' > /tmp/sub_trees.txt)
    python3 research/e178_tree_groups.py
"""
import collections
import json
import os

BOARD = "/tmp/yukon-board/full.json"
TREES = "/tmp/sub_trees.txt"
OUT = "/tmp/e178/groups.json"

PROMPTS = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
LADDER = {"5a9f130a": "A", "180db842": "B", "fda590bb": "C",
          "2c885d64": "D", "90c131dc": "E"}


def load():
    rows = json.load(open(BOARD))
    tree = {}
    for line in open(TREES):
        ref, commit, t = line.split()
        tree[ref.rsplit("/", 1)[-1]] = (commit, t)
    return rows, tree


def receipt(row, tree):
    om = row.get("officialMetrics") or {}
    pp = om.get("per_prompt") or []
    if not pp:
        return None
    per = {}
    for p in pp:
        name = PROMPTS.get(p["prompt_sha256"][:8])
        per[name] = {
            "serial": p["serial_seconds_per_token_mean"],
            "mtp": p["mtp_seconds_per_token_mean"],
            "prefill": p.get("prefill_seconds_per_token"),
            "raw": p["raw_ratio_of_means"],
            "draft_len": p["effective_mean_draft_len"],
            "non_drafting": p["non_drafting_round_count"],
            "head": p["head_provenance_sha256"][:12],
            "parity_ok": p["parity_ok"],
        }
    sid = row["id"]
    return {
        "id8": sid[:8],
        "ladder": LADDER.get(sid[:8]),
        "solver": row["solverUsername"],
        "status": row["status"],
        "promotion": row.get("promotionStatus"),
        "score": row["officialScore"],
        "createdAt": row["createdAt"],
        "commit": (row.get("officialMetrics") or {}).get("commit"),
        "snapshot_commit": tree[sid][0],
        "tree": tree[sid][1],
        "depth": om.get("mtp_max_draft_depth"),
        "per_prompt": per,
    }


def main():
    rows, tree = load()
    scored = []
    for r in rows:
        if r["id"] in tree:
            rec = receipt(r, tree)
            if rec and len(rec["per_prompt"]) == 8 and rec["score"]:
                scored.append(rec)
    groups = collections.defaultdict(list)
    for rec in scored:
        groups[rec["tree"]].append(rec)
    for members in groups.values():
        members.sort(key=lambda m: m["createdAt"])
    multi = {t: m for t, m in groups.items() if len(m) >= 2}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"scored": scored, "groups": multi}, open(OUT, "w"))

    print("scored receipts with exact tree identity: %d" % len(scored))
    print("distinct trees: %d" % len(groups))
    print("identical-tree groups with >=2 scored members: %d (%d receipts)"
          % (len(multi), sum(len(m) for m in multi.values())))
    print("group size histogram: %s"
          % dict(sorted(collections.Counter(len(m) for m in multi.values()).items())))
    print()
    hdr = "%-14s %-4s %-9s %-14s %-24s %-13s %s"
    print(hdr % ("tree", "n", "id8", "solver", "createdAt", "score", "span_h"))
    for t, m in sorted(multi.items(), key=lambda kv: -len(kv[1])):
        t0 = m[0]["createdAt"]
        for rec in m:
            span = (_hours(rec["createdAt"]) - _hours(t0))
            print(hdr % (t[:12], len(m), rec["id8"], rec["solver"][:14],
                         rec["createdAt"], "%.10f" % rec["score"], "%.2f" % span))
        print()


def _hours(iso):
    import datetime
    return datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() / 3600.0


if __name__ == "__main__":
    main()
