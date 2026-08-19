#!/usr/bin/env python3
"""Who owns the extreme-fast MTP legs, and are they credible?"""
import json, math, statistics as st
FP, OURS, CROWN = "559b24eb", "2b0c36a078", "ef42e04327"
rows = json.load(open("/tmp/rows_live.json"))
sel = []
for r in rows:
    om = r.get("officialMetrics") or {}
    pp = om.get("per_prompt")
    if (isinstance(pp, list) and len(pp) == 8 and r.get("officialScore") is not None
            and all(isinstance(p.get("head_provenance_sha256"), str)
                    and p["head_provenance_sha256"].startswith(FP) for p in pp)):
        sel.append(r)
def tab(r):
    return {p["prompt_sha256"][:8]: p for p in r["officialMetrics"]["per_prompt"]}
prompts = sorted(tab(sel[0]).keys())
order = sorted(prompts, key=lambda q: st.median(
    [tab(r)[q]["serial_seconds_per_token_mean"]/tab(r)[q]["mtp_seconds_per_token_mean"]
     for r in sel]))
NAMES = ["plutarch","drama","travel","beagle","medicine","essays","republic","botany"]
name = {q: NAMES[i] for i, q in enumerate(order)}
ours = next(r for r in sel if str(r.get("submissionCommitSha","")).startswith(OURS))

for q in order:
    if name[q] not in ("beagle", "medicine"):
        continue
    print("=== " + name[q] + " : 6 fastest MTP legs among the 94 matched rows ===")
    ranked = sorted(sel, key=lambda r: tab(r)[q]["mtp_seconds_per_token_mean"])
    vals = [tab(r)[q]["mtp_seconds_per_token_mean"]*1000 for r in ranked]
    med = st.median(vals)
    print(f"{'ms/tok':>9s} {'vs med':>8s} {'commit':>9s} {'solver':18s} "
          f"{'score':>10s} {'draftlen':>9s} {'nondraft':>8s} {'parity':>6s} status")
    for r in ranked[:6]:
        p = tab(r)[q]
        v = p["mtp_seconds_per_token_mean"]*1000
        print(f"{v:9.3f} {100*(v/med-1):+7.2f}% {str(r.get('submissionCommitSha'))[:8]:>9s} "
              f"{str(r.get('solverUsername'))[:18]:18s} {r['officialScore']:10.6f} "
              f"{p.get('effective_mean_draft_len'):9.4f} "
              f"{p.get('non_drafting_round_count'):8d} "
              f"{str(p.get('parity_ok')):>6s} {r.get('status')}")
    ov = tab(ours)[q]["mtp_seconds_per_token_mean"]*1000
    orank = sum(1 for v in vals if v < ov) + 1
    print(f"  OURS {ov:.3f} ms/tok, rank {orank} of {len(vals)}, "
          f"{100*(ov/med-1):+.2f}% vs median")
    print(f"  rank1->rank2 step {100*(vals[1]/vals[0]-1):+.2f}%, "
          f"rank2->rank3 {100*(vals[2]/vals[1]-1):+.2f}%, "
          f"rank3->rank4 {100*(vals[3]/vals[2]-1):+.2f}%")
    print()

# does the rank-1 row on a binding prompt look anomalous elsewhere?
print("=== profile of the per-prompt argmin rows across ALL prompts (z vs cohort) ===")
argmins = {}
for q in order:
    b = min(sel, key=lambda r: tab(r)[q]["mtp_seconds_per_token_mean"])
    argmins[str(b.get("submissionCommitSha"))[:8]] = b
for sha, r in argmins.items():
    zs = []
    for q in order:
        vals = [tab(x)[q]["mtp_seconds_per_token_mean"] for x in sel]
        m, s = st.median(vals), st.stdev(vals)
        zs.append((tab(r)[q]["mtp_seconds_per_token_mean"] - m)/s)
    print(f"{sha} {str(r.get('solverUsername'))[:16]:16s} score {r['officialScore']:.6f} "
          f"z: " + " ".join(f"{z:+5.1f}" for z in zs))
print("prompt order: " + " ".join(f"{name[q][:5]:>5s}" for q in order))
