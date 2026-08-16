import json

d = json.load(open("research/arm-summary.json"))
pa = d["position_acceptance"]
for run in ("I", "J", "K", "L", "M", "N", "O", "P"):
    s = pa.get(run)
    if not s:
        continue
    cond = [
        round(h / o, 3) if o else None
        for h, o in zip(s["conditional_hits"], s["conditional_obs"])
    ]
    unc = [
        round(h / o, 3) if o else None
        for h, o in zip(s["unconditional_hits"], s["unconditional_obs"])
    ]
    print(run, "obs   ", s["conditional_obs"][:8])
    print(run, "cond  ", cond[:8])
    print(run, "uncond", unc[:8])
print()
print("HEADLINE")
print(json.dumps(d["headline"], indent=1)[:1800])
