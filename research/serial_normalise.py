# Advisor tool for ledger items 111-112: serial-leg variance decomposition and
# serial-normalised leaderboard over the same-head (559b24eb) ranked population.
# Requires /tmp/rows.json from research/ranked_telemetry.py --refresh (or the
# cache at .mlxfast-private/ranked-telemetry.json). Read-only; no GPU.

import json, statistics as st

NAMES = {"919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
         "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
         "ea82dcb5": "republic", "3b10cb4d": "travel"}
rows = json.load(open("/tmp/rows.json"))
order = ["plutarch", "drama", "travel", "beagle", "medicine", "republic", "essays", "botany"]


def prof(r):
    m = r.get("officialMetrics") or {}
    return {NAMES.get(p["prompt_sha256"][:8], p["prompt_sha256"][:8]): p
            for p in (m.get("per_prompt") or [])}


pop = []
for r in rows:
    if r.get("officialScore") is None:
        continue
    p = prof(r)
    if len(p) < 8:
        continue
    if {v["head_provenance_sha256"][:8] for v in p.values()} != {"559b24eb"}:
        continue
    pop.append((r, p))

# per-prompt grand-mean serial across the whole same-head population
gser = {k: st.mean(p[k]["serial_seconds_per_token_mean"] for _, p in pop) for k in order}
print("per-prompt grand-mean serial (n=%d runs):" % len(pop))
print("   " + "  ".join("%s %.7f" % (k[:4], gser[k]) for k in order))


def score_from(ratios):
    v = sorted(ratios)
    return (v[3] + v[4]) / 2


out = []
for r, p in pop:
    raw = [p[k]["raw_ratio_of_means"] for k in order]
    # replace each prompt's serial numerator with the population grand mean,
    # holding the run's own measured MTP leg fixed
    norm = [p[k]["raw_ratio_of_means"] * gser[k] / p[k]["serial_seconds_per_token_mean"]
            for k in order]
    out.append({"id": r["id"][:8], "who": (r.get("solverUsername") or "")[:17],
                "off": r["officialScore"], "norm": score_from(norm),
                "ser": st.mean(p[k]["serial_seconds_per_token_mean"] for k in order)})

by_off = sorted(out, key=lambda x: -x["off"])
by_norm = sorted(out, key=lambda x: -x["norm"])
nrank = {x["id"]: i + 1 for i, x in enumerate(by_norm)}

print()
print("%-4s %-9s %-17s %12s %12s %8s %7s" % ("off", "id", "solver", "official", "serial-norm", "d%", "normrk"))
for i, x in enumerate(by_off[:16], 1):
    mark = "  <== OURS" if x["id"] == "ca9251b8" else ""
    print("%-4d %-9s %-17s %12.6f %12.6f %+8.3f %7d%s"
          % (i, x["id"], x["who"], x["off"], x["norm"],
             100 * (x["norm"] / x["off"] - 1), nrank[x["id"]], mark))

print()
print("TOP 8 AFTER NORMALISING THE SERIAL LEG:")
for i, x in enumerate(by_norm[:8], 1):
    mark = "  <== OURS" if x["id"] == "ca9251b8" else ""
    print("  %d. %-9s %-17s norm %.6f  (official %.6f, official rank %d)%s"
          % (i, x["id"], x["who"], x["norm"], x["off"],
             1 + sum(1 for y in out if y["off"] > x["off"]), mark))

ours = [x for x in out if x["id"] == "ca9251b8"][0]
print()
print("OURS: official %.6f (rank %d) -> serial-normalised %.6f (rank %d)"
      % (ours["off"], 1 + sum(1 for y in out if y["off"] > ours["off"]),
         ours["norm"], nrank["ca9251b8"]))
top = by_norm[0]
print("normalised gap to best: %+.3f %%" % (100 * (ours["norm"] / top["norm"] - 1)))
sp = [x["norm"] for x in by_norm[:10]]
print("normalised top-10 span: %.3f %%   (official top-10 span %.3f %%)"
      % (100 * (sp[0] / sp[-1] - 1), 100 * (by_off[0]["off"] / by_off[9]["off"] - 1)))
