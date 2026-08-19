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
    ser = [p[k]["serial_seconds_per_token_mean"] for k in order]
    mtp = [p[k]["mtp_seconds_per_token_mean"] for k in order]
    pop.append({"id": r["id"][:8], "who": (r.get("solverUsername") or "")[:16],
                "score": r["officialScore"], "ser": ser, "mtp": mtp,
                "sermean": sum(ser) / 8})
print("population on head 559b24eb with 8 prompts: %d" % len(pop))

# variance decomposition: within-run vs between-run for the serial leg
grand = st.mean(x["sermean"] for x in pop)
between = st.pstdev([x["sermean"] for x in pop])
within = st.mean([st.pstdev([v / st.mean(x["ser"]) for v in x["ser"]]) for x in pop])
print("serial leg: grand mean %.7f" % grand)
print("  BETWEEN-run sd of per-run mean : %.4f %%" % (100 * between / grand))
print("  WITHIN-run sd across 8 prompts : %.4f %% (normalised)" % (100 * within))
print("  -> ratio between/within = %.2f" % ((between / grand) / within))

# Is a whole-run serial offset a real, persistent per-run property?
pop.sort(key=lambda x: x["sermean"])
print("\nslowest-serial 5 runs:")
for x in pop[-5:]:
    print("   %-9s %-16s ser %.7f  score %.6f" % (x["id"], x["who"], x["sermean"], x["score"]))
print("fastest-serial 5 runs:")
for x in pop[:5]:
    print("   %-9s %-16s ser %.7f  score %.6f" % (x["id"], x["who"], x["sermean"], x["score"]))

# correlation of per-run serial offset with score
def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** .5
    db = sum((y - mb) ** 2 for y in b) ** .5
    return num / (da * db)


s = [x["sermean"] for x in pop]
sc = [x["score"] for x in pop]
m = [sum(x["mtp"]) / 8 for x in pop]
print("\ncorr(serial_mean, score)   = %+.3f" % pearson(s, sc))
print("corr(mtp_mean,    score)   = %+.3f" % pearson(m, sc))
print("corr(serial_mean, mtp_mean)= %+.3f   <-- if strongly +, a run-level box-speed effect exists" % pearson(s, m))

# restrict to the tight top cluster
tight = [x for x in pop if x["score"] > 3.22]
s2 = [x["sermean"] for x in tight]
sc2 = [x["score"] for x in tight]
m2 = [sum(x["mtp"]) / 8 for x in tight]
print("\ntop cluster (score > 3.22), n=%d" % len(tight))
print("  corr(serial_mean, score)    = %+.3f" % pearson(s2, sc2))
print("  corr(serial_mean, mtp_mean) = %+.3f" % pearson(s2, m2))
print("  serial_mean range: %.7f .. %.7f  (%.3f %%)"
      % (min(s2), max(s2), 100 * (max(s2) / min(s2) - 1)))
ours = [x for x in pop if x["id"] == "ca9251b8"][0]
rank_ser = 1 + sum(1 for x in pop if x["sermean"] < ours["sermean"])
print("  our serial_mean %.7f -> %d-th slowest of %d (1 = fastest)" % (ours["sermean"], rank_ser, len(pop)))
