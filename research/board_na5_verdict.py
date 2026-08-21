import json, math
from fractions import Fraction

SHA2P = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}

def rows():
    raw = json.load(open("/tmp/yukon-board/full.json"))
    if isinstance(raw, dict):
        for k in ("submissions", "rows", "data", "items"):
            if k in raw:
                raw = raw[k]
                break
    return [r for r in raw if isinstance(r, dict)]


def recover_rounds(dl):
    if dl <= 0:
        return None
    fr = Fraction(dl).limit_denominator(4000)
    n = fr.denominator
    while n <= 512:
        if (fr * n).denominator == 1:
            tpr = 512 / n
            rate = (512 - n) / (dl * n)
            if 1.0 <= tpr <= 8.0 and 0.30 <= rate <= 0.985:
                return n
        n += fr.denominator
    return None


def prompts(r):
    out = {}
    for pp in r["officialMetrics"]["per_prompt"]:
        name = SHA2P.get(pp["prompt_sha256"][:8], pp["prompt_sha256"][:8])
        out[name] = pp
    return out


BOARD_SERIAL = {
    "beagle": 0.037990260, "botany": 0.037996402, "drama": 0.037994712,
    "essays": 0.037997448, "medicine": 0.037994720, "plutarch": 0.037993427,
    "republic": 0.037993760, "travel": 0.038002089,
}


def serialfree(r):
    pr = prompts(r)
    vals = sorted(BOARD_SERIAL[p] / pr[p]["mtp_seconds_per_token_mean"] for p in pr)
    return 0.5 * (vals[3] + vals[4])


def LS(r):
    """centred fit on the five G=2 prompts, my committed definition"""
    pr = prompts(r)
    xs, ys = [], []
    for p in ("beagle", "republic", "essays", "medicine", "botany"):
        d = pr[p]["effective_mean_draft_len"]
        R = recover_rounds(d)
        if R is None:
            return None
        ru = 512.0 * pr[p]["mtp_seconds_per_token_mean"] / R * 1e6
        xs.append(1.0 + d)
        ys.append(ru)
    MBAR = 6.1723
    sxx = sum((x - MBAR) ** 2 for x in xs)
    sxy = sum((x - MBAR) * y for x, y in zip(xs, ys))
    S = sxy / sxx
    L = sum(ys) / len(ys) - S * (sum(xs) / len(xs) - MBAR)
    dp = pr["plutarch"]
    Rp = recover_rounds(dp["effective_mean_draft_len"])
    pl = 512.0 * dp["mtp_seconds_per_token_mean"] / Rp * 1e6 if Rp else float("nan")
    return L, S, pl, L / pl


ALL = rows()
TARGETS = ["ca9251b8", "9b241879", "cb8aeefb", "f04b102e", "83f0b282"]
sel = {}
for r in ALL:
    for t in TARGETS:
        if r["id"].startswith(t) and r.get("officialMetrics"):
            sel[t] = r

print("%-10s %-10s %14s %14s %10s %9s %8s" % ("id", "status", "published", "serialfree", "L", "S", "L/plut"))
for t in TARGETS:
    r = sel.get(t)
    if not r:
        print(t, "MISSING")
        continue
    ls = LS(r)
    if ls:
        L, S, pl, ratio = ls
        print("%-10s %-10s %14.8f %14.8f %10.1f %9.1f %8.4f"
              % (t, r["status"], r["officialScore"], serialfree(r), L, S, ratio))
    else:
        print("%-10s %-10s %14.8f %14.8f   round-recovery failed"
              % (t, r["status"], r["officialScore"], serialfree(r)))

print()
print("=== per-prompt candidate leg, ca9251b8 (NA=5 table) vs 9b241879 (NA<=4) ===")
a = prompts(sel["ca9251b8"])
b = prompts(sel["9b241879"])
print("%-9s %9s %9s %10s %12s %12s %9s" % ("prompt", "d_ca", "d_9b", "same_sched", "spt_ca", "spt_9b", "delta%"))
tot = []
for p in ("plutarch", "drama", "travel", "beagle", "republic", "essays", "medicine", "botany"):
    da = a[p]["effective_mean_draft_len"]
    db = b[p]["effective_mean_draft_len"]
    sa = a[p]["mtp_seconds_per_token_mean"]
    sb = b[p]["mtp_seconds_per_token_mean"]
    d = (sa / sb - 1.0) * 100.0
    tot.append(d)
    print("%-9s %9.5f %9.5f %10s %12.8f %12.8f %+9.4f"
          % (p, da, db, "YES" if abs(da - db) < 1e-12 else "no", sa, sb, d))
m = sum(tot) / len(tot)
sd = math.sqrt(sum((x - m) ** 2 for x in tot) / (len(tot) - 1))
print("mean %+0.4f %%  sd %0.4f %%   (positive = ca9251b8 SLOWER)" % (m, sd))
