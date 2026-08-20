#!/usr/bin/env python3
"""The RANKED price of removing one QMV weight stream, measured on the board.

WHY THIS EXISTS
---------------
The whole remaining kernel axis of this campaign (`t55`, `t6`) is one
mechanism: make a verify width run as ONE weight stream instead of two. Every
price the campaign has quoted for it came from a LOCAL M4 Pro bandwidth ladder
pushed through a modelled ranked width mixture. Nothing anchored it to the
ranked M5 runner.

It turns out the board already contains the experiment, many times over.
`stream_dispatch_census.py ab` finds submission trees that are byte-identical
on every file EXCEPT the two QMV kernel files. Where two such trees differ at
exactly one verify width, that is a clean single-mechanism ranked A/B, already
paid for by another solver.

THE READOUT IS THE CANDIDATE LEG, NOT THE SCORE
-----------------------------------------------
`officialMetrics.per_prompt[].mtp_seconds_per_token_mean` IS the candidate leg,
per prompt, matched by `prompt_sha256`. Using it instead of the published score
avoids the median-of-eight aggregation and the pricer kink.

Two nulls make the instrument falsifiable rather than merely suggestive:

  * `serial_seconds_per_token_mean` comes from the runner-owned PREBUILT
    baseline workspace. No candidate edit can move it. Measured across every
    contrast in this file it is -0.031 % with sd 0.163 %. If that ever moves,
    the pairing is wrong and every number here is void.
  * `effective_mean_draft_len` must be IDENTICAL between the two arms. A
    kernel change that preserved exactness cannot change which drafts were
    accepted. Measured max |difference| across all contrasts: 0.0000.

WHAT IT MEASURES
----------------
Removing one weight stream at one verify width is worth about

    -0.64 % +/- 0.31 %   of the ranked candidate leg      (t = -2.0)

pooled over widths 4, 6 and 8 across 83 ranked runs in 13 fingerprint groups.

WHAT IT REFUTES
---------------
The pure weight-stream cost model over-prices this by 2.6x to 5x. Least
squares against the model's own per-width predictions gives a realisation
factor of about 0.19; a CONSTANT effect per stream removal fits the three
widths about twenty times better (chi-square 0.06 vs 1.37 on 2 dof). The
model's LARGEST prediction (M=6) has the SMALLEST measured effect.

The likely reason is visible in the wrapper. `first_m = tid.x * IPG` with an
early return at `first_m >= M` means the grid always launches M threadgroups
in x and the ones past the last group exit immediately. At M=4 with IPG 4,
ONE x-threadgroup in four does the work; with IPG 2, two do. Going to fewer
streams therefore also empties the machine, and the pure bytes/bandwidth model
has no term for that. That is edward's E63 memory-level-parallelism
hypothesis, arriving from the ranked side.

WHAT IT DOES NOT SAY
--------------------
It does NOT say `t55` and `t6` are not worth doing. On either model they are
worth roughly +1.0 % to +1.6 % of published score, against a base deficit far
smaller than that. It says the campaign should quote +1.0..+1.6 %, not the
+1.9..+2.4 % that ledger 200(A) derived from the local model alone.

A SECOND INSTRUMENT FACT, WORTH AS MUCH AS THE FIRST
-----------------------------------------------------
The same-fingerprint same-table pairs -- byte-identical SUBMITTED SURFACES,
two separate ranked runs -- give the empirical null of the ranked candidate
leg:

    n = 261 pairs   median +0.024 %   IQR [-1.07, +1.17] %
    MAD-scaled pair sd 1.65 %   =>  sd of ONE candidate leg  1.17 %
    5.7 % of pairs are beyond 3 %; the worst is 14.7 %

while the serial leg on those SAME pairs has sd 0.163 %. So the ranked
candidate leg is heavy-tailed and about seven times noisier than the serial
leg, and a single ranked candidate-leg comparison is not evidence.

Usage:
  ranked_stream_ab.py selftest
  ranked_stream_ab.py report
"""

import json
import math
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stream_dispatch_census as census  # noqa: E402

# Ranked QMV time share by verify width, beagle midpoints (ledger 199/200).
# Used ONLY to state the model this file tests, never to price a result.
RANKED_QMV_SHARE = {3: 0.0325, 4: 0.142, 5: 0.241, 6: 0.334,
                    7: 0.122, 8: 0.0735, 9: 0.0575}
PSI = 0.826                 # QMV share of the ranked candidate leg, 200(A)
BW = {2: 223.784, 3: 199.693, 4: 175.238, 5: 150.946, 6: 117.8, 7: 97.9}
MADK = 1.4826

# Pinned so a re-run that moves them fails loudly instead of quietly
# re-pricing the campaign. Update WITH the ledger entry that supersedes them.
EXPECT_SERIAL_NULL_ABS = 0.10       # %, mean |serial leg| across contrasts
EXPECT_DL_MAX = 1e-9                # draft length must match exactly
EXPECT_POOLED = -0.639              # %, candidate leg, one stream removal
EXPECT_POOLED_SE = 0.314            # %
EXPECT_RUN_SD = 1.092               # %, one ranked candidate leg, MAD-scaled
                                    # re-pinned 2026-08-20 with the 471-tree
                                    # corpus (275 null pairs, was 428 trees).
                                    # The pooled effect held at -0.639 % and its
                                    # se tightened 0.314 -> 0.294.

YUKON_ENV = "SENPAI_YUKON_ALL_JSON"
# Distilled board export, committed so this gate has real power in any
# checkout. The full `yukon_all.json` is ~10 MB and must never be committed;
# `distil` regenerates this file from it.
COMPACT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ranked_stream_ab_board.json")


def yukon_path():
    p = os.environ.get(YUKON_ENV)
    if p:
        return p
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True).stdout.strip()
    for cand in (os.path.join(root, "research", "yukon_all.json"),
                 os.path.join(os.path.dirname(root), "_advisor_scratch",
                              "yukon_all.json")):
        if os.path.exists(cand):
            return cand
    return None


def groups(M, ipg):
    out, g = [], 0
    while g * ipg < M:
        out.append(min(ipg, M - g * ipg))
        g += 1
    return out


def cost(M, ipg):
    return sum(1.0 / BW[max(2, n)] for n in groups(M, ipg))


def cell_pct(M, lo, hi):
    return 100.0 * (cost(M, lo) - cost(M, hi)) / cost(M, hi)


def mad_sd(v):
    m = statistics.median(v)
    return MADK * statistics.median([abs(x - m) for x in v])


def tkey(t):
    return tuple(sorted(t.items()))


def load():
    """Return (fingerprint -> [uuid]), (uuid -> table), (uuid -> metrics).

    Prefers the committed compact export so the gate works in a bare
    checkout; falls back to the full board export when regenerating.
    """
    if os.path.exists(COMPACT):
        d = json.load(open(COMPACT))
        fps, tables, obs = {}, {}, {}
        for uuid, e in d["trees"].items():
            tables[uuid] = {int(k): v for k, v in e["table"].items()}
            fps.setdefault(e["fp"], []).append(uuid)
            obs[uuid] = {
                "mtp": {k: math.log(v) for k, v in e["mtp"].items()},
                "ser": {k: math.log(v) for k, v in e["ser"].items()},
                "dl": e["dl"],
            }
        return fps, tables, obs
    path = yukon_path()
    if path is None or not os.path.exists(path):
        return None, None, None
    subs = json.load(open(path))["submissions"]
    by_uuid = {s["id"]: s for s in subs}
    fps, tables, obs = {}, {}, {}
    for name, ref in census.submission_refs():
        tbl = census.dispatch_table(ref)
        fp = census.non_kernel_fingerprint(ref)
        if tbl is None or fp is None:
            continue
        r = by_uuid.get(name)
        pp = ((r or {}).get("officialMetrics") or {}).get("per_prompt")
        if not pp:
            continue
        tables[name] = tbl
        fps.setdefault(fp, []).append(name)
        obs[name] = {
            "mtp": {p["prompt_sha256"]:
                    math.log(p["mtp_seconds_per_token_mean"]) for p in pp},
            "ser": {p["prompt_sha256"]:
                    math.log(p["serial_seconds_per_token_mean"]) for p in pp},
            "dl": {p["prompt_sha256"]:
                   p["effective_mean_draft_len"] for p in pp},
        }
    return fps, tables, obs


def arm_mean(obs, names, shared, field):
    return [statistics.mean(obs[n][field][k] for k in shared) for n in names]


def null_pairs(fps, tables, obs):
    """Same fingerprint AND same dispatch table: byte-identical surface."""
    mtp, ser = [], []
    for names in fps.values():
        byt = {}
        for n in names:
            byt.setdefault(tkey(tables[n]), []).append(n)
        for grp in byt.values():
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    sh = sorted(set(obs[grp[i]]["mtp"])
                                & set(obs[grp[j]]["mtp"]))
                    if len(sh) < 4:
                        continue
                    mtp.append(100.0 * statistics.mean(
                        obs[grp[i]]["mtp"][k] - obs[grp[j]]["mtp"][k]
                        for k in sh))
                    ser.append(100.0 * statistics.mean(
                        obs[grp[i]]["ser"][k] - obs[grp[j]]["ser"][k]
                        for k in sh))
    return mtp, ser


def contrasts(fps, tables, obs, sd_run):
    """One row per (fingerprint, width) where exactly one width differs."""
    out = []
    for fp, names in fps.items():
        byt = {}
        for n in names:
            byt.setdefault(tkey(tables[n]), []).append(n)
        ks = list(byt)
        for i in range(len(ks)):
            for j in range(len(ks)):
                if i == j:
                    continue
                ta, tb = dict(ks[i]), dict(ks[j])
                diff = [M for M in sorted(set(ta) | set(tb))
                        if ta.get(M) != tb.get(M)]
                if len(diff) != 1:
                    continue
                M = diff[0]
                if M not in ta or M not in tb:
                    continue
                sa, sb = math.ceil(M / ta[M]), math.ceil(M / tb[M])
                if sa >= sb:          # keep the LO-minus-HI orientation once
                    continue
                A, B = byt[ks[i]], byt[ks[j]]
                sh = None
                for n in A + B:
                    s = set(obs[n]["mtp"])
                    sh = s if sh is None else (sh & s)
                sh = sorted(sh or [])
                if len(sh) < 4:
                    continue
                d = 100.0 * (statistics.median(arm_mean(obs, A, sh, "mtp"))
                             - statistics.median(arm_mean(obs, B, sh, "mtp")))
                ds = 100.0 * (statistics.median(arm_mean(obs, A, sh, "ser"))
                              - statistics.median(arm_mean(obs, B, sh, "ser")))
                dl = max(abs(statistics.mean(obs[a]["dl"][k] for k in sh)
                             - statistics.mean(obs[b]["dl"][k] for k in sh))
                         for a in A for b in B)
                out.append(dict(fp=fp[:12], M=M, lo=ta[M], hi=tb[M],
                                sl=sa, sh=sb, nA=len(A), nB=len(B),
                                d=d, ser=ds, dl=dl,
                                se=sd_run * math.sqrt(1.0 / len(A)
                                                      + 1.0 / len(B))))
    return out


def pooled(rows):
    w = [1.0 / r["se"] ** 2 for r in rows]
    eff = sum(wi * r["d"] for wi, r in zip(w, rows)) / sum(w)
    return eff, math.sqrt(1.0 / sum(w))


def analyse():
    fps, tables, obs = load()
    if fps is None:
        return None
    nm, ns = null_pairs(fps, tables, obs)
    sd_run = mad_sd(nm) / math.sqrt(2.0)
    rows = contrasts(fps, tables, obs, sd_run)
    return dict(fps=fps, tables=tables, obs=obs, null_mtp=nm, null_ser=ns,
                sd_run=sd_run, rows=rows)


def report():
    a = analyse()
    if a is None:
        print("yukon_all.json not found; set %s" % YUKON_ENV)
        return 2
    nm, ns, rows, sd_run = a["null_mtp"], a["null_ser"], a["rows"], a["sd_run"]
    nm_s = sorted(nm)
    n = len(nm_s)
    print("=" * 76)
    print("EMPIRICAL NULL -- byte-identical submitted surface, two ranked runs")
    print("=" * 76)
    print("  pairs %d   median %+0.4f %%   IQR [%+0.3f, %+0.3f] %%"
          % (n, statistics.median(nm_s), nm_s[n // 4], nm_s[3 * n // 4]))
    print("  candidate leg pair sd: raw %0.3f %%  MAD-scaled %0.3f %%"
          % (statistics.stdev(nm), mad_sd(nm)))
    print("  serial leg    pair sd: raw %0.3f %%   mean %+0.4f %%"
          % (statistics.stdev(ns), statistics.mean(ns)))
    print("  => sd of ONE ranked candidate leg = %0.3f %%" % sd_run)
    print("  beyond 3 %%: %d of %d (%0.1f %%); worst %0.2f %%"
          % (sum(1 for x in nm if abs(x) > 3), n,
             100.0 * sum(1 for x in nm if abs(x) > 3) / n,
             max(abs(x) for x in nm)))
    print()
    print("=" * 76)
    print("SIGNAL -- one differing verify width, LO(fewer streams) minus HI")
    print("=" * 76)
    print("  fp            M  IPG   streams  nA nB   cand%     se%      t"
          "     serial%")
    for r in sorted(rows, key=lambda x: (x["M"], x["fp"])):
        print("  %-12s  %d  %d>%d  %d>%d    %2d %2d   %+7.3f  %6.3f  %6.2f"
              "  %+7.3f"
              % (r["fp"], r["M"], r["lo"], r["hi"], r["sl"], r["sh"],
                 r["nA"], r["nB"], r["d"], r["se"], r["d"] / r["se"],
                 r["ser"]))
    print()
    print("=" * 76)
    print("POOLED")
    print("=" * 76)
    print("  M   groups runs   effect%    se%      t     model%   ratio")
    ev = []
    for M in sorted(set(r["M"] for r in rows)):
        g = [r for r in rows if r["M"] == M]
        eff, se = pooled(g)
        g0 = max(g, key=lambda r: r["nA"] + r["nB"])
        pred = cell_pct(M, g0["lo"], g0["hi"]) * RANKED_QMV_SHARE[M] * PSI
        runs = sum(r["nA"] + r["nB"] for r in g)
        print("  %d   %5d %4d   %+7.3f  %6.3f %6.2f   %+7.3f  %6.3f"
              % (M, len(g), runs, eff, se, eff / se, pred, eff / pred))
        ev.append((M, eff, se, pred))
    eff, se = pooled(rows)
    print("  ALL %5d %4d   %+7.3f  %6.3f %6.2f   -- one stream removal, "
          "any width" % (len(rows), sum(r["nA"] + r["nB"] for r in rows),
                         eff, se, eff / se))
    print()
    print("  model comparison over the %d measured widths:" % len(ev))
    chi_c = sum(((e - eff) / s) ** 2 for _, e, s, _ in ev)
    rho = (sum(e * p / s ** 2 for _, e, s, p in ev)
           / sum(p * p / s ** 2 for _, e, s, p in ev))
    chi_p = sum(((e - rho * p) / s) ** 2 for _, e, s, p in ev)
    print("    CONSTANT per removal  : %+0.3f %%      chi2 = %0.3f on %d dof"
          % (eff, chi_c, len(ev) - 1))
    print("    PROPORTIONAL to model : rho = %0.3f    chi2 = %0.3f on %d dof"
          % (rho, chi_p, len(ev) - 1))
    print()
    print("=" * 76)
    print("PRICE OF t55 AND t6 UNDER BOTH MODELS")
    print("=" * 76)
    for nm_, M, lo, hi in (("t55", 5, 5, 3), ("t6", 6, 6, 3)):
        pred = cell_pct(M, lo, hi) * RANKED_QMV_SHARE[M] * PSI
        print("  %-4s local model %+7.3f %%   flat %+7.3f %%   "
              "proportional %+7.3f %%" % (nm_, pred, eff, rho * pred))
    tot_model = sum(cell_pct(M, lo, hi) * RANKED_QMV_SHARE[M] * PSI
                    for M, lo, hi in ((5, 5, 3), (6, 6, 3)))
    for label, leg in (("local model", tot_model),
                       ("flat", 2.0 * eff),
                       ("proportional", rho * tot_model)):
        raw = 100.0 * (1.0 / (1.0 + leg / 100.0) - 1.0)
        pub = raw if raw <= 1.0551 else 1.0551 + (raw - 1.0551) * 0.483694
        print("  t55+t6 %-13s leg %+7.3f %% -> raw %+7.3f %% -> published "
              "%+7.3f %%" % (label, leg, raw, pub))
    return 0


def selftest():
    fails = []
    a = analyse()
    if a is None:
        print("SELFTEST SKIP: yukon_all.json not present (set %s). This gate "
              "needs the board export and is advisory-only in a bare "
              "checkout." % YUKON_ENV)
        return 0
    rows, nm, ns = a["rows"], a["null_mtp"], a["null_ser"]

    # 1. THE INSTRUMENT'S OWN NULL. The serial leg is produced by the
    #    runner-owned prebuilt baseline workspace. If a candidate-side
    #    contrast moves it, the pairing is wrong and every effect here is an
    #    artefact. This is the check that can kill the whole file.
    if not rows:
        fails.append("no single-width contrasts found at all")
    else:
        sm = statistics.mean(r["ser"] for r in rows)
        cm = statistics.mean(r["d"] for r in rows)
        if abs(sm) > EXPECT_SERIAL_NULL_ABS:
            fails.append("serial leg moved with a candidate-only contrast: "
                         "mean serial = %+0.4f %%, |.| > %0.2f %%. The A/B "
                         "pairing is invalid." % (sm, EXPECT_SERIAL_NULL_ABS))
        if abs(sm) > 0.4 * abs(cm):
            fails.append("serial shift %+0.4f %% is not small against the "
                         "candidate shift %+0.4f %%; the contrast is not "
                         "candidate-specific" % (sm, cm))
        dmax = max(r["dl"] for r in rows)
        if dmax > EXPECT_DL_MAX:
            fails.append("a contrast changed effective_mean_draft_len by "
                         "%0.6f; a kernel-only A/B cannot move acceptance, so "
                         "that pair is not kernel-only" % dmax)

    # 2. POSITIVE CONTROL. Inject a known offset into one arm and require the
    #    estimator to recover it. Without this, a contrast function that
    #    always returned 0 would pass check 1 perfectly.
    fps, tables, obs = a["fps"], a["tables"], a["obs"]
    target = max(rows, key=lambda r: r["nA"] + r["nB"]) if rows else None
    if target:
        import copy
        obs2 = copy.deepcopy(obs)
        # find the LO arm of that contrast and slow it by exactly 5 %
        moved = 0
        for fp, names in fps.items():
            if fp[:12] != target["fp"]:
                continue
            for n in names:
                if math.ceil(target["M"] / tables[n][target["M"]]) \
                        == target["sl"]:
                    for k in obs2[n]["mtp"]:
                        obs2[n]["mtp"][k] += math.log(1.05)
                    moved += 1
        r2 = [r for r in contrasts(fps, tables, obs2, a["sd_run"])
              if r["fp"] == target["fp"] and r["M"] == target["M"]
              and r["sl"] == target["sl"]]
        if moved == 0:
            fails.append("positive control could not find the LO arm")
        elif not r2:
            fails.append("positive control lost the contrast entirely")
        else:
            got = r2[0]["d"] - target["d"]
            want = 100.0 * math.log(1.05)
            if abs(got - want) > 1e-6:
                fails.append("positive control: injected %+0.4f %% into the "
                             "LO arm and recovered %+0.4f %%" % (want, got))

    # 3. NEGATIVE CONTROL on the same-table null. If `non_kernel_fingerprint`
    #    were vacuous, unrelated trees would land in one group and the null
    #    would be enormous. Pin its centre, not its spread: the spread is the
    #    finding.
    if abs(statistics.median(nm)) > 0.20:
        fails.append("same-table null median is %+0.4f %%, expected |.| <= "
                     "0.20 %% -- the fingerprint is grouping unlike trees"
                     % statistics.median(nm))
    if statistics.stdev(ns) > 0.30:
        fails.append("same-table SERIAL null sd is %0.4f %%, expected <= "
                     "0.30 %% -- the runner baseline is not stable and no "
                     "ranked comparison in this campaign is safe"
                     % statistics.stdev(ns))

    # 4. THE HEADLINE, PINNED. A board refresh that moves it must be noticed.
    if rows:
        eff, se = pooled(rows)
        if abs(eff - EXPECT_POOLED) > 0.05:
            fails.append("pooled one-stream-removal effect is %+0.3f %%, "
                         "pinned %+0.3f %% -- the board moved; re-read the "
                         "report and update the ledger" % (eff, EXPECT_POOLED))
        if abs(se - EXPECT_POOLED_SE) > 0.05:
            fails.append("pooled se is %0.3f %%, pinned %0.3f %%"
                         % (se, EXPECT_POOLED_SE))
    if abs(a["sd_run"] - EXPECT_RUN_SD) > 0.05:
        fails.append("ranked candidate-leg run sd is %0.3f %%, pinned %0.3f %%"
                     % (a["sd_run"], EXPECT_RUN_SD))

    # 5. ARITHMETIC, on constructed inputs.
    if groups(9, 5) != [5, 4] or groups(6, 3) != [3, 3] or groups(4, 2) != [2, 2]:
        fails.append("group decomposition is wrong")
    if abs(cell_pct(6, 6, 3) - (-15.24)) > 0.02:
        fails.append("t6 cell effect is %0.3f %%, expected -15.24 %%"
                     % cell_pct(6, 6, 3))
    if abs(cell_pct(5, 5, 3) - (-30.09)) > 0.02:
        fails.append("t55 cell effect is %0.3f %%, expected -30.09 %%"
                     % cell_pct(5, 5, 3))

    if fails:
        print("SELFTEST FAIL (%d)" % len(fails))
        for f in fails:
            print("  - %s" % f)
        return 1
    eff, se = pooled(rows)
    print("SELFTEST PASS: %d single-width ranked contrasts over %d runs; "
          "serial-leg null holds; draft length matches exactly; a +5.00 %% "
          "injection is recovered to 1e-6; one stream removal = %+0.3f %% "
          "+/- %0.3f %% of the ranked candidate leg; candidate-leg run sd "
          "%0.3f %%."
          % (len(rows), sum(r["nA"] + r["nB"] for r in rows), eff, se,
             a["sd_run"]))
    return 0


def distil():
    """Rebuild the committed compact export from the full board export."""
    path = yukon_path()
    if path is None or not os.path.exists(path):
        print("full board export not found; set %s" % YUKON_ENV)
        return 2
    subs = json.load(open(path))["submissions"]
    by_uuid = {s["id"]: s for s in subs}
    trees = {}
    for name, ref in census.submission_refs():
        tbl = census.dispatch_table(ref)
        fp = census.non_kernel_fingerprint(ref)
        if tbl is None or fp is None:
            continue
        r = by_uuid.get(name)
        pp = ((r or {}).get("officialMetrics") or {}).get("per_prompt")
        if not pp:
            continue
        trees[name] = {
            "fp": fp,
            "table": {str(k): v for k, v in sorted(tbl.items())},
            "mtp": {p["prompt_sha256"][:8]: p["mtp_seconds_per_token_mean"]
                    for p in pp},
            "ser": {p["prompt_sha256"][:8]: p["serial_seconds_per_token_mean"]
                    for p in pp},
            "dl": {p["prompt_sha256"][:8]: p["effective_mean_draft_len"]
                   for p in pp},
        }
    doc = {
        "what": "distilled Yukon board export for ranked_stream_ab.py",
        "source": "GET /benchmarks/<id>/submissions?all=true",
        "fields": "per-prompt candidate and serial seconds per token, "
                  "effective mean draft length, QMV dispatch table, and the "
                  "blob-SHA fingerprint of every submitted file EXCEPT the "
                  "two QMV kernel files",
        "trees": trees,
    }
    with open(COMPACT, "w") as f:
        json.dump(doc, f, sort_keys=True, separators=(",", ":"))
    print("wrote %s  (%d trees, %d bytes)"
          % (COMPACT, len(trees), os.path.getsize(COMPACT)))
    return 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if mode == "selftest":
        return selftest()
    if mode == "report":
        return report()
    if mode == "distil":
        return distil()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
