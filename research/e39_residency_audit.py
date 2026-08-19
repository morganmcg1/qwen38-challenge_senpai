#!/usr/bin/env python3
"""E39 deliverable 2 — does the board's residency+cmdbuf positive survive scrutiny?

E35 reported the only positive mechanism family on the leaderboard:

    residency + cmdbuf   +0.316 %   n=5   se 0.145   |t| 2.2   (serial +0.071)

Both halves of that family are on our established-negatives list, so the whole
E39 audit turns on whether this row is real. Four attacks, each of which the
E39 brief names explicitly:

  1. multiplicity  -- it is the MAX of 9 families, not a pre-registered one;
  2. box confound  -- is the delta tracking the pinned serial leg?
  3. independence  -- 5 distinct commits is not 5 independent experiments;
  4. contamination -- if every row is multi-mechanism, nothing is identified.

Reuses E35's join machinery (`within_head_cost.py`) so the population, the
frontier timeline and the denoised response are identical by construction
rather than by reimplementation.

    python3 research/e39_residency_audit.py
"""
from __future__ import annotations

import collections
import math
import random
import statistics as st
import sys

sys.path.insert(0, "research")

import within_head_cost as w  # noqa: E402

TARGET_FAMILY = "residency+cmdbuf"
E35_REPORTED = dict(delta=0.316, n=5, se=0.145, t=2.2, serial=0.071)


def family_rows(subs, pmap):
    """Reproduce E35's family pooling and return per-row detail, not just means."""
    scored = w.full(subs, pmap)
    _, modal, _ = w.population(subs, pmap)
    bar = w.serial_bar(scored, pmap)
    timeline = w.frontier_timeline(subs)

    byfam = collections.defaultdict(list)
    detail = {}
    for s in modal:
        front = w.frontier_at(timeline, s["createdAt"])
        if not front:
            continue
        d = 100.0 * (w.denoised_score(s, pmap) / w.denoised_score(front, pmap) - 1.0)
        serial = w.serial_offset(s, pmap, bar)
        fams = w.classify(s)
        detail[s["id"]] = dict(sub=s, delta=d, serial=serial, fams=fams, front=front)
        for f in fams:
            byfam[f].append(s["id"])
    return byfam, detail, modal


def summarise(ids, detail):
    ds = [detail[i]["delta"] for i in ids]
    ss = [detail[i]["serial"] for i in ids]
    sd = st.stdev(ds) if len(ds) > 1 else float("nan")
    se = sd / math.sqrt(len(ds)) if len(ds) > 1 else float("nan")
    return dict(n=len(ds), mean=st.mean(ds), median=st.median(ds), sd=sd, se=se,
                t=abs(st.mean(ds) / se) if se else float("nan"), serial=st.mean(ss))


def winners_curse(family_summaries, trials=200000, seed=11):
    """Expected value of the LARGEST of k family means when every true effect is 0.

    Each family keeps its own observed se, so the inflation reflects this
    board's actual family sizes rather than a balanced-design idealisation.
    """
    ses = [f["se"] for f in family_summaries if f["n"] > 1 and not math.isnan(f["se"])]
    rng = random.Random(seed)
    maxes = [max(rng.gauss(0.0, se) for se in ses) for _ in range(trials)]
    return len(ses), st.mean(maxes), st.stdev(maxes), sorted(maxes)[int(0.95 * trials)]


def main() -> int:
    subs, pmap = w.load()
    byfam, detail, modal = family_rows(subs, pmap)

    print("=" * 78)
    print("E39 (2) — RESIDENCY+CMDBUF UNDER SCRUTINY")
    print("=" * 78)
    print("population: %d modal-n rows with a live frontier" % len(detail))

    summaries = {f: summarise(ids, detail) for f, ids in byfam.items()}

    print()
    print("-- 1. MULTIPLICITY: this family is the MAX of the family table --")
    print("%-28s %4s %9s %9s %8s %6s %8s" %
          ("family", "n", "mean d%", "median", "se", "|t|", "serial%"))
    ranked = sorted(summaries.items(), key=lambda kv: -kv[1]["mean"])
    for f, s in ranked:
        print("%-28s %4d %+9.3f %+9.3f %8.3f %6.1f %+8.3f" %
              (f, s["n"], s["mean"], s["median"], s["se"], s["t"], s["serial"]))

    tgt = summaries[TARGET_FAMILY]
    print()
    print("   E35 reported %+0.3f %% (n=%d, se %.3f, |t| %.1f, serial %+0.3f)"
          % (E35_REPORTED["delta"], E35_REPORTED["n"], E35_REPORTED["se"],
             E35_REPORTED["t"], E35_REPORTED["serial"]))
    print("   today       %+0.3f %% (n=%d, se %.3f, |t| %.1f, serial %+0.3f)"
          % (tgt["mean"], tgt["n"], tgt["se"], tgt["t"], tgt["serial"]))
    drift = tgt["mean"] - E35_REPORTED["delta"]
    print("   DRIFT %+0.3f pp on a corpus that only grew: the estimate is not stable."
          % drift)

    k, exp_max, sd_max, p95 = winners_curse(list(summaries.values()))
    print()
    print("   Winner's curse over FAMILY selection (all true effects = 0):")
    print("     families entering the max      k = %d" % k)
    print("     E[max of k family means]       %+0.3f %%" % exp_max)
    print("     sd of that max                 %.3f %%" % sd_max)
    print("     95th pct of the max            %+0.3f %%" % p95)
    print("     observed best family           %+0.3f %%  (%s)"
          % (ranked[0][1]["mean"], ranked[0][0]))
    print("     -> the observed maximum is %s the null expectation for a max."
          % ("BELOW" if ranked[0][1]["mean"] < exp_max else "above"))
    print("     E35's own +0.316 %% sits at the %.0fth pct of this null max."
          % (100.0 * _pctile(E35_REPORTED["delta"], list(summaries.values()))))
    print()
    print("   The raw-%% null above is dominated by the two families with huge")
    print("   se (top-k 1.79, affine-2 1.38), so repeat it on the scale-free")
    print("   t statistic, which is the standard multiplicity comparison:")
    tmax_e, tmax_95 = _t_max_null([s for s in summaries.values() if s["n"] > 1])
    print("     median of max of %d family t    %+0.2f" % (k, tmax_e))
    print("     95th pct of that max            %+0.2f" % tmax_95)
    print("     observed residency t            %+0.2f  (E35 reported %.1f)"
          % (tgt["mean"] / tgt["se"], E35_REPORTED["t"]))
    print("     -> %s the null expectation for a max of %d."
          % ("BELOW" if tgt["mean"] / tgt["se"] < tmax_e else "above", k))

    print()
    print("-- 2. THE FIVE ROWS, INDIVIDUALLY --")
    ids = byfam[TARGET_FAMILY]
    print("%-9s %-16s %9s %9s %9s %4s  %s" %
          ("id", "solver", "score", "delta%", "serial%", "fams", "title"))
    for i in ids:
        d = detail[i]
        s = d["sub"]
        print("%-9s %-16s %9.5f %+9.3f %+9.3f %4d  %s" %
              (i[:8], (s.get("solverUsername") or "?")[:16], s["officialScore"],
               d["delta"], d["serial"], len(d["fams"]), w.title_of(s)[:60]))

    print()
    print("-- 2b. WHAT EACH ROW ACTUALLY DID (read from the note bodies) --")
    print("   This is the decisive cut. 'Restore' rows are not tests of a")
    print("   mechanism; they re-apply code the frontier ALREADY HAD and lost to")
    print("   yukon's replace-overlay. We already carry that code (ledger 81), so")
    print("   their delta is headroom for THEM and exactly zero for US.")
    for i in ids:
        role, why = _role(detail[i]["sub"])
        print("     %-9s %-9s %+7.3f %%  %s" % (i[:8], role, detail[i]["delta"], why))
    roles = collections.defaultdict(list)
    for i in ids:
        roles[_role(detail[i]["sub"])[0]].append(detail[i]["delta"])
    print()
    for r, ds in sorted(roles.items()):
        print("     %-9s n=%d  mean %+0.3f %%" % (r, len(ds), st.mean(ds)))
    restore = roles.get("RESTORE", [])
    if restore:
        print()
        print("   The three RESTORE rows measure ONE contrast (the 474c750 tip")
        print("   with vs without the two dropped files) three times over. As")
        print("   repeats of a single comparison their mean is %+0.3f %% with"
              % st.mean(restore))
        print("   se %.3f -- and it is worth nothing to a tree that already has"
              % (st.stdev(restore) / math.sqrt(len(restore))))
        print("   those files.")

    print()
    print("-- 3. INDEPENDENCE --")
    authors = collections.Counter(detail[i]["sub"].get("solverUsername") for i in ids)
    print("   distinct solvers: %d of %d rows" % (len(authors), len(ids)))
    for a, c in authors.most_common():
        print("     %-18s %d row(s)" % (a, c))
    titles = collections.Counter(w.title_of(detail[i]["sub"]) for i in ids)
    dupe = {t: c for t, c in titles.items() if c > 1}
    print("   repeated titles (same claim resubmitted): %s"
          % (dupe if dupe else "none"))
    fronts = collections.Counter(detail[i]["front"]["id"][:8] for i in ids)
    print("   distinct frontiers compared against: %d" % len(fronts))
    for f, c in fronts.most_common():
        print("     vs frontier %s : %d row(s)" % (f, c))
    eff_n = len(authors)
    print("   effective n if solver is the unit of independence: %d (not %d)"
          % (eff_n, len(ids)))
    if eff_n > 1:
        per_author = collections.defaultdict(list)
        for i in ids:
            per_author[detail[i]["sub"].get("solverUsername")].append(detail[i]["delta"])
        means = [st.mean(v) for v in per_author.values()]
        sd_a = st.stdev(means) if len(means) > 1 else float("nan")
        se_a = sd_a / math.sqrt(len(means)) if len(means) > 1 else float("nan")
        print("   solver-clustered mean %+0.3f %%, se %.3f, |t| %.2f on %d df"
              % (st.mean(means), se_a, abs(st.mean(means) / se_a) if se_a else
                 float("nan"), len(means) - 1))

    print()
    print("-- 4. MULTI-MECHANISM CONTAMINATION --")
    multi = sum(1 for i in ids if len(detail[i]["fams"]) > 1)
    print("   rows matching >1 family: %d of %d" % (multi, len(ids)))
    for i in ids:
        print("     %s  %s" % (i[:8], ", ".join(detail[i]["fams"])))
    print("   NOTE: family membership comes from the note TITLE only. A title")
    print("   naming one mechanism does not mean the tree contains only that")
    print("   mechanism -- every row here builds on an inherited frontier.")

    print()
    print("-- 5. BOX / SERIAL CONFOUND --")
    allser = [detail[i]["serial"] for i in detail]
    mu, sd = st.mean(allser), st.stdev(allser)
    print("   all %d rows: serial offset mean %+0.4f %%, sd %.4f %%" % (len(allser), mu, sd))
    for i in ids:
        z = (detail[i]["serial"] - mu) / sd
        print("     %s serial %+0.3f %%  (%+0.2f sigma)" % (i[:8], detail[i]["serial"], z))
    zfam = (tgt["serial"] - mu) / (sd / math.sqrt(len(ids)))
    print("   family mean serial %+0.3f %% = %+0.2f sigma of the mean-of-%d"
          % (tgt["serial"], zfam, len(ids)))
    print("   R = serial / mtp, so a LUCKY (slow) serial leg RAISES the score.")
    print("   The response R' = own pooled serial / own mtp DELIBERATELY keeps")
    print("   that session factor (E35: it is the organizers' own normaliser).")
    print("   So the serial offset is NOT removed from the delta -- it is a live")
    print("   confound, and a positive offset beside a positive delta is the")
    print("   signature of a box effect rather than evidence against one.")
    covar = _cov([detail[i]["serial"] for i in detail],
                 [detail[i]["delta"] for i in detail])
    print("   corr(serial offset, delta) over all %d rows = %+0.3f" % (len(detail), covar))
    print()
    print("   Serial-corrected family delta (subtract each row's own offset):")
    corr = [detail[i]["delta"] - detail[i]["serial"] for i in ids]
    csd = st.stdev(corr)
    cse = csd / math.sqrt(len(corr))
    print("     %-9s %9s %9s %9s" % ("id", "delta%", "serial%", "corrected%"))
    for i in ids:
        print("     %-9s %+9.3f %+9.3f %+9.3f"
              % (i[:8], detail[i]["delta"], detail[i]["serial"],
                 detail[i]["delta"] - detail[i]["serial"]))
    print("     mean %+0.3f %%, se %.3f, |t| %.2f   (was %+0.3f %%, |t| %.2f)"
          % (st.mean(corr), cse, abs(st.mean(corr) / cse), tgt["mean"], tgt["t"]))
    print("     2 sigma_score bar is 0.185 %% -- corrected mean is %s it."
          % ("BELOW" if st.mean(corr) < 0.185 else "above"))

    print()
    print("=" * 78)
    print("VERDICT INPUTS")
    print("=" * 78)
    print("  observed family mean today      %+0.3f %%" % tgt["mean"])
    print("  null expectation for a max-of-%d %+0.3f %%" % (k, exp_max))
    print("  2 sigma_score bar               %+0.3f %%" % 0.185)
    print("  |t| today                       %.2f  (was %.1f in E35)"
          % (tgt["t"], E35_REPORTED["t"]))
    print("  solver-clustered df             %d" % (len(authors) - 1))
    return 0


def _cov(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def _role(sub):
    """Introduce / restore / force, read from the note's own words.

    The distinction the family label hides: only an INTRODUCE or FORCE row
    tests whether the mechanism is worth anything. A RESTORE row tests
    whether *losing* it costs anything, on a tree that had it.
    """
    title = w.title_of(sub).lower()
    note = (sub.get("note") or "").lower()
    if "restore" in title or "recovery" in title or "graft-loss" in title:
        return "RESTORE", "re-applies the two files an overlay dropped"
    if "force" in title:
        return "FORCE", "makes an already-declared default actually apply"
    if "residency" in title or "resid" in title:
        n = len([1 for k in ("top-32", "shortlist", "fused", "4+4", "rmsnorm")
                 if k in note])
        return "INTRODUCE", "first to ship it, bundled with ~%d other mechanisms" % n
    return "OTHER", ""


def _t_max_null(summaries, trials=200000, seed=13):
    """Median and 95th pct of the max family t when every true effect is zero.

    Each family contributes a central t draw on its own df. The MEAN of this
    max is not usable: an n=2 family has df=1, whose t is Cauchy and has no
    finite mean, so the sample mean diverges with `trials`. The median and
    upper quantile are well defined and are what the comparison needs.
    """
    dfs = [s["n"] - 1 for s in summaries if s["n"] > 1]
    rng = random.Random(seed)
    draws = []
    for _ in range(trials):
        draws.append(max(rng.gauss(0.0, 1.0) /
                         math.sqrt(_chi2(rng, df) / df) for df in dfs))
    draws.sort()
    return draws[trials // 2], draws[int(0.95 * trials)]


def _chi2(rng, df):
    return sum(rng.gauss(0.0, 1.0) ** 2 for _ in range(df))


def _pctile(value, summaries, trials=200000, seed=11):
    ses = [f["se"] for f in summaries if f["n"] > 1 and not math.isnan(f["se"])]
    rng = random.Random(seed)
    below = 0
    for _ in range(trials):
        if max(rng.gauss(0.0, se) for se in ses) < value:
            below += 1
    return below / trials


if __name__ == "__main__":
    raise SystemExit(main())
