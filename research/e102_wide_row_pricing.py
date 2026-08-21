#!/usr/bin/env python3
"""E102: price the QMV wide-row (NA >= 5) mechanism at the RANKED level.

Question
    Eleven upstream trees put five or more input rows through one shared weight
    pass in the scored ``out_vec_size >= 4096`` affine QMV dispatch. Does the
    ranked board show a width-INCREASING cost from that change, or only a flat
    level shift, or nothing?

Method
    A contrast is only a round-cost contrast when the two runs ran the SAME
    draft trajectory. Cohort membership therefore requires the eight
    ``effective_mean_draft_len`` values to be digit identical (exact float
    equality), exactly as ``research/board_same_schedule.py`` selects the crown
    cohort. Inside such a cohort the per-prompt round count R is identical too,
    so a per-prompt candidate-seconds-per-token delta is a pure per-round cost
    delta with no schedule confound.

    Round time needs the per-prompt round count R, which the board does not
    publish. ``research/prompt_round_reconstruction.py`` derives it from the
    exact rational form of ``effective_mean_draft_len`` plus window closure,
    the non-drafting census and the depth-0 round-cost floor. ``control``
    proves that this reproduces all eight hard-coded crown round counts in
    ``board_same_schedule.py``. R is resolved ONCE per cohort from the target
    receipt and reused for every member, so a timing difference can never leak
    into the denominator.

    Shape is read from the centered fit over the five high-width prompts,

        round_us(M) = L + S * (M - Mbar),   M = 1 + effective_mean_draft_len

    never the raw intercept, for the leverage reason given in
    ``board_same_schedule.py``. ``L/plutarch`` is the mode classifier of
    ``research/e87_s15_level_slope.py``: the 7-prompt raw-intercept level
    divided by the plutarch round time, with plutarch held out as the ~92 %
    non-drafting probe of machine speed.

Controls
    ``qmv``   labels every cohort member by the blob identity of the five
              scored QMV files against reference tree 9b241879, whose kernel
              table is byte identical to the current campaign base. A member
              with BASE kernels is a clean control for a wide-row tree.
    ``noise`` measures the per-run noise floor from board runs whose whole git
              tree hash is identical, i.e. the same bytes measured twice.

Usage
    python3 research/board_per_prompt.py fetch       # refresh /tmp/yukon-board
    bash research/e102_kernel_fingerprint.sh         # refresh /tmp/e102_kernel_fp.tsv
    python3 research/e102_wide_row_pricing.py control
    python3 research/e102_wide_row_pricing.py noise
    python3 research/e102_wide_row_pricing.py cohort <id-prefix>
    python3 research/e102_wide_row_pricing.py price <target> [<sibling>]
    python3 research/e102_wide_row_pricing.py all
"""

import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt_round_reconstruction as prr  # noqa: E402

CACHE = "/tmp/yukon-board/full.json"
FP = "/tmp/e102_kernel_fp.tsv"
TREES = "/tmp/e102_trees.txt"
T = 512
BASE_REF = "9b241879"          # kernel table byte identical to the campaign base
FRONTIER = "f04b102e"

PROMPTS = {
    "919318e1": "beagle", "192fb621": "botany", "4b9e88cd": "drama",
    "a2ea8b60": "essays", "00142a44": "medicine", "c1ec5866": "plutarch",
    "ea82dcb5": "republic", "3b10cb4d": "travel",
}
CROWN_ROUNDS = {"plutarch": 487, "drama": 252, "travel": 212, "beagle": 110,
                "republic": 93, "essays": 92, "medicine": 90, "botany": 81}
CROWN_ID = "8819b108"
LOCAL_ACCEPT = 0.8770161290322581   # e87_s15 calibration constant

QMV_FILES = [
    "Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized.cpp",
    "Vendor/mlx-swift/Source/Cmlx/mlx-generated/metal/quantized.h",
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/quantized.cpp",
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.h",
    "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/quantized.metal",
]
MTP_DIR = "Sources/MLXFastModel"

TARGETS = [
    ("ca9251b8", "shared _wide widened to NA<=5; M5 IPG=5, M9 IPG=5; rps=4"),
    ("2da69933", "NA<=6; M5 IPG=5, M6 IPG=6, M9 IPG=5; rps=4"),
    ("ff73cbbd", "NA<=6; M5 IPG=5, M6 IPG=6, M9 IPG=5; rps=4"),
    ("3ff80e86", "separate _wideN NA<=9; M5 IPG=5, M9 IPG=5; shared _wide NA<=4"),
    ("596761ef", "qmv_fast_weightstat_affine4_g64 IPG=5 at M5"),
    ("0741d679", "private onestream_m5 / onestream_m6 kernels"),
    ("afb688fe", "m9_fivefour_direct: 5+4, five-row group rows_per_simd=2"),
    ("e617ef07", "m9_fivefour_direct: 5+4, five-row group rows_per_simd=2"),
    ("60a5ac1f", "m8_fivethree_direct + m9_sixthree_direct, rows_per_simd=2"),
]


# ----------------------------------------------------------------- loading

def load_rows():
    payload = json.load(open(CACHE))
    rows = payload
    if isinstance(rows, dict):
        for key in ("submissions", "rows", "data", "items"):
            if isinstance(rows.get(key), list):
                rows = rows[key]
                break
    return [r for r in rows if isinstance(r, dict)]


def scored_rows():
    out = []
    for r in load_rows():
        per = (r.get("officialMetrics") or {}).get("per_prompt") or []
        if len(per) != 8 or r.get("officialScore") is None:
            continue
        table = {}
        ok = True
        for e in per:
            name = PROMPTS.get(e["prompt_sha256"][:8])
            if name is None or e.get("mtp_seconds_per_token_mean") in (None, 0):
                ok = False
                break
            table[name] = e
        if ok and len(table) == 8:
            r["_t"] = table
            out.append(r)
    return out


def fingerprints():
    if not os.path.exists(FP):
        return {}
    out = {}
    for line in open(FP):
        parts = line.rstrip("\n").split("\t")
        rec = {}
        for kv in parts[1:]:
            path, _, obj = kv.partition("=")
            rec[path] = obj
        out[parts[0][:8]] = rec
    return out


def tree_hashes():
    if not os.path.exists(TREES):
        return {}
    out = {}
    for line in open(TREES):
        ref, _commit, tree = line.split()
        out[ref.rsplit("/", 1)[-1][:8]] = tree
    return out


# ------------------------------------------------------- round reconstruction

def cohort_rounds(cohort):
    """Modal per-prompt R over a whole same-schedule cohort.

    The C4 step of the reconstruction resolves a remaining ambiguity with the
    run's own measured round cost, so a much slower run can land on a different
    multiple of the same reduced denominator. Inside a digit-identical schedule
    cohort R cannot depend on speed, so take the mode over the cohort. That
    also stops a target's own timing from entering its denominator.
    """
    votes = {}
    for row in cohort:
        try:
            R, _amb, _res = round_table(row)
        except RuntimeError:
            continue
        for name, value in R.items():
            votes.setdefault(name, []).append(value)
    if not votes or len(votes) != 8:
        return None
    return {name: st.mode(vals) for name, vals in votes.items()}


def round_table(row):
    """Per-prompt round count R through the validated C1-C4 reconstruction."""
    feed = {}
    for name, e in row["_t"].items():
        feed[name] = {"mean_draft_len": e["effective_mean_draft_len"],
                      "non_drafting_rounds": e["non_drafting_round_count"],
                      "mtp_spt": e["mtp_seconds_per_token_mean"],
                      "serial_spt": e["serial_seconds_per_token_mean"],
                      "raw_ratio": e["raw_ratio_of_means"]}
    res = prr.reconstruct(feed)["prompts"]
    R = {n: p["rounds"] for n, p in res.items()}
    amb = [n for n, p in res.items() if not p["unique_under_c1_c4"]]
    return R, amb, res


# ----------------------------------------------------------------- statistics

def groups_of(row):
    """High-width group = the five prompts with the largest draft length.

    ``board_same_schedule.py`` hard-codes that set for the crown schedule; this
    reproduces the same rule from the data and flags any run whose membership
    differs from the canonical set.
    """
    order = sorted(row["_t"], key=lambda n: -(row["_t"][n]["effective_mean_draft_len"] or 0))
    return order[:5], order[5:]


def centered_fit(row, R, high):
    ms = [1.0 + row["_t"][n]["effective_mean_draft_len"] for n in high]
    ys = [T * row["_t"][n]["mtp_seconds_per_token_mean"] / R[n] * 1e6 for n in high]
    mbar = sum(ms) / len(ms)
    sxx = sum((m - mbar) ** 2 for m in ms)
    L = sum(ys) / len(ys)
    S = sum((m - mbar) * (y - L) for m, y in zip(ms, ys)) / sxx
    resid = [y - (L + S * (m - mbar)) for m, y in zip(ms, ys)]
    sig = math.sqrt(sum(e * e for e in resid) / (len(ys) - 2))
    return L, S, mbar, sxx, sig / math.sqrt(sxx)


def mode_classifier(row, accept=LOCAL_ACCEPT):
    """e87_s15 L/plutarch: 7-prompt raw-intercept level over plutarch round time."""
    xs, ys, plut = [], [], None
    for name, e in row["_t"].items():
        d = e["effective_mean_draft_len"] or 0.0
        round_us = e["mtp_seconds_per_token_mean"] * (1.0 + accept * d) * 1e6
        if name == "plutarch":
            plut = round_us
            continue
        xs.append(1.0 + d)
        ys.append(round_us)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return (my - slope * mx) / plut


def serial_means(scored):
    return {name: st.mean([r["_t"][name]["serial_seconds_per_token_mean"]
                           for r in scored])
            for name in PROMPTS.values()}


def serial_free(row, means):
    vals = sorted(means[n] / row["_t"][n]["mtp_seconds_per_token_mean"]
                  for n in row["_t"])
    return 0.5 * (vals[3] + vals[4])


def sched_key(row):
    return tuple(row["_t"][n]["effective_mean_draft_len"]
                 for n in sorted(PROMPTS.values()))


# ----------------------------------------------------------------- commands

def pick(scored, prefix):
    hits = [r for r in scored if r["id"].startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("prefix %r matched %d scored rows" % (prefix, len(hits)))
    return hits[0]


def qmv_label(fp, ident):
    base = fp.get(BASE_REF)
    rec = fp.get(ident[:8])
    if not base or not rec:
        return "?"
    diff = [p for p in QMV_FILES if rec.get(p) != base.get(p)]
    if not diff:
        return "BASE-qmv"
    short = {"mlx-generated/quantized.cpp": "gen.cpp",
             "mlx-generated/metal/quantized.h": "gen.h",
             "backend/metal/quantized.cpp": "disp",
             "kernels/quantized.h": "src.h",
             "kernels/quantized.metal": "src.metal"}
    tags = []
    for p in diff:
        for k, v in short.items():
            if p.endswith(k):
                tags.append(v)
                break
    return "MOD:" + ",".join(tags)


def cmd_control(scored):
    print("=== positive control: recover the crown round counts from public fields ===")
    row = pick(scored, CROWN_ID)
    R, amb, res = round_table(row)
    ok = True
    print("  %-9s %6s %6s %8s %8s" % ("prompt", "R", "crown", "unique", "accept"))
    for name in sorted(CROWN_ROUNDS, key=lambda n: -CROWN_ROUNDS[n]):
        flag = "" if R[name] == CROWN_ROUNDS[name] else "  MISMATCH"
        ok &= R[name] == CROWN_ROUNDS[name]
        print("  %-9s %6d %6d %8s %8.4f%s"
              % (name, R[name], CROWN_ROUNDS[name],
                 res[name]["unique_under_c1_c4"],
                 res[name]["per_draft_accept_rate"], flag))
    print("  prompts needing the C4 round-cost floor to disambiguate: %s"
          % (", ".join(sorted(amb)) or "none"))
    print("  %s" % ("RECOVERY EXACT on all 8 prompts" if ok else "RECOVERY FAILED"))
    high, low = groups_of(row)
    print("  high-width group from data: %s" % " ".join(sorted(high)))
    print("  low-width  group from data: %s" % " ".join(sorted(low)))
    L, S, mbar, _sxx, seS = centered_fit(row, R, high)
    print("  crown centered fit: L %.1f us  S %.1f us/row  se(S) %.1f  Mbar %.4f"
          % (L, S, seS, mbar))
    print("  board_same_schedule.py crown reference: Mbar 6.1723")
    return ok


def cmd_noise(scored):
    """Per-run noise from board runs whose whole git tree is byte identical."""
    trees = tree_hashes()
    by_tree = {}
    for r in scored:
        h = trees.get(r["id"][:8])
        if h:
            by_tree.setdefault(h, []).append(r)
    reps = [v for v in by_tree.values() if len(v) >= 2]
    print("=== replicate noise: byte-identical git trees submitted more than once ===")
    print("  %d scored runs, %d distinct trees, %d trees with n>=2 (%d runs)"
          % (len(scored), len(by_tree), len(reps), sum(len(v) for v in reps)))
    if not reps:
        print("  no byte-identical repeats; no replicate noise floor available")
        return
    means = serial_means(scored)
    acc = {k: [0.0, 0] for k in ("spt", "L", "S", "mode", "sf")}   # ss, dof

    def add(key, values):
        m = st.mean(values)
        if m == 0:
            return
        rel = [100.0 * (v - m) / abs(m) for v in values]
        acc[key][0] += sum(x * x for x in rel)
        acc[key][1] += len(values) - 1

    used = 0
    hilo = []
    for grp in reps:
        try:
            R, _a, _r = round_table(grp[0])
        except Exception:
            continue
        used += 1
        high, _low = groups_of(grp[0])
        for i, g in enumerate(grp):
            rest = grp[:i] + grp[i + 1:]
            d = {}
            for name in PROMPTS.values():
                ref = st.mean([x["_t"][name]["mtp_seconds_per_token_mean"] for x in rest])
                d[name] = 100.0 * (g["_t"][name]["mtp_seconds_per_token_mean"] / ref - 1)
            hilo.append(st.mean([d[n] for n in high])
                        - st.mean([d[n] for n in d if n not in high]))
        for name in PROMPTS.values():
            add("spt", [g["_t"][name]["mtp_seconds_per_token_mean"] for g in grp])
        stats = []
        for g in grp:
            high, _low = groups_of(g)
            L, S, _m, _x, _s = centered_fit(g, R, high)
            stats.append((L, S, mode_classifier(g)))
        add("L", [s[0] for s in stats])
        add("S", [s[1] for s in stats])
        add("mode", [s[2] for s in stats])
        add("sf", [serial_free(g, means) for g in grp])
    print("  pooled within-tree sd (sample, n-1 pooled over %d usable groups):" % used)
    labels = {"spt": "per-prompt candidate s/tok", "L": "centered level L",
              "S": "centered slope S", "mode": "mode classifier L/plutarch",
              "sf": "serial-free score"}
    for key in ("spt", "L", "S", "mode", "sf"):
        ss, dof = acc[key]
        print("    %-27s %6.3f %%   (dof %d)"
              % (labels[key], math.sqrt(ss / dof) if dof else float("nan"), dof))
    print("  a two-run difference has sd = sqrt(2) x the single-run sd above")
    print("  WIDTH SHAPE noise floor: high minus low, one run against its own "
          "byte-identical repeats")
    print("    sd %.3f pp   mean %+0.3f pp   max |value| %.3f pp   n = %d"
          % (st.pstdev(hilo), st.mean(hilo), max(abs(x) for x in hilo), len(hilo)))
    print("  groups (same bytes, repeated ranked runs):")
    for grp in sorted(reps, key=lambda g: -len(g)):
        ids = " ".join(g["id"][:8] for g in grp)
        span = "%s..%s" % (min((g.get("createdAt") or "")[:10] for g in grp),
                           max((g.get("createdAt") or "")[:10] for g in grp))
        pub = [g["officialScore"] for g in grp]
        print("    n=%d %-14s %s  pub %.5f..%.5f  %s"
              % (len(grp), str(grp[0].get("solverUsername"))[:14], span,
                 min(pub), max(pub), ids))


def cohort_of(scored, row):
    key = sched_key(row)
    return [r for r in scored if sched_key(r) == key]


def describe(scored, row, means, fp, R=None):
    amb = []
    if R is None:
        try:
            R, amb, _res = round_table(row)
        except RuntimeError as exc:
            amb = ["ROUNDS UNRESOLVED: %s" % exc]
    high, _low = groups_of(row)
    if R:
        L, S, _m, _x, seS = centered_fit(row, R, high)
    else:
        L = S = seS = float("nan")
    return dict(id=row["id"][:8], user=str(row.get("solverUsername"))[:14],
                created=(row.get("createdAt") or "")[:10],
                status=row.get("status"), pub=row["officialScore"],
                sf=serial_free(row, means), L=L, S=S, seS=seS,
                mode=mode_classifier(row), qmv=qmv_label(fp, row["id"]),
                R=R, high=high, amb=amb)


def cmd_cohort(scored, prefix):
    means = serial_means(scored)
    fp = fingerprints()
    row = pick(scored, prefix)
    coh = cohort_of(scored, row)
    R, amb, _res = round_table(row)
    print("=== schedule cohort of %s (%s) : n = %d ==="
          % (row["id"][:8], row.get("solverUsername"), len(coh)))
    print("  %-9s %-14s %-10s %9s %9s %9s %8s %9s  %s"
          % ("id", "user", "created", "serfree", "L us", "S us/row", "L/plut", "pub", "qmv"))
    for r in sorted(coh, key=lambda z: z.get("createdAt") or ""):
        d = describe(scored, r, means, fp, R)
        mark = " <-- target" if r["id"].startswith(prefix) else ""
        print("  %-9s %-14s %-10s %9.5f %9.1f %9.1f %8.4f %9.5f  %s%s"
              % (d["id"], d["user"], d["created"], d["sf"], d["L"], d["S"],
                 d["mode"], d["pub"], d["qmv"], mark))


def _refs():
    out = {}
    for line in open(TREES):
        ref = line.split()[0]
        out[ref.rsplit("/", 1)[-1][:8]] = ref
    return out


REFS = _refs() if os.path.exists(TREES) else {}
_DIFF_CACHE = {}


def diff_files(a, b):
    """Whole-tree file difference between two submission branches."""
    key = tuple(sorted((a[:8], b[:8])))
    if key in _DIFF_CACHE:
        return _DIFF_CACHE[key]
    if a[:8] not in REFS or b[:8] not in REFS:
        _DIFF_CACHE[key] = None
        return None
    import subprocess
    out = subprocess.run(["git", "diff", "--name-only", REFS[a[:8]], REFS[b[:8]]],
                         capture_output=True, text=True, check=True).stdout
    files = [x for x in out.splitlines() if x]
    _DIFF_CACHE[key] = files
    return files


def control_set(scored, row, fp):
    """Tiered control group inside the same-schedule cohort.

    Tier A is the only unconfounded contrast: the sibling runs the unmodified
    BASE QMV kernel and its tree is otherwise byte identical to the target, so
    the QMV kernel is the single difference. Tier B keeps the solver and the
    schedule but admits other differing files. Tier C is the whole BASE-qmv
    cohort and is a population backdrop, not a sibling.
    """
    coh = [r for r in cohort_of(scored, row) if r["id"] != row["id"]]
    base = [r for r in coh if qmv_label(fp, r["id"]) == "BASE-qmv"]
    mine = fp.get(row["id"][:8], {})
    same_mtp = [r for r in base
                if fp.get(r["id"][:8], {}).get(MTP_DIR) == mine.get(MTP_DIR)]
    tier_a = []
    for r in same_mtp:
        files = diff_files(row["id"], r["id"])
        if files is not None and all(f in QMV_FILES for f in files):
            tier_a.append(r)
    tier_b = [r for r in base if r.get("solverUsername") == row.get("solverUsername")]
    out = []
    if tier_a:
        out.append(("A", tier_a))
    if len(same_mtp) >= 3:
        out.append(("A2", same_mtp))
    if tier_b:
        out.append(("B", tier_b))
    if len(base) >= 3:
        out.append(("C", base))
    return out


def _stamp(row):
    s = (row.get("createdAt") or "")[:19].replace("T", " ")
    try:
        import datetime
        return datetime.datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


TIER_TEXT = {
    "A": "A   same schedule, tree identical outside the QMV kernel  UNCONFOUNDED",
    "A2": "A2  same schedule and MLXFastModel subtree, BASE qmv      near-clean",
    "B": "B   same schedule and solver, other files also differ     CONFOUNDED",
    "C": "C   same-schedule BASE-qmv population, unrelated trees    CONFOUNDED",
    "X": "X   sibling named on the command line",
}


def cmd_price(scored, prefix, sib_prefix=None, note=""):
    means = serial_means(scored)
    fp = fingerprints()
    row = pick(scored, prefix)
    coh = cohort_of(scored, row)
    R = cohort_rounds(coh)
    a = describe(scored, row, means, fp, R)
    print()
    print("=== %s  %s ===" % (a["id"], note or ""))
    print("  target  %s %-14s %s  serfree %.5f  pub %.5f  status %-9s qmv %s"
          % (a["id"], a["user"], a["created"], a["sf"], a["pub"],
             a["status"], a["qmv"]))
    print("  cohort n = %d (draft length digit-identical on all 8 prompts, target included)"
          % len(coh))
    if sib_prefix:
        sets = [("X", [pick(scored, sib_prefix)])]
    else:
        sets = control_set(scored, row, fp)
    if not sets:
        print("  no same-schedule control exists: EVIDENCE ABSENT")
        return []
    return [_contrast(row, a, ctrl, tier, R, means, note) for tier, ctrl in sets]


def _contrast(row, a, ctrl, tier, R, means, note):
    print("  ---- control tier %s   n = %d" % (TIER_TEXT[tier], len(ctrl)))
    for c in sorted(ctrl, key=lambda z: z.get("createdAt") or "")[:8]:
        files = diff_files(row["id"], c["id"])
        print("    %s %-14s %s  serfree %.5f  files differing %s"
              % (c["id"][:8], str(c.get("solverUsername"))[:14],
                 (c.get("createdAt") or "")[:10], serial_free(c, means),
                 "?" if files is None else len(files)))
    if len(ctrl) > 8:
        print("    ... %d more control runs" % (len(ctrl) - 8))

    high = a["high"]
    same_sched = all(sched_key(c) == sched_key(row) for c in ctrl)

    def gmean(values):
        return math.exp(sum(math.log(v) for v in values) / len(values))

    deltas = {}
    print("  %-9s %6s %14s %14s %10s %8s  %s"
          % ("prompt", "M", "ctrl s/tok", "tgt s/tok", "delta %", "se pp", "group"))
    for name in sorted(PROMPTS.values(),
                       key=lambda n: -(row["_t"][n]["effective_mean_draft_len"] or 0)):
        vals = [c["_t"][name]["mtp_seconds_per_token_mean"] for c in ctrl]
        ref = gmean(vals)
        y = row["_t"][name]["mtp_seconds_per_token_mean"]
        deltas[name] = 100.0 * (y / ref - 1.0)
        spread = (100.0 * st.stdev([v / ref for v in vals]) / math.sqrt(len(vals))
                  if len(vals) > 1 else float("nan"))
        print("  %-9s %6.3f %14.8f %14.8f %+10.4f %8.3f  %s"
              % (name, 1 + (row["_t"][name]["effective_mean_draft_len"] or 0),
                 ref, y, deltas[name], spread, "HIGH" if name in high else "low"))
    hi = [deltas[n] for n in high]
    lo = [deltas[n] for n in deltas if n not in high]
    print("  high-width mean %+0.4f %% (n=%d, sd %.4f)   low-width mean %+0.4f %% (n=%d, sd %.4f)"
          % (st.mean(hi), len(hi), st.stdev(hi), st.mean(lo), len(lo), st.stdev(lo)))
    print("  all-8 mean      %+0.4f %%      high minus low %+0.4f pp"
          % (st.mean(list(deltas.values())), st.mean(hi) - st.mean(lo)))

    # Null distribution for the same statistics: price each control member
    # against the rest of the control group. Any spread here is noise plus
    # whatever residual tree variation the tier admits, never the mechanism.
    null_hilo, null_L, null_S, null_mode = [], [], [], []
    if len(ctrl) >= 3:
        for i, c in enumerate(ctrl):
            rest = ctrl[:i] + ctrl[i + 1:]
            d = {}
            for name in PROMPTS.values():
                ref = gmean([x["_t"][name]["mtp_seconds_per_token_mean"] for x in rest])
                d[name] = 100.0 * (c["_t"][name]["mtp_seconds_per_token_mean"] / ref - 1.0)
            null_hilo.append(st.mean([d[n] for n in high])
                             - st.mean([d[n] for n in d if n not in high]))
            if R and same_sched:
                cl, cs, _m, _x, _s = centered_fit(c, R, high)
                rl = [centered_fit(x, R, high)[:2] for x in rest]
                null_L.append(100 * (cl / st.mean([z[0] for z in rl]) - 1))
                null_S.append(100 * (cs / st.mean([z[1] for z in rl]) - 1))
            null_mode.append(100 * (mode_classifier(c)
                                    / st.mean([mode_classifier(x) for x in rest]) - 1))

    if R and same_sched:
        cl = [centered_fit(c, R, high)[:2] for c in ctrl]
        Lc, Sc = st.mean([z[0] for z in cl]), st.mean([z[1] for z in cl])
        dL, dS = 100 * (a["L"] / Lc - 1), 100 * (a["S"] / Sc - 1)
        print("  level  L  ctrl %9.1f -> tgt %9.1f us      %+7.3f %%%s"
              % (Lc, a["L"], dL, _z(dL, null_L)))
        print("  slope  S  ctrl %9.1f -> tgt %9.1f us/row  %+7.3f %%%s"
              % (Sc, a["S"], dS, _z(dS, null_S)))
    else:
        dL = dS = float("nan")
        print("  level and slope not computed: round counts unresolved or schedule differs")
    mc = st.mean([mode_classifier(c) for c in ctrl])
    dmode = 100 * (a["mode"] / mc - 1)
    print("  mode   L/plutarch ctrl %.4f -> tgt %.4f       %+7.3f %%%s"
          % (mc, a["mode"], dmode, _z(dmode, null_mode)))
    hl = st.mean(hi) - st.mean(lo)
    print("  WIDTH SHAPE  high minus low %+0.3f pp%s" % (hl, _z(hl, null_hilo)))
    print("               delta on M, all 8 prompts %+0.4f pp per row;"
          " inside the high group %+0.4f pp per row"
          % (_slope(row, deltas, list(deltas)), _slope(row, deltas, high)))
    return dict(target=a, ctrl=ctrl, tier=tier, deltas=deltas, high=high,
                note=note, dL=dL, dS=dS, dmode=dmode, hl=hl,
                z_hl=_zval(hl, null_hilo), z_dS=_zval(dS, null_S))


def _slope(row, deltas, names):
    """OLS slope of the per-prompt delta on M, in percentage points per row."""
    xs = [1.0 + row["_t"][n]["effective_mean_draft_len"] for n in names]
    ys = [deltas[n] for n in names]
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def _zval(value, null):
    if len(null) < 3:
        return float("nan")
    sd = st.stdev(null)
    return value / sd if sd else float("nan")


def _z(value, null):
    z = _zval(value, null)
    if z != z:
        return "   (no null: control n < 3)"
    return "   null sd %.3f, z = %+.2f" % (st.stdev(null), z)


def cmd_all(scored):
    out = []
    for prefix, note in TARGETS:
        if not [r for r in scored if r["id"].startswith(prefix)]:
            print("\n=== %s  %s ===" % (prefix, note))
            print("  NOT SCORED on the board: no per-prompt rows, so this tree "
                  "carries NO ranked evidence")
            continue
        out.extend(cmd_price(scored, prefix, None, note))
    print()
    print("=== summary: one row per target and control tier ===")
    print("  %-9s %-4s %5s %8s %8s %9s %7s %9s %7s %9s"
          % ("target", "tier", "nctl", "hi %", "lo %", "hi-lo pp", "z", "dS %",
             "z", "dmode %"))
    for rec in out:
        hi = st.mean([rec["deltas"][n] for n in rec["high"]])
        lo = st.mean([rec["deltas"][n] for n in rec["deltas"] if n not in rec["high"]])
        print("  %-9s %-4s %5d %+8.3f %+8.3f %+9.3f %+7.2f %+9.3f %+7.2f %+9.3f"
              % (rec["target"]["id"], rec["tier"], len(rec["ctrl"]),
                 hi, lo, rec["hl"], rec["z_hl"], rec["dS"], rec["z_dS"],
                 rec["dmode"]))
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "control"
    scored = scored_rows()
    print("scored board runs with 8 per-prompt rows: %d" % len(scored))
    if cmd == "control":
        cmd_control(scored)
    elif cmd == "noise":
        cmd_noise(scored)
    elif cmd == "cohort":
        for p in sys.argv[2:]:
            cmd_cohort(scored, p)
    elif cmd == "price":
        cmd_price(scored, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "all":
        cmd_all(scored)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
