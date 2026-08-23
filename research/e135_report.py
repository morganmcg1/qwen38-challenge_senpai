#!/usr/bin/env python3
"""Report one E135 reference-against-candidate palindrome session.

`--session` selects the arm pair; see `SESSIONS`.

The headline is absolute `mtp_seconds_per_token`. Two estimators are printed:

  paired palindrome   mean(C) - mean(R) inside each `R C C R` replicate. The
                      arm code (-1, +1, +1, -1) is orthogonal to the centred
                      leg index (-1.5, -0.5, +0.5, +1.5), so a linear drift in
                      leg index cancels exactly rather than approximately.

  drift-corrected     one least-squares fit over the whole session with an
                      intercept, the arm contrast and the centred global leg
                      index. The residual standard deviation is the error bar,
                      as the assignment specifies.

The local serial-to-MTP ratio is reported next to it. Both session kinds are
confined to the candidate leg, so the two instruments should agree and a
disagreement is itself a finding. The session script states the confinement
argument for its own change.

The session is ungated by construction. Entry and exit temperature per leg and
the entry spread per arm are printed, and the gate labels are reproduced
verbatim, because an ungated reading is directional evidence inside its own
counterbalanced session and nothing more.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics

OUT = pathlib.Path("research/out")

# Every E135 palindrome session has the same shape: one reference arm, one
# candidate arm, `R C C R` inside each replicate, and one meta key naming the
# arm. Only those three facts change between sessions, so they live here and
# the estimators stay shared.
SESSIONS = {
    "grid": {
        "meta_key": "e135_grid",
        "arms": ("wide", "tight"),
        "headline": "e135_tight_grid_candidate_leg_pct",
        "invariance": "the arms differ in launch geometry only",
    },
    "e87": {
        "meta_key": "e135_e87_arm",
        "arms": ("incumbent", "select"),
        "headline": "e135_e87_select_candidate_leg_pct",
        "invariance": "the arms differ in probe selection only",
        "per_round": True,
    },
    # T29-A has three arms, so it is reported as three pairwise contrasts by
    # `e135_composition_report.py`. Each arm sits at mean position 3.5 in the
    # palindrome, so every pair stays balanced against a linear drift when the
    # third arm is filtered out. The arms here name the pair the advisor asked
    # for; the reporter overrides them for the other two contrasts.
    "composition": {
        "meta_key": "e135_arm",
        "arms": ("base", "composed"),
        "headline": "e135_composition_local_pct",
        "invariance": ("the arms move the table, grid, probe fraction and"
                       " width-2 route together, so a schedule change is"
                       " expected only through the probe fraction"),
    },
}

# The local fixture drafts deeper than ranked beagle, so the same fixed
# per-round saving is a smaller fraction of the local round. Regressing round
# time on depth over the 7 drafting prompts of `572b2cc4` gives
# 51.292 ms + 1.757 ms per row, so a round is 84 % fixed cost and the
# correction is small. `1760479a` gives x1.0630, so the factor is stable.
# See `research/e135_board_contrast.py --round-model` and `--shapes`.
BEAGLE_TOKENS_PER_ROUND = 5.3818
LOCAL_TO_BEAGLE_ROUND = 1.0572
PER_ROUND_BAR_PCT = 1.4097

ARMS = SESSIONS["grid"]["arms"]
META_KEY = SESSIONS["grid"]["meta_key"]
HEADLINE = SESSIONS["grid"]["headline"]
INVARIANCE = SESSIONS["grid"]["invariance"]
PER_ROUND = SESSIONS["grid"].get("per_round", False)


def configure(kind: str) -> None:
    """Point the module at one session kind. Importers call this first."""
    global ARMS, META_KEY, HEADLINE, INVARIANCE, PER_ROUND
    spec = SESSIONS[kind]
    ARMS = spec["arms"]
    META_KEY = spec["meta_key"]
    HEADLINE = spec["headline"]
    INVARIANCE = spec["invariance"]
    PER_ROUND = spec.get("per_round", False)


def read_meta(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def legs(label: str) -> list[dict]:
    found = []
    for d in sorted(OUT.glob(f"e135{label}k*")):
        meta = read_meta(d / "meta.txt")
        arm = meta.get(META_KEY)
        if arm not in ARMS:
            continue
        metrics = {}
        score = d / "score.json"
        if score.exists():
            try:
                metrics = json.loads(score.read_text()).get("metrics", {})
            except json.JSONDecodeError:
                metrics = {}
        found.append({
            "tag": d.name, "arm": arm, "meta": meta, "metrics": metrics,
            "dir": d,
            "cell": meta.get("e135_cell", ""),
            "table": meta.get("e135_table", "onepass67"),
            "rep": int(meta.get("e135_replicate", 0)),
            "pos": int(meta.get("e135_position", 0)),
            "idx": int(meta.get("e135_leg_index", 0)),
        })
    return sorted(found, key=lambda r: (r["rep"], r["pos"]))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def ols_arm_and_drift(rows: list[dict], key: str):
    """Fit y ~ 1 + arm + centred leg index. Returns candidate minus reference."""
    ys, arms, idxs = [], [], []
    for r in rows:
        y = fnum(r["metrics"].get(key))
        if y is None:
            continue
        ys.append(y)
        arms.append(1.0 if r["arm"] == ARMS[1] else -1.0)
        idxs.append(float(r["idx"]))
    n = len(ys)
    if n < 4:
        return None
    ibar = statistics.fmean(idxs)
    ci = [i - ibar for i in idxs]
    abar = statistics.fmean(arms)
    ca = [a - abar for a in arms]
    # Orthogonalise the arm code against the index so the two coefficients are
    # separable even when the design is not perfectly balanced.
    denom_i = sum(v * v for v in ci)
    if denom_i > 0:
        proj = sum(a * i for a, i in zip(ca, ci)) / denom_i
        ca_o = [a - proj * i for a, i in zip(ca, ci)]
    else:
        ca_o = ca
    denom_a = sum(v * v for v in ca_o)
    if denom_a == 0:
        return None
    ybar = statistics.fmean(ys)
    cy = [y - ybar for y in ys]
    beta_a = sum(a * y for a, y in zip(ca_o, cy)) / denom_a
    resid_y = [y - beta_a * a for y, a in zip(cy, ca_o)]
    beta_i = (sum(i * y for i, y in zip(ci, resid_y)) / denom_i
              if denom_i > 0 else 0.0)
    resid = [y - beta_a * a - beta_i * i for y, a, i in zip(cy, ca_o, ci)]
    dof = n - 3
    sigma = math.sqrt(sum(r * r for r in resid) / dof) if dof > 0 else float("nan")
    # `arm` is coded +-1, so the T - W contrast is twice the coefficient.
    contrast = 2.0 * beta_a
    se = 2.0 * sigma / math.sqrt(denom_a) if denom_a > 0 else float("nan")
    return {"contrast": contrast, "se": se, "sigma": sigma, "n": n,
            "dof": dof, "mean": ybar, "drift_per_leg": beta_i}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="s1")
    ap.add_argument("--session", default="grid", choices=sorted(SESSIONS))
    args = ap.parse_args()
    configure(args.session)
    ref, cand = ARMS

    rows = legs(args.label)
    if not rows:
        print(f"e135_report: no legs for label {args.label!r}")
        return 1
    complete = [r for r in rows if r["metrics"].get("mtp_seconds_per_token")]

    print(f"E135 {ref}-against-{cand} session {args.label}: {len(rows)} legs, "
          f"{len(complete)} with a score")
    gate = sorted({r["meta"].get("gate_qualified_for_timing") for r in rows}
                  - {None})
    real = sorted({r["meta"].get("cool_gate_passed_real_gate") for r in rows}
                  - {None})
    print(f"gate_qualified_for_timing={gate} cool_gate_passed_real_gate={real}")
    workers = {r["meta"].get("worker_sha256") for r in rows}
    print("worker_sha256 across the session: "
          + (f"one build {list(workers)[0][:16]}" if len(workers) == 1
             else f"MORE THAN ONE BUILD {sorted(workers)}"))
    commits = {r["meta"].get("base_sha") for r in rows}
    print("commit across the session:        "
          + (f"one commit {list(commits)[0][:16]}" if len(commits) == 1
             else f"MORE THAN ONE COMMIT {sorted(commits)}"))
    print()

    print("per leg")
    print(f"{'tag':32s} {'arm':9s} {'idx':>3s} "
          f"{'mtp s/tok':>10s} {'serial':>9s} {'ratio':>7s} "
          f"{'in C':>6s} {'out C':>6s} {'draft':>7s} {'div':>4s} {'match':>6s}")
    for r in rows:
        m = r["metrics"]
        print(f"{r['tag']:32s} {r['arm']:9s} {r['idx']:3d} "
              f"{fnum(m.get('mtp_seconds_per_token')) or float('nan'):10.6f} "
              f"{fnum(m.get('serial_seconds_per_token')) or float('nan'):9.6f} "
              f"{fnum(m.get('mtp_decode_speedup')) or float('nan'):7.4f} "
              f"{fnum(r['meta'].get('gpu_temp_entry_c')) or float('nan'):6.1f} "
              f"{fnum(r['meta'].get('gpu_temp_exit_c')) or float('nan'):6.1f} "
              f"{fnum(m.get('effective_mean_draft_len')) or float('nan'):7.4f} "
              f"{str(m.get('residual_divergence_count', '?')):>4s} "
              f"{str(m.get('all_tokens_matched', '?')):>6s}")
    print()

    print("schedule invariance")
    drafts = {fnum(r["metrics"].get("effective_mean_draft_len"))
              for r in complete} - {None}
    rates = {fnum(r["metrics"].get("accepted_draft_rate"))
             for r in complete} - {None}
    print(f"  effective_mean_draft_len across scored legs: {sorted(drafts)}")
    print(f"  accepted_draft_rate across scored legs:      {sorted(rates)}")
    print("  verdict: "
          + (f"identical, so {INVARIANCE}"
             if len(drafts) == 1 and len(rates) == 1
             else "NOT IDENTICAL, THE ARMS ARE CONFOUNDED"))
    print()

    print("per arm")
    print(f"{'arm':9s} {'n':>2s} {'mtp mean':>10s} {'sd':>9s} "
          f"{'serial mean':>11s} {'ratio':>8s} {'entry C':>8s} {'spread':>7s}")
    means = {}
    for arm in ARMS:
        rs = [r for r in complete if r["arm"] == arm]
        if not rs:
            continue
        mtp = [fnum(r["metrics"]["mtp_seconds_per_token"]) for r in rs]
        ser = [fnum(r["metrics"].get("serial_seconds_per_token")) or 0 for r in rs]
        rat = [fnum(r["metrics"].get("mtp_decode_speedup")) or 0 for r in rs]
        ent = [e for e in (fnum(r["meta"].get("gpu_temp_entry_c")) for r in rs)
               if e is not None]
        means[arm] = statistics.fmean(mtp)
        print(f"{arm:9s} {len(rs):2d} {statistics.fmean(mtp):10.6f} "
              f"{(statistics.stdev(mtp) if len(mtp) > 1 else 0):9.6f} "
              f"{statistics.fmean(ser):11.6f} {statistics.fmean(rat):8.4f} "
              f"{(statistics.fmean(ent) if ent else float('nan')):8.1f} "
              f"{((max(ent) - min(ent)) if len(ent) > 1 else 0):7.1f}")
    print()

    print("paired inside each palindrome, arm code orthogonal to leg index")
    pairs = []
    for rep in sorted({r["rep"] for r in complete}):
        per = {}
        for arm in ARMS:
            v = [fnum(r["metrics"]["mtp_seconds_per_token"])
                 for r in complete if r["rep"] == rep and r["arm"] == arm]
            if v:
                per[arm] = statistics.fmean(v)
        if len(per) < 2:
            continue
        pct = 100 * (per[ref] - per[cand]) / per[ref]
        pairs.append(pct)
        print(f"  rep {rep}: {ref} {per[ref]:.6f}  {cand} {per[cand]:.6f}"
              f"   {cand} is {pct:+.3f} % faster")
    if pairs:
        sd = statistics.stdev(pairs) if len(pairs) > 1 else float("nan")
        print(f"  mean {statistics.fmean(pairs):+.3f} % over {len(pairs)} "
              f"palindrome(s), sd {sd:.3f}")
    print()

    print("least squares over the whole session, y ~ 1 + arm + centred leg index")
    for key, name in (("mtp_seconds_per_token", "candidate mtp s/token"),
                      ("serial_seconds_per_token", "serial s/token"),
                      ("mtp_decode_speedup", "local serial-to-MTP ratio")):
        fit = ols_arm_and_drift(complete, key)
        if fit is None:
            continue
        pct = 100 * fit["contrast"] / fit["mean"]
        pct_se = 100 * fit["se"] / fit["mean"]
        better = "faster" if key.endswith("seconds_per_token") else "higher"
        sign = -pct if key.endswith("seconds_per_token") else pct
        print(f"  {name:26s} mean {fit['mean']:.6f}  "
              f"{cand} - {ref} {fit['contrast']:+.6f} +- {fit['se']:.6f}  "
              f"({pct:+.3f} +- {pct_se:.3f} %)  "
              f"{cand} is {sign:+.3f} % {better}")
        print(f"  {'':26s} drift {fit['drift_per_leg']:+.6f} per leg, "
              f"residual sd {fit['sigma']:.6f}, dof {fit['dof']}")
    print()

    fit = ols_arm_and_drift(complete, "mtp_seconds_per_token")
    if fit:
        headline = -100 * fit["contrast"] / fit["mean"]
        err = 100 * fit["se"] / fit["mean"]
        print(f"HEADLINE {HEADLINE} = {headline:+.4f} "
              f"(+- {err:.4f}), positive means {cand} is faster")
        if headline < -0.10:
            verdict = (f"STOP AND REPORT: {cand} is more than 0.10 % SLOWER "
                       f"than {ref}.")
        elif headline < 0.10:
            verdict = (f"Dead at the local fixture: the contrast is inside "
                       f"+-0.10 %.")
        else:
            verdict = "At or above the +0.10 % gate."
        print(f"STOP RULE: {verdict}")

        if PER_ROUND:
            edl = statistics.fmean(
                d for d in (fnum(r["metrics"].get("effective_mean_draft_len"))
                            for r in complete) if d is not None)
            beagle_pct = headline * LOCAL_TO_BEAGLE_ROUND
            print()
            print("reading the local percentage as a beagle percentage")
            print(f"  local fixture tokens per round     {1.0 + edl:.4f}")
            print(f"  ranked beagle tokens per round     {BEAGLE_TOKENS_PER_ROUND:.4f}")
            print(f"  depth correction                   x{LOCAL_TO_BEAGLE_ROUND:.4f}")
            print(f"  predicted beagle-equivalent gain   {beagle_pct:+.4f} %"
                  f" (+- {err * LOCAL_TO_BEAGLE_ROUND:.4f})")
            print(f"  bar for a per-round mechanism      "
                  f"+{PER_ROUND_BAR_PCT:.4f} %"
                  f"   -> {100.0 * beagle_pct / PER_ROUND_BAR_PCT:.1f} % of it")
            print("  the saving is one selection step per drafting round, and a"
                  " ranked round is 84 % fixed cost, so a shallower beagle round"
                  " carries the same saving as a slightly larger fraction")
            print("  this is a depth correction only. It does not cross g16s to"
                  " g17s, which Rule 83 still governs.")
    print()
    print("This session is ungated and counterbalanced. It is directional "
          "causal evidence inside itself, not a gate-qualified reading and "
          "not any kind of official score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
