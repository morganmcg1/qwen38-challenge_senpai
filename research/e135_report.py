#!/usr/bin/env python3
"""Report one E135 wide-against-tight palindrome session.

The headline is absolute `mtp_seconds_per_token`. Two estimators are printed:

  paired palindrome   mean(T) - mean(W) inside each `W T T W` replicate. The
                      arm code (-1, +1, +1, -1) is orthogonal to the centred
                      leg index (-1.5, -0.5, +0.5, +1.5), so a linear drift in
                      leg index cancels exactly rather than approximately.

  drift-corrected     one least-squares fit over the whole session with an
                      intercept, the arm contrast and the centred global leg
                      index. The residual standard deviation is the error bar,
                      as the assignment specifies.

The local serial-to-MTP ratio is reported next to it. Unlike the E129 table
arm, this change is confined to the candidate leg: the serial leg decodes at
M = 1, the routed set is 3...9, and `default: break` means the serial leg
launches no routed QMV. The two instruments should therefore agree, and a
disagreement is itself a finding.

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

ARMS = ("wide", "tight")
OUT = pathlib.Path("research/out")


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
        arm = meta.get("e135_grid")
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
    """Fit y ~ 1 + arm + centred leg index. Returns the arm contrast T - W."""
    ys, arms, idxs = [], [], []
    for r in rows:
        y = fnum(r["metrics"].get(key))
        if y is None:
            continue
        ys.append(y)
        arms.append(1.0 if r["arm"] == "tight" else -1.0)
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
    args = ap.parse_args()

    rows = legs(args.label)
    if not rows:
        print(f"e135_report: no legs for label {args.label!r}")
        return 1
    complete = [r for r in rows if r["metrics"].get("mtp_seconds_per_token")]

    print(f"E135 wide-against-tight session {args.label}: {len(rows)} legs, "
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
    print(f"{'tag':32s} {'grid':6s} {'idx':>3s} "
          f"{'mtp s/tok':>10s} {'serial':>9s} {'ratio':>7s} "
          f"{'in C':>6s} {'out C':>6s} {'draft':>7s} {'div':>4s} {'match':>6s}")
    for r in rows:
        m = r["metrics"]
        print(f"{r['tag']:32s} {r['arm']:6s} {r['idx']:3d} "
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
          + ("identical, so the arms differ in launch geometry only"
             if len(drafts) == 1 and len(rates) == 1
             else "NOT IDENTICAL, THE ARMS ARE CONFOUNDED"))
    print()

    print("per arm")
    print(f"{'grid':6s} {'n':>2s} {'mtp mean':>10s} {'sd':>9s} "
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
        print(f"{arm:6s} {len(rs):2d} {statistics.fmean(mtp):10.6f} "
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
        pct = 100 * (per["wide"] - per["tight"]) / per["wide"]
        pairs.append(pct)
        print(f"  rep {rep}: wide {per['wide']:.6f}  tight {per['tight']:.6f}"
              f"   tight is {pct:+.3f} % faster")
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
              f"tight - wide {fit['contrast']:+.6f} +- {fit['se']:.6f}  "
              f"({pct:+.3f} +- {pct_se:.3f} %)  "
              f"tight is {sign:+.3f} % {better}")
        print(f"  {'':26s} drift {fit['drift_per_leg']:+.6f} per leg, "
              f"residual sd {fit['sigma']:.6f}, dof {fit['dof']}")
    print()

    fit = ols_arm_and_drift(complete, "mtp_seconds_per_token")
    if fit:
        headline = -100 * fit["contrast"] / fit["mean"]
        err = 100 * fit["se"] / fit["mean"]
        print(f"HEADLINE e135_tight_grid_candidate_leg_pct = {headline:+.4f} "
              f"(+- {err:.4f}), positive means tight is faster")
        if headline < -0.10:
            verdict = ("STOP AND REPORT: tight is more than 0.10 % SLOWER. "
                       "The extra threadgroups are doing something useful.")
        elif headline < 0.10:
            verdict = ("The launch hypothesis is dead at the local fixture: "
                       "the contrast is inside +-0.10 %.")
        else:
            verdict = "At or above the +0.10 % gate; rung 2 is licensed."
        print(f"STOP RULE: {verdict}")
    print()
    print("This session is ungated and counterbalanced. It is directional "
          "causal evidence inside itself, not a gate-qualified reading and "
          "not any kind of official score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
