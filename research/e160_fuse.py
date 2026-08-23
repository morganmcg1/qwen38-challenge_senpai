#!/usr/bin/env python3
"""Witness and price the E160 SwiGLU producer fusion.

The mechanism gives the MLP decode path a candidate-owned SwiGLU kernel whose
epilogue emits the wide-QMV chunk-sum table of the activation it just wrote, so
the routed `mlp.down` consumer reads a published table instead of launching
`qwen35_custom_affine4_g64_xsums_v1` on its own. `MLX_E160_SWIGLU_ARM` selects:

  off      MLX's compiled `silu(a) * b`, then the standalone fill
  replica  the candidate kernel with no epilogue, then the standalone fill
  fuse     the candidate kernel with the epilogue, and no standalone fill

`fuse` minus `off` is the shippable contrast. `replica` minus `off` prices
owning the elementwise kernel and `fuse` minus `replica` prices deleting the
fill, so a null headline can be attributed instead of guessed at.

  witness  read one leg's trace and prove which arm the worker ran
  report   the counterbalanced contrast over the timed legs

Every arm computes the same activation bytes and therefore emits the same
tokens and the same draft schedule. The report refuses to price the contrast if
the arms disagree on `effective_mean_draft_len` or on `accepted_draft_rate`,
which is the RULE 179 contamination gate: a schedule change would make the
timing a mixture of cost and answer, not a cost.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys

ROUND_RE = re.compile(r"^mtp-trace: round=(\d+) ")
FIELD_RE = re.compile(r"(\w+)=(-?\d+(?:\.\d+)?)(?=\s|$)")

# Per table-paying round on the current base, measured by the E160 F1 census
# leg (research/out/e160f1hist, 512 tokens, 78 rounds): every transition into a
# round with S = 1 + d >= 4 adds this many hits and fills, and the MLP
# contributes one activation per layer.
FILLS_PER_ROUND = 130
HITS_PER_ROUND = 127
MLP_LAYERS = 64

ARMS = ("off", "replica", "fuse")


def signature(arm: str, fills: int, hits: int, mlp: int) -> tuple[int, int, int]:
    """(sg_cand, xs_fill, xs_hit) expected per table-paying round."""
    if arm == "off":
        return (0, fills, hits)
    if arm == "replica":
        return (mlp, fills, hits)
    return (mlp, fills - mlp, hits + mlp)


def rounds_from_trace(path: pathlib.Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        m = ROUND_RE.match(line)
        if not m:
            continue
        fields = {k: v for k, v in FIELD_RE.findall(line)}
        out.append({k: (float(v) if "." in v else int(v)) for k, v in fields.items()})
    return out


def deltas(rounds: list[dict], key: str) -> list[int]:
    return [b[key] - a[key] for a, b in zip(rounds, rounds[1:])]


def witness(args: argparse.Namespace) -> int:
    path = pathlib.Path(args.trace)
    rounds = rounds_from_trace(path)
    if len(rounds) < 3:
        print(f"only {len(rounds)} traced rounds; the witness needs at least 3")
        return 1

    for key in ("sg_cand", "xs_fill", "xs_hit"):
        if key not in rounds[0]:
            print(f"FAIL the trace carries no {key} counter")
            return 1

    want = signature(args.want, args.fills, args.hits, args.mlp)
    seen = list(
        zip(deltas(rounds, "sg_cand"), deltas(rounds, "xs_fill"),
            deltas(rounds, "xs_hit")))
    # A round at S = 1 + d < 4 pays no table and moves no counter. Those
    # transitions carry no information about the arm, so they are reported and
    # excluded rather than treated as failures.
    paying = [t for t in seen if t != (0, 0, 0)]
    bad = [t for t in paying if t != want]

    print(f"trace          {path}")
    print(f"traced rounds  {len(rounds)}")
    print(f"want {args.want:<8} (sg_cand, xs_fill, xs_hit) = {want} per paying round")
    print(f"paying rounds  {len(paying)} of {len(seen)} transitions")
    print(f"observed       {sorted(set(paying))}")
    if not paying:
        print("  FAIL no table-paying round in this leg, so no arm is witnessed")
        return 1
    for t in sorted(set(bad)):
        print(f"  FAIL {t} != {want}")
    print("OK" if not bad else "FAILED")
    return 1 if bad else 0


def leg(path: pathlib.Path) -> dict:
    score = json.load(open(path / "score.json"))["metrics"]
    meta = dict(
        line.split("=", 1)
        for line in (path / "meta.txt").read_text().splitlines()
        if "=" in line
    )
    return {
        "tag": path.name,
        "arm": meta.get("e160_arm", "?"),
        "rep": int(meta.get("e160_rep", 1)),
        "position": int(meta.get("e160_position", 0)),
        "mtp": score["mtp_seconds_per_token"],
        "serial": score["serial_seconds_per_token"],
        "edl": score["effective_mean_draft_len"],
        "acc": score["accepted_draft_rate"],
        "matched": score["all_tokens_matched"],
        "tokens": score["decode_tokens"],
        "entry_c": meta.get("gpu_temp_entry_c", "?"),
        "exit_c": meta.get("gpu_temp_exit_c", "?"),
        "gated": meta.get("cool_gate_passed_real_gate", "?"),
    }


def load_legs(label: str) -> list[dict]:
    return sorted(
        (leg(p) for p in pathlib.Path("research/out").glob(f"e160{label}k*")
         if (p / "score.json").exists()),
        key=lambda d: (d["rep"], d["position"]),
    )


def require_balanced(legs: list[dict]) -> None:
    """Refuse to fit anything that is not a complete balanced palindrome.

    Each contrast below is read off by projection, which is the least squares
    answer only while the arm indicators stay orthogonal to the drift
    regressor. A session that is still running, or one that lost a leg, breaks
    that silently and an arm coefficient absorbs the drift instead.
    """
    for rep in sorted({d["rep"] for d in legs}):
        sub = [d for d in legs if d["rep"] == rep]
        counts = {a: sum(1 for d in sub if d["arm"] == a) for a in ARMS}
        if len(set(counts.values())) != 1 or min(counts.values()) == 0:
            raise SystemExit(
                f"session k{rep} is not balanced: {counts}. It is unfinished "
                "or it lost a leg; the palindrome fit is not valid on it.")
        positions = sorted(d["position"] for d in sub)
        if positions != list(range(1, len(sub) + 1)):
            raise SystemExit(
                f"session k{rep} has positions {positions}, not a contiguous "
                "palindrome")
        centre = statistics.fmean(positions)
        for a in ARMS:
            mean_pos = statistics.fmean(
                d["position"] for d in sub if d["arm"] == a)
            if abs(mean_pos - centre) > 1e-9:
                raise SystemExit(
                    f"session k{rep} arm {a} has mean position {mean_pos}, not "
                    f"{centre}; the order is not counterbalanced")


def contamination_gate(legs: list[dict]) -> bool:
    """RULE 179. Every arm must emit the same tokens and the same schedule."""
    ok = True
    edls = {round(d["edl"], 9) for d in legs}
    accs = {round(d["acc"], 9) for d in legs}
    print("\n## Contamination gate (RULE 179)")
    print(f"  effective_mean_draft_len  {len(edls)} distinct  {sorted(edls)}")
    print(f"  accepted_draft_rate       {len(accs)} distinct  {sorted(accs)}")
    if len(edls) != 1 or len(accs) != 1:
        print("  FAIL the arms do not share one schedule, so the contrast is "
              "not a pure cost")
        ok = False
    unmatched = [d["tag"] for d in legs if d["matched"] is not True]
    print(f"  all_tokens_matched        "
          f"{'every leg' if not unmatched else 'FAIL ' + str(unmatched)}")
    ok = ok and not unmatched
    tokens = {d["tokens"] for d in legs}
    print(f"  decode_tokens             {sorted(tokens)}")
    ok = ok and len(tokens) == 1
    if ok:
        edl = next(iter(edls))
        acc = next(iter(accs))
        tok = next(iter(tokens))
        rounds = tok / (1.0 + edl * acc)
        print(f"  derived accepted drafts per round  a = {edl * acc:.6f}")
        print(f"  derived rounds over {tok} tokens    {rounds:.3f}")
    return ok


def fit(legs: list[dict]) -> dict:
    require_balanced(legs)
    y = [d["mtp"] for d in legs]
    n = len(legs)
    grand = statistics.fmean(y)
    means = {a: statistics.fmean(d["mtp"] for d in legs if d["arm"] == a)
             for a in ARMS}

    reps = sorted({d["rep"] for d in legs})
    centre = {r: statistics.fmean(d["position"] for d in legs if d["rep"] == r)
              for r in reps}
    x = [d["position"] - centre[d["rep"]] for d in legs]
    drift = (sum(xi * yi for xi, yi in zip(x, y)) / sum(xi * xi for xi in x)
             if any(x) else 0.0)

    def residuals(with_drift: bool) -> list[float]:
        out = []
        for i, d in enumerate(legs):
            pred = means[d["arm"]] + (drift * x[i] if with_drift else 0.0)
            out.append(y[i] - pred)
        return out

    per_arm = n // len(ARMS)
    var_plain = sum(r * r for r in residuals(False)) / (n - len(ARMS))
    var_drift = sum(r * r for r in residuals(True)) / (n - len(ARMS) - 1)
    se_plain = (2.0 * var_plain / per_arm) ** 0.5
    se_drift = (2.0 * var_drift / per_arm) ** 0.5
    return {
        "means": means,
        "grand": grand,
        "drift": drift,
        "se_plain": se_plain,
        "se_drift": se_drift,
        "dof_plain": n - len(ARMS),
        "dof_drift": n - len(ARMS) - 1,
        "per_arm": per_arm,
    }


def report(args: argparse.Namespace) -> int:
    legs = load_legs(args.label)
    if not legs:
        print("no timed legs found")
        return 1

    print("## Legs")
    for d in legs:
        print(f"  k{d['rep']}p{d['position']} {d['arm']:<8} mtp {d['mtp']:.6f}  "
              f"serial {d['serial']:.6f}  edl {d['edl']:.6f}  "
              f"acc {d['acc']:.6f}  entry {d['entry_c']} exit {d['exit_c']} "
              f"gated {d['gated']}")

    clean = contamination_gate(legs)
    stats = fit(legs)
    means = stats["means"]
    base = means["off"]

    print("\n## Candidate MTP seconds per token, harness=local")
    for a in ARMS:
        print(f"  {a:<8} {means[a]:.6f}  n={stats['per_arm']}")
    print(f"  session drift  {stats['drift'] / base * 100.0:+.4f} %/leg")

    print("\n## Contrasts, per cent of the off arm")
    se = stats["se_plain"] / base * 100.0
    sed = stats["se_drift"] / base * 100.0
    for name, a, b in (
        ("fuse - off      (shippable)", "fuse", "off"),
        ("replica - off   (kernel swap)", "replica", "off"),
        ("fuse - replica  (fill removal)", "fuse", "replica"),
    ):
        d = (means[a] - means[b]) / base * 100.0
        # A negative delta is a speedup, so the reported gain flips the sign.
        print(f"  {name:<32} {-d:+.4f} % faster  "
              f"2se {2 * se:.4f} pp ({stats['dof_plain']} dof)  "
              f"drift-fitted 2se {2 * sed:.4f} pp")

    # Per-round and per-fill microseconds, so the fill-removal arm can be
    # compared with the standalone fill's own price. `rounds` comes from the
    # schedule the contamination gate already proved identical across arms.
    edl = legs[0]["edl"]
    acc = legs[0]["acc"]
    tokens = legs[0]["tokens"]
    rounds = tokens / (1.0 + edl * acc)
    round_us = base * tokens / rounds * 1e6
    print(f"\n## Per round, {rounds:.3f} rounds over {tokens} tokens")
    print(f"  off round      {round_us:.1f} us")
    for name, a, b, count in (
        ("fuse - off", "fuse", "off", args.fills_removed),
        ("replica - off", "replica", "off", 0),
        ("fuse - replica", "fuse", "replica", args.fills_removed),
    ):
        us = (means[a] - means[b]) * tokens / rounds * 1e6
        tail = f"  = {us / count:+.3f} us per fill" if count else ""
        print(f"  {name:<16} {us:+.1f} us/round{tail}")
    print(f"  2se            "
          f"{2 * stats['se_plain'] * tokens / rounds * 1e6:.1f} us/round")

    gain = (base - means["fuse"]) / means["fuse"] * 100.0
    print(f"\n## Headline, harness=local")
    print(f"  R_off / R_fuse - 1 = {gain:+.4f} %")
    print(f"  minimum useful effect {args.minimum:+.4f} % with 2 sigma "
          f"clearing zero")
    lo = gain - 2 * se
    hi = gain + 2 * se
    print(f"  interval [{lo:+.4f}, {hi:+.4f}] %")
    verdict = "PROMOTE" if (lo > 0 and gain >= args.minimum) else "STOP"
    print(f"  stop rule verdict: {verdict}")
    print(f"  ranked transfer at k=1.0 (RULE 176 as hypothesis): "
          f"{gain:+.4f} % published gain; at k=1.7: {gain * 1.7:+.4f} %")
    if not clean:
        print("\nCONTAMINATED: the contrast above is not a pure cost.")
        return 1

    srepl = statistics.fmean(d["serial"] for d in legs if d["arm"] == "off")
    sfuse = statistics.fmean(d["serial"] for d in legs if d["arm"] == "fuse")
    print(f"\n  serial null    {(sfuse - srepl) / srepl * 100.0:+.4f} %  "
          f"(M = 1 never reaches the candidate kernel, so this is drift)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("witness")
    w.add_argument("trace")
    w.add_argument("--want", choices=ARMS, required=True)
    w.add_argument("--fills", type=int, default=FILLS_PER_ROUND)
    w.add_argument("--hits", type=int, default=HITS_PER_ROUND)
    w.add_argument("--mlp", type=int, default=MLP_LAYERS)
    w.set_defaults(fn=witness)

    r = sub.add_parser("report")
    r.add_argument("--label", default="fuse")
    r.add_argument("--minimum", type=float, default=0.06)
    r.add_argument("--fills-removed", type=int, default=MLP_LAYERS)
    r.set_defaults(fn=report)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
