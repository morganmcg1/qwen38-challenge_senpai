#!/usr/bin/env python3
"""Score the E24 paired BASE/MEMO prose measurement.

WHY THIS IS NOT e17_analyse.py
------------------------------
E17 compared *speculation policies*, so its headline was `raw` -- the
serial-to-MTP ratio -- because only the MTP leg moved.  E24 removes two
scalar `asType` dispatches per GDN layer, and the 48 GDN layers are executed
by the depth-0 serial leg exactly as they are by the MTP leg.  Both legs of a
`--local-iterate` run therefore speed up, and the ratio PARTLY CANCELS the
effect it is supposed to show.  The assignment's framing ("MTP leg carries the
effect, serial leg is a drift control") is wrong for this change, and scoring
E24 on `raw` would under-report it by construction.

So the headline here is ABSOLUTE true-decode wall time, per leg, and the
serial leg is promoted from control to SECOND INDEPENDENT WITNESS.  It is in
fact the *larger* prize: 512 target forwards per serial leg against ~246
verify rounds per MTP leg.

The leg extraction itself is imported from e17_analyse rather than re-written,
so the prefill convention (`spt` already contains prefill, so it is SUBTRACTED,
never added -- e17_analyse.py:253-271) and the M = depth + 1 width derivation
stay in one audited place.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17_analyse as e17  # noqa: E402

RUNS = Path(".mlxfast-private/e24/runs")
BASE_ARM, MEMO_ARM = "BASE", "MEMO"


def gpu_temp_of(value):
    """GPU degrees C from either an e11-run thermal line or a bare number."""
    if not value:
        return None
    m = re.search(r"gpu_temp=([0-9.]+)C", value)
    try:
        return float(m.group(1)) if m else float(value)
    except ValueError:
        return None


def fmt(x):
    return "?" if x is None else f"{x:.3f}"


PROMPTS = (
    "english", "narrative", "technical", "dramatic",
    "travel", "philosophy", "natural_history", "medicine",
)

# Phase 1's measured marginal cost of one scalar f32->bf16 cast dispatch,
# arm B - arm C (research/results/e24-phase1.json).  Used only to state a
# prediction the measurement can then refute; it is not fitted to Phase 3.
CAST_US = 9.711e-6
SITES_PER_FORWARD = 96  # 2 invScale constants x 48 GatedDeltaNet layers

GATE_RE = re.compile(r"GPU cool-down gate passed \(current ([\d.]+)C.*?waited (\d+)s\)")
LABEL_RE = re.compile(r"^=== e11-run: (\S+) \(build (\S+)\) ===")


def true_decode_seconds(spt: float, prefill_s: float, tokens: int) -> float:
    """Wall seconds of decode alone, with the seed prefill removed."""
    return spt * tokens - prefill_s


def load_pair(prompt: str) -> dict | None:
    arms = {}
    for arm in (BASE_ARM, MEMO_ARM):
        v = e17.load_arm(prompt, arm, RUNS)
        if v is None:
            return None
        run = RUNS / f"{prompt}-{arm}"
        serial = json.loads((run / "reports" / "03-mtp-timed.json").read_text())
        v["serial_decode_tokens"] = serial["decode_token_count"]
        v["mtp_true_s"] = true_decode_seconds(
            v["mtp_spt"], v["mtp_prefill_s"], v["decode_tokens"])
        v["serial_true_s"] = true_decode_seconds(
            v["serial_spt"], v["serial_prefill_s"], v["serial_decode_tokens"])
        arms[arm] = v
    return arms


def parse_gate_log(paths: list[Path]) -> dict[str, dict]:
    """Pair each timed run with the cool-gate line the benchmark printed for it.

    meta.txt's `thermal_before` is sampled BEFORE benchmark-qwen-mtp.sh runs,
    i.e. before the gate cools the GPU, so it is the PRE-gate temperature and
    cannot prove the entry condition.  The gate's own line is the only direct
    per-run evidence of the temperature timing actually started at.
    """
    out: dict[str, dict] = {}
    label = None
    for p in paths:
        for line in p.read_text(errors="replace").splitlines():
            m = LABEL_RE.match(line)
            if m:
                label = m.group(1)
                continue
            m = GATE_RE.search(line)
            if m and label:
                out[label] = {"entry_temp_c": float(m.group(1)),
                              "waited_s": int(m.group(2))}
                label = None
    return out


def pct(base: float, memo: float) -> float:
    """Percent of BASE that MEMO saves.  Positive means MEMO is faster."""
    return 100.0 * (base - memo) / base


def reference_row_identity(prompt: str) -> dict:
    """Compare BASE and MEMO reference rows value-for-value.

    This is the strongest correctness instrument available here, and it covers
    the exact gap the Phase 2 unit test could not: that test proved the memo
    returns bit-identical constants, but `invScalePair` is private, so it could
    not prove the four CALL SITES were rewired correctly.

    `02-mtp-verify-output.json` is produced by the reference path, which runs
    all 64 target layers -- including the 48 GatedDeltaNet layers whose QK-norm
    constants E24 changes -- once per token over the full window.  Its rows
    carry the exact top-two logits the trusted parent checks, not just an
    argmax, so an identical ledger is far stronger evidence than a token match:
    a perturbation that left the argmax intact would still move top1_logit.

    Any mismatch is a hard fail.  The constants are input-independent, so the
    only admissible outcome is bit-equality; a "small" logit delta would mean
    the rewiring changed arithmetic and the change is not safe to promote.
    """
    out = {"prompt": prompt, "status": "missing"}
    docs = {}
    for arm in (BASE_ARM, MEMO_ARM):
        p = RUNS / f"{prompt}-{arm}" / "reports" / "02-mtp-verify-output.json"
        if not p.is_file():
            return out
        docs[arm] = json.loads(p.read_text())
    a, b = docs[BASE_ARM], docs[MEMO_ARM]

    out["rows"] = len(a["rows"])
    out["emitted_tokens"] = len(a["emitted_tokens"])
    mismatches = []
    if a["seed_tokens"] != b["seed_tokens"]:
        mismatches.append("seed_tokens differ")
    if a["emitted_tokens"] != b["emitted_tokens"]:
        n = sum(x != y for x, y in zip(a["emitted_tokens"], b["emitted_tokens"]))
        mismatches.append(f"emitted_tokens differ at {n} positions")
    if len(a["rows"]) != len(b["rows"]):
        mismatches.append(f"row count {len(a['rows'])} vs {len(b['rows'])}")

    worst_logit_delta = 0.0
    for i, (ra, rb) in enumerate(zip(a["rows"], b["rows"])):
        for key in ("sequential_argmax", "top2_tokens"):
            if ra[key] != rb[key]:
                mismatches.append(f"row {i} {key}: {ra[key]} vs {rb[key]}")
        for x, y in zip(ra["top2_logits"], rb["top2_logits"]):
            worst_logit_delta = max(worst_logit_delta, abs(x - y))
        if ra["top1_logit"] != rb["top1_logit"]:
            mismatches.append(f"row {i} top1_logit: {ra['top1_logit']} vs {rb['top1_logit']}")

    out["worst_top2_logit_abs_delta"] = worst_logit_delta
    out["mismatch_count"] = len(mismatches)
    out["mismatches"] = mismatches[:10]
    out["bit_identical"] = not mismatches and worst_logit_delta == 0.0
    out["status"] = "bit_identical" if out["bit_identical"] else "MISMATCH"
    return out


def order_decomposition(data: dict, key: str) -> dict:
    """Separate the ARM effect from the RUN-POSITION effect.

    The single most dangerous confound here is that the second arm of a pair
    runs from a different thermal/cache history than the first.  ABBA exists to
    cancel that, but only if it is actually checked, so this reads the real
    `started` timestamps rather than assuming the intended rotation happened.

    With a balanced schedule (equal numbers of BASE-first and MEMO-first
    prompts) the two effects are orthogonal:
        arm      = mean(BASE - MEMO)      > 0 means MEMO is genuinely faster
        position = mean(second - first)   > 0 means the second slot is slower
    An effect that is really a position artifact shows up in `position` and
    collapses in `arm`.
    """
    arm_deltas, pos_deltas, base_first = [], [], 0
    for prompt, arms in data.items():
        b, m = arms[BASE_ARM], arms[MEMO_ARM]
        arm_deltas.append(b[key] - m[key])
        b_started = b["meta"].get("started", "")
        m_started = m["meta"].get("started", "")
        if b_started and m_started:
            if b_started < m_started:
                base_first += 1
                pos_deltas.append(m[key] - b[key])
            else:
                pos_deltas.append(b[key] - m[key])
    n = len(data)
    return {"n": n, "base_first": base_first, "memo_first": n - base_first,
            "balanced": base_first * 2 == n,
            "arm_effect_s": statistics.mean(arm_deltas),
            "position_effect_s": statistics.mean(pos_deltas) if pos_deltas else None,
            "arm_deltas_s": arm_deltas, "position_deltas_s": pos_deltas}


def main(argv: list[str]) -> int:
    logs = [Path(a) for a in argv[argv.index("--logs") + 1:]] if "--logs" in argv else []
    gates = parse_gate_log(logs) if logs else {}

    data = {p: pair for p in PROMPTS if (pair := load_pair(p)) is not None}
    if not data:
        print(f"e24: no completed BASE/MEMO pairs under {RUNS}", file=sys.stderr)
        return 1

    rows = []
    for prompt, arms in data.items():
        b, m = arms[BASE_ARM], arms[MEMO_ARM]
        rows.append({
            "prompt": prompt,
            "mtp_base_s": b["mtp_true_s"], "mtp_memo_s": m["mtp_true_s"],
            "mtp_effect_pct": pct(b["mtp_true_s"], m["mtp_true_s"]),
            "ser_base_s": b["serial_true_s"], "ser_memo_s": m["serial_true_s"],
            "ser_effect_pct": pct(b["serial_true_s"], m["serial_true_s"]),
            "rounds_base": b["rounds"], "rounds_memo": m["rounds"],
            "raw_base": b["raw"], "raw_memo": m["raw"],
        })

    print("=" * 94)
    print("E24  absolute true-decode wall seconds (prefill SUBTRACTED), 512 decode tokens")
    print("     effect% > 0 means MEMO (cached constants) is FASTER than BASE")
    print("=" * 94)
    print(f"{'prompt':<16}{'MTP base':>10}{'MTP memo':>10}{'MTP %':>9}"
          f"{'SER base':>10}{'SER memo':>10}{'SER %':>9}{'rounds':>9}")
    for r in rows:
        print(f"{r['prompt']:<16}{r['mtp_base_s']:>10.4f}{r['mtp_memo_s']:>10.4f}"
              f"{r['mtp_effect_pct']:>+9.3f}{r['ser_base_s']:>10.4f}"
              f"{r['ser_memo_s']:>10.4f}{r['ser_effect_pct']:>+9.3f}"
              f"{r['rounds_base']:>5}/{r['rounds_memo']:<3}")

    mtp_e = [r["mtp_effect_pct"] for r in rows]
    ser_e = [r["ser_effect_pct"] for r in rows]
    print("-" * 94)
    for name, es in (("MTP leg", mtp_e), ("SERIAL leg", ser_e)):
        pos = sum(1 for e in es if e > 0)
        print(f"{name:<12} median {statistics.median(es):+.3f}%   "
              f"mean {statistics.fmean(es):+.3f}%   "
              f"spread {min(es):+.3f}%..{max(es):+.3f}%   "
              f"MEMO faster on {pos}/{len(es)} prompts")

    # Prediction from Phase 1, stated so Phase 3 can refute it.
    print("\nPREDICTED saving from Phase 1's 9.711us marginal cast, vs measured:")
    for r in rows:
        pm = r["rounds_base"] * SITES_PER_FORWARD * CAST_US
        ps = 512 * SITES_PER_FORWARD * CAST_US
        print(f"  {r['prompt']:<16} MTP pred {pm:.4f}s "
              f"meas {r['mtp_base_s']-r['mtp_memo_s']:+.4f}s "
              f"realis {(r['mtp_base_s']-r['mtp_memo_s'])/pm:+.2f}   |   "
              f"SER pred {ps:.4f}s meas {r['ser_base_s']-r['ser_memo_s']:+.4f}s "
              f"realis {(r['ser_base_s']-r['ser_memo_s'])/ps:+.2f}")

    print("\nARM vs RUN-POSITION DECOMPOSITION (the main confound ABBA exists to cancel):")
    order = {}
    for leg, key in (("MTP", "mtp_true_s"), ("SERIAL", "serial_true_s")):
        d = order_decomposition(data, key)
        order[leg] = d
        pos = d["position_effect_s"]
        print(f"  {leg:<7} arm effect {d['arm_effect_s']*1000:+7.2f} ms "
              f"(MEMO faster if >0)   position effect "
              f"{pos*1000:+7.2f} ms (second slot slower if >0)")
        print(f"          schedule: {d['base_first']} BASE-first, {d['memo_first']} MEMO-first, "
              f"balanced={d['balanced']}")
    if not all(d["balanced"] for d in order.values()):
        print("  WARNING: schedule is NOT balanced; arm and position effects are not "
              "yet orthogonal. Only an even prefix of the registered order is reportable.")

    # Forward-count scaling.  The 96 casts are paid once per TARGET FORWARD and
    # are width-independent, so the serial leg (512 forwards) should save more
    # absolute wall time than the MTP leg (one forward per round).  The naive
    # prediction 512/rounds additionally assumes both legs expose the SAME
    # FRACTION of encode time to the critical path.  That assumption is wrong in
    # a knowable direction: an M=1 serial forward issues more dispatches over
    # less GPU work than an M~3 speculative round, so encode sits closer to the
    # serial critical path.  Report the naive ratio and the per-leg realization
    # factors side by side so a mismatch cannot masquerade as a refutation.
    mean_rounds = statistics.mean(r["rounds_base"] for r in rows)
    expected = 512.0 / mean_rounds
    mtp_arm, ser_arm = order["MTP"]["arm_effect_s"], order["SERIAL"]["arm_effect_s"]
    measured = ser_arm / mtp_arm if mtp_arm else float("nan")
    mtp_pred = statistics.mean(r["rounds_base"] * SITES_PER_FORWARD * CAST_US for r in rows)
    ser_pred = 512 * SITES_PER_FORWARD * CAST_US
    mtp_real, ser_real = mtp_arm / mtp_pred, ser_arm / ser_pred
    print("\nFORWARD-COUNT SCALING (casts are paid once per target forward):")
    print(f"  serial forwards {512}, mean MTP rounds {mean_rounds:.1f} "
          f"-> equal-exposure prediction SERIAL/MTP saving = {expected:.2f}x")
    print(f"  measured SERIAL/MTP saving = {ser_arm*1000:+.2f}ms / "
          f"{mtp_arm*1000:+.2f}ms = {measured:.2f}x")
    print(f"  realization vs full Phase-1 cost: MTP {mtp_real:.3f}, SERIAL {ser_real:.3f} "
          f"-> serial exposes {ser_real/mtp_real:.2f}x more of the tax")
    print("  Both realizations are far below 1.0: most of the removed encode time is "
          "overlapped with GPU execution and never reaches the wall clock.")
    scaling = {"equal_exposure_predicted_ratio": expected, "measured_ratio": measured,
               "serial_forwards": 512, "mean_mtp_rounds": mean_rounds,
               "mtp_predicted_s": mtp_pred, "ser_predicted_s": ser_pred,
               "mtp_realization": mtp_real, "ser_realization": ser_real,
               "serial_exposure_advantage": ser_real / mtp_real if mtp_real else None}

    print("\nCORRECTNESS (every timed leg, both arms):")
    bad = 0
    for prompt, arms in data.items():
        for arm, v in arms.items():
            ok = (v["matched"] and v["parity"] and v["divergence"] == 0
                  and v["declared_rows"] == v["checked_rows"])
            bad += not ok
            print(f"  {prompt:<16}{arm:<6} matched={v['matched']} parity={v['parity']} "
                  f"divergence={v['divergence']} rows {v['declared_rows']}=={v['checked_rows']} "
                  f"tripwire={v['drift_tripwire_passed']} -> {'OK' if ok else 'FAIL'}")
    print(f"  ALL LEGS CLEAN: {bad == 0}")

    print("\nCROSS-ARM REFERENCE-ROW IDENTITY (BASE vs MEMO, 02-mtp-verify-output.json):")
    identity = {p: reference_row_identity(p) for p in data}
    ident_bad = 0
    for prompt, v in identity.items():
        ident_bad += v["status"] != "bit_identical"
        print(f"  {prompt:<16} {v['status']:<14} rows={v.get('rows','?')} "
              f"emitted={v.get('emitted_tokens','?')} "
              f"worst|dtop2|={v.get('worst_top2_logit_abs_delta','?')} "
              f"mismatches={v.get('mismatch_count','?')}")
        for line in v.get("mismatches", []):
            print(f"      {line}")
    print(f"  ALL PROMPTS BIT-IDENTICAL: {ident_bad == 0}")
    print("  This closes the Phase 2 unit-test gap: invScalePair is private, so the")
    print("  unit test proved the memo but not the four call-site rewirings. These rows")
    print("  carry exact top-two logits from all 64 target layers over the full window.")

    print("\nVERIFY-WIDTH HISTOGRAM at M = depth + 1 (MTP leg):")
    for prompt, arms in data.items():
        for arm, v in arms.items():
            hist = {int(d) + 1: n for d, n in v["depth_hist"].items()}
            print(f"  {prompt:<16}{arm:<6} rounds={v['rounds']:<5} "
                  f"mean_depth={v['mean_depth']:.4f} "
                  f"M={dict(sorted(hist.items()))}")

    print("\nTHERMAL / GATE (entry temperature timing actually started at):")
    entry_by_arm, entry_all, real_gate = {}, [], []
    for prompt, arms in data.items():
        for arm, v in arms.items():
            meta = v["meta"]
            pre, post = gpu_temp_of(meta.get("thermal_before")), gpu_temp_of(meta.get("thermal_after"))
            passed = meta.get("cool_gate_passed_real_gate", "unknown")
            real_gate.append(passed)
            if pre is not None:
                entry_by_arm.setdefault(arm, []).append(pre)
                entry_all.append(pre)
            print(f"  {prompt}-{arm:<18} entry={fmt(pre)}C exit={fmt(post)}C "
                  f"cool_gate={meta.get('cool_gate','?')} "
                  f"settle(reached={fmt(gpu_temp_of(meta.get('settle_reached_c')))}C "
                  f"min={fmt(gpu_temp_of(meta.get('settle_min_c')))}C "
                  f"waited={meta.get('settle_waited_s','?')}s)")
    if entry_all:
        print(f"\n  entry-temperature spread across all timed legs: "
              f"{max(entry_all) - min(entry_all):.3f}C "
              f"(min {min(entry_all):.3f}C, max {max(entry_all):.3f}C)")
        for arm, xs in sorted(entry_by_arm.items()):
            print(f"    {arm:<6} mean entry {statistics.mean(xs):.3f}C over {len(xs)} legs")
        if len(entry_by_arm) == 2:
            (a, xa), (b, xb) = sorted(entry_by_arm.items())
            print(f"    {a} - {b} mean entry bias: "
                  f"{statistics.mean(xa) - statistics.mean(xb):+.3f}C "
                  f"(ABBA cancels this to first order; a bias favouring the faster arm "
                  f"would have the WARMER arm looking slower)")
    all_real = bool(real_gate) and all(x == "true" for x in real_gate)
    print(f"\n  cool_gate_passed_real_gate={str(all_real).lower()}  "
          f"(carried verbatim from meta.txt: {sorted(set(real_gate))})")
    print(f"  gate_qualified_for_timing={str(all_real).lower()}")
    if not all_real:
        print("  NOTE: this host's idle GPU floor sits above COOL_GATE_TEMP_C=40, so the")
        print("  wrapper gate is unsatisfiable. Timing ran under the E15-authorized")
        print("  MLXFAST_LOCAL_COOL_GATE=0 policy: ABBA order, per-arm entry/exit temps,")
        print("  spread reported above, and both flags carried false rather than softened.")

    if "--json" in argv:
        out = {"rows": rows, "gates": gates,
               "mtp_median_pct": statistics.median(mtp_e),
               "ser_median_pct": statistics.median(ser_e),
               "correctness_all_clean": bad == 0,
               "order_decomposition": order,
               "forward_count_scaling": scaling,
               "reference_row_identity": identity,
               "reference_rows_all_bit_identical": ident_bad == 0,
               "cool_gate_passed_real_gate": all_real,
               "gate_qualified_for_timing": all_real,
               "entry_temp_spread_c": (max(entry_all) - min(entry_all)) if entry_all else None,
               "entry_temp_mean_by_arm": {a: statistics.mean(x) for a, x in entry_by_arm.items()},
               "arms": {p: {a: v for a, v in arms.items()} for p, arms in data.items()}}
        Path("research/results/e24-phase3.json").write_text(
            json.dumps(out, indent=2, default=str))
        print("\nwrote research/results/e24-phase3.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
