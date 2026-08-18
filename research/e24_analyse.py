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

    print("\nVERIFY-WIDTH HISTOGRAM at M = depth + 1 (MTP leg):")
    for prompt, arms in data.items():
        for arm, v in arms.items():
            hist = {int(d) + 1: n for d, n in v["depth_hist"].items()}
            print(f"  {prompt:<16}{arm:<6} rounds={v['rounds']:<5} "
                  f"mean_depth={v['mean_depth']:.4f} "
                  f"M={dict(sorted(hist.items()))}")

    print("\nTHERMAL / GATE (entry temperature timing actually started at):")
    for prompt, arms in data.items():
        for arm, v in arms.items():
            label = f"{prompt}-{arm}"
            g = gates.get(label)
            meta = v["meta"]
            gs = (f"gate_entry={g['entry_temp_c']}C waited={g['waited_s']}s"
                  if g else "gate_entry=NOT_CAPTURED")
            print(f"  {label:<24}{gs:<40} pre={meta.get('thermal_before','?')}")
            print(f"  {'':<24}post={meta.get('thermal_after','?')}")
    print(f"\n  cool_gate_passed_real_gate={str(len(gates) >= 2 * len(data)).lower()} "
          f"({len(gates)} gate passes captured for {2*len(data)} timed runs)")
    print(f"  gate_qualified_for_timing={str(len(gates) >= 2 * len(data)).lower()}")

    if "--json" in argv:
        out = {"rows": rows, "gates": gates,
               "mtp_median_pct": statistics.median(mtp_e),
               "ser_median_pct": statistics.median(ser_e),
               "correctness_all_clean": bad == 0,
               "arms": {p: {a: v for a, v in arms.items()} for p, arms in data.items()}}
        Path("research/results/e24-phase3.json").write_text(
            json.dumps(out, indent=2, default=str))
        print("\nwrote research/results/e24-phase3.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
