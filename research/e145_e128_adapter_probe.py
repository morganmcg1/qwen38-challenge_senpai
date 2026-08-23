#!/usr/bin/env python3
"""E145 R7-5: exercise the E128 -> E145 arm adapter on every E128 arm name.

`harness=local`, zero GPU, no decode. This is an **interface** probe, not a
pricing run: it asks only whether the documented four-line adapter in
`research/e145-result.md` ("How to re-price any E128 arm on the measured
curve") returns a legal depth for each arm name in `e128_price.ARMS`, and
whether the `install(measured)` ordering precondition is detectable by the
one-line assertion the document hands the caller.

Advisor F12 item 2 asked for exactly two things: the ordering trap stated as a
numbered precondition with a copyable assertion, and an explicit statement of
which arm names the adapter has been exercised on. This script produces the
evidence for the second and demonstrates the first.

It does NOT price any arm. Pricing all fourteen against the measured curve is
the next assignment.

Usage:
  python3 e145_e128_adapter_probe.py
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e128_price  # noqa: E402
import e140_cells  # noqa: E402
from e145_r7_state import curves_and_prices  # noqa: E402

ARTIFACT = HERE / "e145-artifacts" / "r7-adapter-probe.json"
JENSEN = HERE / "e128-artifacts" / "jensen-and-sign.json"

# The eight names interim 9 reported as checked. Everything else in
# `e128_price.ARMS` was unexercised before this probe.
PREVIOUSLY_EXERCISED = ["ship", "nomargin", "recal", "marginup", "marginfull",
                        "rankedprice", "static7", "oracle"]

# E128's published fourteen-arm table, in its own published order.
E128_TABLE = ["oracle", "static7", "marginfull", "expectedonly", "levelfix",
              "recal", "reachonly", "nomargin1", "nomargin0", "nomargin",
              "marginup", "rankedprice", "jensen_both", "jensen"]


def e128_walker(arm, recal=(2.0, 3.0), level=None):
    """The adapter exactly as `research/e145-result.md` publishes it."""
    policy = e128_price.make_policy(arm, recal=recal, level=level)

    def chooser(ema, margin, offer, adjust, ctx, force, price):
        return policy(ema, margin, offer, ctx["capability"])

    return chooser


def probe_states():
    """A small ladder of round states that spans the shipped decision space."""
    ema = list(e128_price.EMA_PRIOR)
    hot = [min(0.99, v + 0.10) for v in ema]
    cold = [max(0.30, v - 0.35) for v in ema]
    return [
        ("prior_mid", ema, 0.90, 8, 4),
        ("prior_tight_margin", ema, 0.05, 8, 4),
        ("hot_wide_offer", hot, 3.10, 8, 8),
        ("cold_short_offer", cold, 0.40, 2, 1),
        ("cold_zero_capability", cold, 0.01, 8, 0),
    ]


def legal(depth, offer) -> bool:
    cap = min(offer, e128_price.MAX_DEPTH, e128_price.SEGMENTED_VERIFY_DEPTH_CAP)
    return isinstance(depth, int) and 0 <= depth <= cap


def main() -> int:
    # Step 1: build both cost environments. NOTE that `curves_and_prices`
    # leaves the REPLAYED curve installed, because it installs each curve in
    # turn to read its price table and `replayed` is the last one it touches.
    points, measured, replayed, measured_price, replayed_price = \
        curves_and_prices(anchor="measured")

    # ---- Precondition 1, demonstrated rather than asserted --------------
    # `make_policy` freezes the rankedprice table at construction time, so the
    # arm built here, BEFORE install(measured), keeps the replayed table.
    stale_marginal, _ = e128_price.ranked_price_table()
    e140_cells.install(measured)
    fresh_marginal, _ = e128_price.ranked_price_table()
    trap_visible = any(abs(a - b) > 1e-12
                       for a, b in zip(stale_marginal, fresh_marginal))
    trap_worst_pct = max(
        100.0 * abs(a - b) / abs(b) for a, b in zip(stale_marginal,
                                                    fresh_marginal))

    # The one-line assertion the document hands the caller. It must pass now
    # and it must have failed before `install`.
    assert e128_price.ranked_price_table()[0] == measured_price[0], \
        "call e140_cells.install(measured) before e128_price.make_policy"
    would_have_failed = stale_marginal != measured_price[0]

    level_rows = {row["prompt_id"]: row
                  for row in json.loads(JENSEN.read_text())["hypothesis_j"]}
    # beagle is the Rule 123 prompt; its pooled level is the one the E128
    # level arms were fitted against.
    level = e128_price.pooled_level(
        level_rows, e128_price.RANKED_PROMPTS["beagle"]["fixture"])

    rows = []
    failures = []
    for arm in e128_price.ARMS:
        chooser = e128_walker(arm, level=level)
        depths = {}
        ok = True
        for name, ema, margin, offer, capability in probe_states():
            try:
                depth = chooser(list(ema), margin, offer, None,
                                {"capability": capability, "offer": offer,
                                 "base_rate": 0.83, "prev_slope": 0.0,
                                 "prev_rows": 0, "km": 0.0}, False,
                                measured_price)
            except Exception as exc:  # noqa: BLE001 - probe reports, not raises
                ok = False
                depths[name] = "ERROR: %s" % exc
                continue
            depths[name] = depth
            if not legal(depth, offer):
                ok = False
        rows.append({"arm": arm, "legal": ok, "depths": depths,
                     "in_e128_table": arm in E128_TABLE,
                     "previously_exercised": arm in PREVIOUSLY_EXERCISED})
        if not ok:
            failures.append(arm)

    newly = [r["arm"] for r in rows if not r["previously_exercised"]]
    table_covered = [a for a in E128_TABLE
                     if any(r["arm"] == a and r["legal"] for r in rows)]

    print("## R7-5  the E128 adapter, exercised on every arm name\n")
    print("  cost environment: measured curve installed, width-1 anchor "
          "measured")
    print("  ordering trap visible          %s, worst marginal shift %.2f %%"
          % (trap_visible, trap_worst_pct))
    print("  assertion would have failed before install(measured)  %s"
          % would_have_failed)
    print("  arms in e128_price.ARMS        %d" % len(rows))
    print("  arms returning a legal depth   %d" % sum(r["legal"] for r in rows))
    print("  E128 published table covered   %d / %d"
          % (len(table_covered), len(E128_TABLE)))
    print("  newly exercised by this probe  %d" % len(newly))
    if failures:
        print("  FAILURES: %s" % ", ".join(failures))
    print("\n  arm                     %s" % "  ".join(
        "%-18s" % s[0] for s in probe_states()))
    for row in rows:
        print("  %-22s %s" % (row["arm"], "  ".join(
            "%-18s" % row["depths"][s[0]] for s in probe_states())))

    out = {
        "harness": "local",
        "gpu_used": False,
        "what": "interface probe of the E128->E145 arm adapter, not a pricing "
                "run",
        "arms_total": len(rows),
        "arms_legal": sum(r["legal"] for r in rows),
        "arms_failed": failures,
        "e128_table": E128_TABLE,
        "e128_table_covered": table_covered,
        "previously_exercised": PREVIOUSLY_EXERCISED,
        "newly_exercised": newly,
        "ordering_trap_visible": trap_visible,
        "ordering_trap_worst_marginal_shift_pct": trap_worst_pct,
        "assertion_would_have_failed_before_install": would_have_failed,
        "probe_states": [
            {"name": s[0], "margin": s[2], "offer": s[3], "capability": s[4]}
            for s in probe_states()],
        "rows": rows,
    }
    ARTIFACT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("\nwrote %s" % ARTIFACT.relative_to(HERE.parent))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
