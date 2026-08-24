#!/usr/bin/env python3
"""E182: the band summary read off the phase ladder report.

    usage: research/e182_bands.py [--report research/out/e182/report.json]

`harness=local` throughout. The band arm drains the device 129 times per verify
forward, so its ROUND time is inflated by construction and only band shares and
band shapes are read from it.

Edward's E184 isolated `gatedDeltaKernel` probe is used as an external
predictor for the GDN recurrence alone: 48 layers x (223.9 + 3.65 * T) us at
T = M + 1, which is 10.9-11.5 ms per round and nearly flat in M.
"""

from __future__ import annotations

import argparse
import json
import pathlib

BANDS = [
    ("gdn_mixer", "band_gdn_mixer_us", 48),
    ("gdn_mlp", "band_gdn_mlp_us", 48),
    ("fa_mixer", "band_fa_mixer_us", 16),
    ("fa_mlp", "band_fa_mlp_us", 16),
    ("pre", "band_pre_us", 1),
]
WIDTHS = list(range(1, 10))


def gdn_probe_ms(m: int) -> float:
    """Edward's isolated GDN scan prediction for the whole round, ms."""
    return 48 * (223.9 + 3.65 * (m + 1)) / 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="research/out/e182/report.json")
    parser.add_argument("--out", default="research/out/e182/bands.json")
    args = parser.parse_args()

    report = json.loads(pathlib.Path(args.report).read_text())
    band = {row["m"]: row for row in report["tables"]["band"]}
    trace = {row["m"]: row for row in report["tables"]["trace"]}

    print("band device time per round, ms (harness=local, attribution arm)")
    print("  M " + "".join("%11s" % name for name, _, _ in BANDS)
          + "%11s%11s%11s" % ("verify", "head", "host"))
    rows = {}
    for m in WIDTHS:
        b = band[m]
        vals = {name: b[field] / 1000.0 for name, field, _ in BANDS}
        verify = sum(vals.values())
        head = b.get("d_submit2_us", 0.0) / 1000.0
        host = sum(
            b.get(f, 0.0) for f in ("readout_us", "commit_us", "upkeep_us")
        ) / 1000.0
        rows[m] = {**vals, "verify": verify, "head": head, "host": host}
        print(
            "%3d " % m
            + "".join("%11.3f" % vals[name] for name, _, _ in BANDS)
            + "%11.3f%11.3f%11.3f" % (verify, head, host)
        )

    print("\nmarginal share of the M=1 -> M=9 growth")
    total = sum(
        rows[9][k] - rows[1][k] for k in ("verify", "head", "host")
    )
    for key in ("verify", "head", "host"):
        d = rows[9][key] - rows[1][key]
        print("  %-8s %+8.3f ms  %6.1f %%" % (key, d, 100 * d / total))
    verify_growth = rows[9]["verify"] - rows[1]["verify"]
    print("  within verify:")
    for name, _, _ in BANDS:
        d = rows[9][name] - rows[1][name]
        print(
            "    %-10s %+8.3f ms  %6.1f %% of verify growth"
            % (name, d, 100 * d / verify_growth)
        )

    print("\nper-layer cost, ms (GDN layer = mixer+mlp over 48; FA over 16)")
    print("  %3s %12s %12s %8s" % ("M", "gdn/layer", "fa/layer", "ratio"))
    per_layer = {}
    for m in WIDTHS:
        g = (rows[m]["gdn_mixer"] + rows[m]["gdn_mlp"]) / 48.0
        f = (rows[m]["fa_mixer"] + rows[m]["fa_mlp"]) / 16.0
        per_layer[m] = {"gdn": g, "fa": f, "ratio": g / f}
        print("  %3d %12.4f %12.4f %8.3f" % (m, g, f, g / f))

    print("\nGDN recurrence against Edward's isolated probe (E184)")
    print(
        "  %3s %14s %14s %10s"
        % ("M", "gdn_mixer ms", "probe ms", "probe share")
    )
    probe = {}
    for m in WIDTHS:
        p = gdn_probe_ms(m)
        probe[m] = p
        print(
            "  %3d %14.3f %14.3f %9.1f %%"
            % (m, rows[m]["gdn_mixer"], p, 100 * p / rows[m]["gdn_mixer"])
        )
    probe_growth = probe[9] - probe[1]
    print(
        "  probe growth M1->M9 %.3f ms = %.1f %% of the measured gdn_mixer "
        "growth %.3f ms"
        % (
            probe_growth,
            100 * probe_growth / (rows[9]["gdn_mixer"] - rows[1]["gdn_mixer"]),
            rows[9]["gdn_mixer"] - rows[1]["gdn_mixer"],
        )
    )

    print("\nshipped-overlap round, ms (harness=local, trace arm)")
    print(
        "  %3s %3s %10s %10s %10s %10s %10s"
        % ("M", "G", "round", "increment", "draft", "verify+eval", "host")
    )
    prev = None
    shipped = {}
    for m in WIDTHS:
        t = trace[m]
        r = t["round_us"] / 1000.0
        draft = t.get("draft_build_us", 0.0) / 1000.0
        dev = (t.get("verify_build_us", 0.0) + t.get("eval_wall_us", 0.0)) / 1000.0
        host = sum(
            t.get(f, 0.0) for f in ("readout_us", "commit_us", "upkeep_us")
        ) / 1000.0
        shipped[m] = {"round": r, "draft": draft, "device": dev, "host": host}
        inc = "" if prev is None else "%10.3f" % (r - prev)
        print(
            "  %3d %3d %10.3f %10s %10.3f %10.3f %10.3f"
            % (m, t["groups"], r, inc, draft, dev, host)
        )
        prev = r

    out = {
        "harness": "local",
        "gate_qualified_for_timing": False,
        "bands_ms": rows,
        "per_layer_ms": per_layer,
        "gdn_probe_ms": probe,
        "shipped_round_ms": shipped,
    }
    path = pathlib.Path(args.out)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
