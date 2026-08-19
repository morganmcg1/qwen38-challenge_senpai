#!/usr/bin/env python3
"""E54's zero-GPU deliverable: per-width score-weighted QMV cost shares, twice.

The advisor asked for M=5's individual share, then corrected the request when
edward's E53 merged: report BOTH mixture models side by side and do not pick a
winner. They agree on the wide bucket and on the narrow sum and roughly swap the
split inside it, so a cell's price depends on which model you believe:

  askeladd E48   max-entropy exponential tilt of a measured local corpus
                 histogram to each prompt's published mean width.
  edward E53     two-state burst acceptance process driven through a port of the
                 real `costModelDepth` greedy walk, fitted to the published
                 board rows. Its interval is an IDENTIFICATION interval over 48
                 feasible (beagle, medicine) solution pairs, not a standard
                 error.

Both models are used exactly as their authors published them. E48 is imported
from this base; E53 is not on this base, so its module and artefact are read
from the pinned advisor ref that merged it, never forked into this branch.

The self-check is the point of trusting the numbers: this file recomputes E53's
own published f456 / f78 / f9 envelope from its per-width mixtures and requires
a match, so the per-width split it adds is the same arithmetic as the buckets
edward reported.

  python3 research/e54_shares.py --out research/e54-artifacts/e54-shares.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
E53_REF = "a2c3dbc"            # advisor branch commit that merged E53 (PR #56)
E53_MODULE = "research/e53_width_mixture.py"
E53_ARTIFACT = "research/e53-width-mixture.json"
WIDTHS = list(range(2, 10))
BUCKETS = {"f123": (1, 2, 3), "f456": (4, 5, 6), "f78": (7, 8), "f9": (9,)}
ENVELOPE_TOLERANCE = 1e-9


def _from_ref(path: str) -> str:
    return subprocess.run(["git", "show", "%s:%s" % (E53_REF, path)],
                          capture_output=True, text=True, check=True).stdout


def _load_module_from_ref(path: str, name: str):
    text = _from_ref(path)
    tmp = pathlib.Path(tempfile.mkdtemp()) / ("%s.py" % name)
    tmp.write_text(text)
    sys.path.insert(0, str(REPO / "research"))   # its own imports
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def e48_shares() -> dict:
    sys.path.insert(0, str(REPO / "research"))
    import e48_score_weighted_shares as e48       # noqa: E402
    from qmv_score_leverage import marginal_weights  # noqa: E402

    weights = marginal_weights()
    per_prompt = {}
    for name, target in e48.MEAN_M.items():
        tilted, lam = e48.exponential_tilt(e48.CORPUS_HISTOGRAM, target)
        per_prompt[name] = {"tilt_lambda": lam,
                            "dispatch_shares": tilted,
                            "cost_shares": e48.cost_shares(tilted)}
    combined = {
        m: sum(weights[p] * per_prompt[p]["cost_shares"].get(m, 0.0)
               for p in weights)
        for m in WIDTHS
    }
    return {
        "model": "askeladd E48, max-entropy tilt of a measured local corpus",
        "source": "research/e48_score_weighted_shares.py on this base",
        "weights": weights,
        "mean_m": e48.MEAN_M,
        "per_width": combined,
        "per_width_by_prompt": {p: per_prompt[p]["cost_shares"] for p in per_prompt},
        "buckets": {b: sum(combined.get(m, 0.0) for m in ms)
                    for b, ms in BUCKETS.items()},
        "identification": "PREDICTION. The hidden prompts expose one width "
                          "statistic each, so this is an extrapolation from a "
                          "local corpus, not a measured histogram.",
        "caveat_m3": "M=3 is exactly 0 because the 78-round local corpus "
                     "observed no M=3 round and an exponential tilt cannot "
                     "move a zero. It is a corpus artefact, not a claim that "
                     "M=3 never dispatches.",
    }


def e53_composite(mix_b: dict, mix_m: dict, cost, weights: dict) -> dict:
    """Per-width shares of score-weighted candidate-leg QMV cost, E53's rule."""
    prompts = {"beagle": mix_b, "medicine": mix_m}
    totals = {p: sum(s * cost(int(w)) for w, s in mix.items())
              for p, mix in prompts.items()}
    denom = sum(weights[p] * totals[p] for p in prompts)
    per_width = {
        m: sum(weights[p] * prompts[p].get(str(m), prompts[p].get(m, 0.0))
               * cost(m) for p in prompts) / denom
        for m in WIDTHS
    }
    return {"per_width": per_width,
            "buckets": {b: sum(per_width.get(m, 0.0) for m in ms)
                        for b, ms in BUCKETS.items()},
            "mean_cost_ms": {p: totals[p] for p in totals}}


def e53_shares() -> dict:
    mod = _load_module_from_ref(E53_MODULE, "e53_width_mixture")
    art = json.loads(_from_ref(E53_ARTIFACT))
    weights = art["weights"]

    feasible = {p: [s for s in art["burst"][p] if s.get("feasible")]
                for p in ("beagle", "medicine")}
    pairs = []
    for b in feasible["beagle"]:
        for m in feasible["medicine"]:
            comp = e53_composite(b["mixture"], m["mixture"], mod.cost_ms, weights)
            comp["beagle"] = {"persistence": b["persistence"], "q_easy": b["q_easy"]}
            comp["medicine"] = {"persistence": m["persistence"], "q_easy": m["q_easy"]}
            pairs.append(comp)

    envelope = {
        "per_width": {m: {"low": min(p["per_width"][m] for p in pairs),
                          "high": max(p["per_width"][m] for p in pairs),
                          "mid": 0.5 * (min(p["per_width"][m] for p in pairs)
                                        + max(p["per_width"][m] for p in pairs))}
                      for m in WIDTHS},
        "buckets": {b: {"low": min(p["buckets"][b] for p in pairs),
                        "high": max(p["buckets"][b] for p in pairs)}
                    for b in BUCKETS},
        "pairs": len(pairs),
    }

    published = art["burst_identification_envelope"]
    check = {"published_pairs": published["pairs"], "recomputed_pairs": len(pairs),
             "buckets": {}}
    ok = published["pairs"] == len(pairs)
    for b in BUCKETS:
        pub, mine = published[b], envelope["buckets"][b]
        row = {"published": pub, "recomputed": mine,
               "low_delta": mine["low"] - pub["low"],
               "high_delta": mine["high"] - pub["high"]}
        row["matches"] = (abs(row["low_delta"]) < ENVELOPE_TOLERANCE
                          and abs(row["high_delta"]) < ENVELOPE_TOLERANCE)
        ok = ok and row["matches"]
        check["buckets"][b] = row
    check["reproduces_published_envelope"] = ok

    return {
        "model": "edward E53, two-state burst process through the real greedy "
                 "costModelDepth walk",
        "source": "%s:%s and %s" % (E53_REF, E53_MODULE, E53_ARTIFACT),
        "weights": weights,
        "envelope": envelope,
        "selfcheck": check,
        "identification": "IDENTIFICATION INTERVAL over 48 feasible (beagle, "
                          "medicine) burst solution pairs, not a standard "
                          "error: 152 content-distinct trees reproduce the "
                          "published board rows to 16 digits.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/e54-artifacts/e54-shares.json")
    args = ap.parse_args()

    e48 = e48_shares()
    e53 = e53_shares()
    payload = {"askeladd_e48": e48, "edward_e53": e53,
               "cells_measured_by_e54": [5, 7, 8], "cell_from_e49": 9}

    print("per-width score-weighted share of candidate-leg QMV cost")
    print("  %-4s %-12s %-26s" % ("M", "E48", "E53 (identification interval)"))
    for m in WIDTHS:
        lo = e53["envelope"]["per_width"][m]["low"]
        hi = e53["envelope"]["per_width"][m]["high"]
        mark = "  <- E54 cell" if m in (5, 7, 8) else ("  <- E49 cell" if m == 9 else "")
        print("  %-4d %-12.4f %6.4f - %-6.4f %s"
              % (m, 100 * e48["per_width"][m], 100 * lo, 100 * hi, mark))
    print("\n  bucket    E48        E53 interval")
    for b in ("f123", "f456", "f78", "f9"):
        print("  %-9s %-10.4f %6.4f - %-6.4f"
              % (b, 100 * e48["buckets"][b],
                 100 * e53["envelope"]["buckets"][b]["low"],
                 100 * e53["envelope"]["buckets"][b]["high"]))
    chk = e53["selfcheck"]
    print("\n  E53 envelope self-check: pairs %d/%d, buckets reproduce = %s"
          % (chk["recomputed_pairs"], chk["published_pairs"],
             chk["reproduces_published_envelope"]))

    dest = pathlib.Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("\nwrote %s" % dest)
    return 0 if chk["reproduces_published_envelope"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
