#!/usr/bin/env python3
"""Select one E68 depth-price arm in the worktree, and say what it will do.

E68's scored surface is one Swift line:

    internal static let depthPriceArm: DepthPriceArm = .ship

The four arms hold the total price at `maxDepth * headStepCostRatio` and vary
only its shape across positions:

  ship   flat, bit-identical to the tip. The control.
  pb5    E56's one-boundary vector priced INTO verify width 5.
  pb7    the same construction priced INTO verify width 7.
  pbfit  the rung-1 measured verify curve, rescaled to the same total.

`pb5` is the POSITIVE CONTROL, not a candidate. Replaying the shipped walk
shows it stops at depth 3 at both ranked acceptance rates, where the tip
reaches depth 5. Shortening drafts is a direction this pool has already
priced: raising `headStepCostRatio` 0.18 -> 0.32 scored 2.84585, a clean -3%.
So `pb5` must lose. If it does not, the arm selector is not reaching the
scored path and no other arm's number can be trusted.

The manifest this writes records the resolved vector and the predicted depth
at each ranked acceptance rate, so the prediction is on the record before the
leg is timed rather than after.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
TARGET = REPO / "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"

ARMS = ("ship", "pb5", "pb7", "pbfit")
HEAD_STEP_COST_RATIO = 0.18
MAX_DEPTH = 8
BOUNDARY_TIER_FACTOR = 2.0301
SDPA_WIDTH_WALL_DEPTH_CAP = 5

ARM_ANCHOR = "    internal static let depthPriceArm: DepthPriceArm = ."
RAW_ANCHOR = "    internal static let measuredRawDepthPrice: [Double] = "

# beagle and medicine, the two reconstructed ranked prompts, then the local
# fixture rate measured on this host.
RANKED_ACCEPTANCE = {"beagle": 0.8351, "medicine": 0.8750, "local": 0.9189}


def uniform_marginals():
    return [HEAD_STEP_COST_RATIO] * MAX_DEPTH


def boundary_marginals(entering_verify_width):
    within = (MAX_DEPTH * HEAD_STEP_COST_RATIO
              / (MAX_DEPTH - 1 + BOUNDARY_TIER_FACTOR))
    out = [within] * MAX_DEPTH
    out[entering_verify_width - 2] = within * BOUNDARY_TIER_FACTOR
    return out


def rescale(raw):
    total = MAX_DEPTH * HEAD_STEP_COST_RATIO
    return [v * total / sum(raw) for v in raw]


def walk_depth(marginal, cap, p):
    """Replay `costModelDepth` under a flat acceptance p."""
    reach, expected, cumulative, depth = 1.0, 0.0, 1.0, 0
    for d in range(min(cap, len(marginal))):
        reach *= p
        threshold = marginal[d] * (1.0 + expected) / cumulative
        if not reach > threshold:
            return {"depth": depth, "verify_width": depth + 1,
                    "stopped_at_depth": d + 1,
                    "reach": reach, "threshold": threshold}
        expected += reach
        cumulative += marginal[d]
        depth = d + 1
    return {"depth": depth, "verify_width": depth + 1,
            "stopped_at_depth": None}


def raw_from_rung1(path, key):
    payload = json.loads(pathlib.Path(path).read_text())
    fits = payload["pbfit_by_verify_forward"]
    if key is None:
        if len(fits) != 1:
            raise SystemExit(
                "e68_swift_arm: rung-1 file holds %d verify-forward fits; "
                "name one with --verify-forward-key" % len(fits))
        key = next(iter(fits))
    if key not in fits:
        raise SystemExit("e68_swift_arm: no fit for verify-forward %r; have %s"
                         % (key, ", ".join(sorted(fits))))
    fit = fits[key]
    if "raw" not in fit or any(v is None for v in fit.get("raw", [])):
        raise SystemExit("e68_swift_arm: rung-1 curve is incomplete")
    return fit["raw"], key


def marginals_for(arm, raw):
    if arm == "ship":
        return uniform_marginals()
    if arm == "pb5":
        return boundary_marginals(5)
    if arm == "pb7":
        return boundary_marginals(7)
    if arm == "pbfit":
        if raw is None:
            raise SystemExit(
                "e68_swift_arm: pbfit needs --raw-from with the rung-1 curve")
        return rescale(raw)
    raise SystemExit("e68_swift_arm: unknown arm %s" % arm)


def swift_literal(values):
    body = ",\n".join("        %.17g" % v for v in values)
    return "[\n%s,\n    ]" % body


def patch(text, arm, raw):
    hits = [line for line in text.splitlines() if line.startswith(ARM_ANCHOR)]
    if len(hits) != 1:
        raise SystemExit("e68_swift_arm: arm anchor matched %d lines, want 1"
                         % len(hits))
    text = text.replace(hits[0], ARM_ANCHOR + arm)

    raw_hits = [line for line in text.splitlines()
                if line.startswith(RAW_ANCHOR)]
    if len(raw_hits) != 1:
        raise SystemExit("e68_swift_arm: raw anchor matched %d lines, want 1"
                         % len(raw_hits))
    # The array literal is multi-line once filled, so replace the whole
    # statement between the anchor and its closing bracket.
    start = text.index(raw_hits[0])
    end = text.index("\n", start)
    if raw_hits[0].rstrip().endswith("["):
        end = text.index("\n    ]", start) + len("\n    ]")
    literal = "[]" if raw is None else swift_literal(raw)
    text = text[:start] + RAW_ANCHOR + literal + text[end:]
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=ARMS)
    ap.add_argument("--raw-from",
                    help="rung-1 analysis JSON, required for pbfit")
    ap.add_argument("--verify-forward-key",
                    help="which verify-forward fit to take from --raw-from")
    ap.add_argument("--out", help="write the arm manifest here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw, key = None, None
    if args.raw_from:
        raw, key = raw_from_rung1(args.raw_from, args.verify_forward_key)
    if args.arm != "pbfit":
        # Only pbfit reads the measured array. Leaving it empty for the other
        # arms keeps their source identical to the unpatched tree.
        raw = None

    marginal = marginals_for(args.arm, raw)
    text = patch(TARGET.read_text(), args.arm, raw)
    if not args.dry_run:
        TARGET.write_text(text)

    predictions = {}
    for name, p in RANKED_ACCEPTANCE.items():
        predictions[name] = {
            "p": p,
            "default_cap": walk_depth(marginal, SDPA_WIDTH_WALL_DEPTH_CAP, p),
            "streak_cap": walk_depth(marginal, MAX_DEPTH, p),
        }

    payload = {
        "arm": args.arm,
        "role": ("positive control" if args.arm == "pb5" else
                 "control" if args.arm == "ship" else "candidate"),
        "dry_run": bool(args.dry_run),
        "marginal": marginal,
        "marginal_total": sum(marginal),
        "shipped_total": MAX_DEPTH * HEAD_STEP_COST_RATIO,
        "measured_raw": raw,
        "verify_forward_key": key,
        "predicted_depth": predictions,
        "sha256": {
            str(TARGET.relative_to(REPO)):
                hashlib.sha256(text.encode()).hexdigest(),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        pathlib.Path(args.out).write_text(rendered)
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
