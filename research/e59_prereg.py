#!/usr/bin/env python3
"""E59 rung 3 pre-registration: what the isolated-cell legs must show.

Writes `research/e59-artifacts/e59-prereg.json` before any timing runs, so the
rung 3 decision bar and the stop rule cannot be chosen after seeing the numbers.

  python3 research/e59_prereg.py

Every constant here is copied from a named prior source. Nothing is re-derived.
"""

from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "research/e59-artifacts/e59-prereg.json"

PREREG = {
    "experiment": "E59",
    "rung": 3,
    "assignment_id": "qwen38-r1-e59-m5-rowblock-r2-route",
    "revision_id": "r1",
    "pr": 62,
    "base_sha": "989596895b7c8f889443dac0c87e024a428e6e9e",
    "orientation": (
        "delta_pct = (treated - control) / control * 100 at the treated width; "
        "negative means the treated build is faster"
    ),
    "decision_bar_pct": 0.77,
    "decision_bar_rule": (
        "max(0.770, worst in-session unchanged-code width |delta|, treated-width "
        "replicate spread) - whichever is largest wins"
    ),
    "decision_bar_provenance": (
        "0.770 is E54's isolated-leg floor (research/e54-artifacts/e54-prereg.json, "
        "decision_bar_pct). Same apparatus, same host, same run-qmv-curve driver, "
        "so the floor carries; the base moved from a35bb006 to 989596895b."
    ),
    "arms": {
        "iso_m5_ipg3": "shipped M=5 route <T,5,3> at rows_per_simd=4 (control)",
        "iso_m5_ipg5_r4": "<T,5,5> at rows_per_simd=4 - E54's lone-NA=5 cell, 125 regs",
        "iso_m5_ipg5_rb2": "<T,5,5> at rows_per_simd=2, sequential row blocks, 100 regs",
        "iso_m5_ipg5_rbx": "<T,5,5> at rows_per_simd=2, two x-groups, 90 regs",
    },
    "counterbalance": {
        "order": ["A", "B", "C", "D", "D", "C", "B", "A"],
        "map": {
            "A": "iso_m5_ipg3",
            "B": "iso_m5_ipg5_r4",
            "C": "iso_m5_ipg5_rb2",
            "D": "iso_m5_ipg5_rbx",
        },
        "why": (
            "a palindromic ABBA order over four arms cancels monotone thermal or "
            "clock drift to first order inside one session"
        ),
    },
    "pairs": {
        "R1": {
            "control": "iso_m5_ipg3",
            "treated": "iso_m5_ipg5_r4",
            "width": 5,
            "question": "does this base reproduce E54's lone-NA=5 cell win at M=5?",
            "reference_pct": -20.253,
            "reference_source": "E54 P1, research/e54-results.md, base a35bb006",
            "note": "replication check only; it is not a gate",
        },
        "T1": {
            "control": "iso_m5_ipg5_r4",
            "treated": "iso_m5_ipg5_rb2",
            "width": 5,
            "question": "what is the real rows_per_simd=2 tax at NA=5, row-block form?",
            "reference_pct": 10.54,
            "reference_source": (
                "E44's r=2 tax at NA=4. Recorded for contrast only. The assignment "
                "forbids inheriting it: NA=5 must be measured, not assumed."
            ),
        },
        "T2": {
            "control": "iso_m5_ipg5_r4",
            "treated": "iso_m5_ipg5_rbx",
            "width": 5,
            "question": "what is the real rows_per_simd=2 tax at NA=5, two-x-group form?",
            "reference_pct": 10.54,
            "reference_source": "same E44 contrast value; also not inherited",
        },
        "N1": {
            "control": "iso_m5_ipg3",
            "treated": "iso_m5_ipg5_rb2",
            "width": 5,
            "question": "net cell win of the shippable rb2 route against the shipped route",
            "gate": True,
        },
        "N2": {
            "control": "iso_m5_ipg3",
            "treated": "iso_m5_ipg5_rbx",
            "width": 5,
            "question": "net cell win of the shippable rbx route against the shipped route",
            "gate": True,
        },
    },
    "stop_rules": {
        "rung3_net_cell_win": {
            "rule": (
                "the better of N1 and N2 must be at most -6.0 %. If the best net "
                "cell win is larger than -6.0 % (that is, closer to zero or "
                "positive), stop and report; do not spend a rung 4 allocation."
            ),
            "threshold_pct": -6.0,
            "source": "PR 62 assignment, rung 3 stop rule",
        },
        "route_choice": (
            "carry the mapping with the more negative net cell win into rung 4. If "
            "both clear -6.0 % and their difference is inside the decision bar, "
            "carry rbx, because its 90-register cell leaves the most headroom "
            "under the 108 floor for later composition."
        ),
    },
    "additivity_check": {
        "claim": "net(N1) should equal R1 + T1, and net(N2) should equal R1 + T2",
        "orientation": (
            "percent deltas compose multiplicatively, so the check is "
            "(1+N/100) vs (1+R1/100)*(1+T/100); the residual is reported"
        ),
        "note": "reported as evidence about the cell model; not a gate",
    },
    "measurement_plan": {
        "driver": "research/e59_session.sh -> research/e49_run_leg.sh -> research/run-qmv-curve.sh",
        "widths": "1..10",
        "reps": 21,
        "inner": 10,
        "unchanged_code_widths": "every width except 5 runs byte-identical code in all four arms",
        "unchanged_code_caveat": (
            "the unchanged-code widths share one [[kernel]] allocation with the "
            "treated width, so their movement is a shared-tax readout as well as a "
            "noise bar"
        ),
    },
}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(PREREG, indent=2, sort_keys=True))
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
