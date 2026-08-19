#!/usr/bin/env python3
"""Emit the Markdown counter table for research/e25-results.md from e25-phase1.json."""
import json
import pathlib
import sys

PROMPTS = [
    "english",
    "narrative",
    "technical",
    "dramatic",
    "travel",
    "philosophy",
    "natural_history",
    "medicine",
]

payload = json.loads(pathlib.Path("research/e25-phase1.json").read_text())
timed = payload["reduced"]["timed_runs"]

hdr = (
    "| prompt | arm | rounds | declared rows | accepted | rejected | accept rate "
    "| mean depth | max depth | non-drafting | true decode s | replayed | p50 block s |"
)
sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
lines = [hdr, sep]
missing = []
for prompt in PROMPTS:
    for arm in ("BASE", "PRICE"):
        run = timed.get(arm, {}).get(prompt)
        if not run:
            missing.append((arm, prompt))
            continue
        lines.append(
            "| {p} | {a} | {rnd} | {rows} | {acc} | {rej} | {rate:.4f} | {mean:.4f} "
            "| {mx} | {nodr} | {td:.4f} | {repl} | {blk:.5f} |".format(
                p=prompt,
                a=arm,
                rnd=run["mtp_rounds"],
                rows=run["mtp_declared_rows"],
                acc=run["mtp_accepted_rows"],
                rej=run["mtp_rejected_rows"],
                rate=run["mtp_accepted_rate"],
                mean=run["mtp_mean_depth"],
                mx=run["mtp_max_depth"],
                nodr=run["mtp_non_drafting_rounds"],
                td=run["mtp_true_decode"],
                repl=run["mtp_replayed_rounds"],
                blk=run["mtp_p50_block_seconds"],
            )
        )

print("\n".join(lines))
if missing:
    print("\nMISSING: " + ", ".join(f"{a}/{p}" for a, p in missing), file=sys.stderr)
    sys.exit(1)
