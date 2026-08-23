#!/usr/bin/env python3
"""E141 rung 0: how much token mass the compact draft vocabulary cannot propose.

A token id `t` is proposable today if and only if `t < 98_304` or
`248_044 <= t < 248_070` (`Qwen35.swift:5432-5437`). Every other id is a
guaranteed reject that truncates the accepted prefix of its round. This script
measures that mass on the campaign's domain-matched public texts for the two
median carriers (Rule 116: beagle 0.478, essays 0.522, all others 0.000).

It reuses `research/e124_corpus.py`'s fetch, body and prose rules verbatim, so
the text is the same corpus the campaign already uses, and it tokenizes with
the target's own pinned tokenizer.

This is a census, not a price. Rule 107 forbids pricing a shortlist change on a
miss rate; rung 3 measures the realised acceptance delta in situ. The census
exists only to run the rung 0 stop rule and to shape the candidate prefix list.

LEGALITY. Local research fixtures only. No prompt text, hash, per-prompt table,
detector or corpus-derived constant may reach the candidate surface. The domain
labels are the organizer's published labels.

  python3 research/e141_census.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e124_corpus as E124  # noqa: E402

TOKENIZER = Path("weights/tokenizer.json")
OUT = Path("research/e141-census.json")

PREFIX_COUNT = 98_304
CONTROL_START = 248_044
CONTROL_END = 248_070
VOCAB = 248_320

# Rule 116 median-pair identity. Only these two prompts carry weight.
MEDPAIR = {"beagle": 0.478, "essays": 0.522}

# (gutenberg id, published domain label). The beagle and essays works are the
# same ones E124 pinned for these labels.
WORKS = [
    (944, "beagle"),  # Darwin, The Voyage of the Beagle
    (3600, "essays"),  # Montaigne, Essays, complete
]

CANDIDATE_PREFIXES = [98_304, 114_688, 131_072, 147_456, 163_840, 196_608, 248_320]


def proposable(t: int) -> bool:
    return t < PREFIX_COUNT or CONTROL_START <= t < CONTROL_END


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(TOKENIZER))
    report = {
        "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        "prefix_count": PREFIX_COUNT,
        "control_start": CONTROL_START,
        "control_end": CONTROL_END,
        "vocabulary_size": VOCAB,
        "unproposable_id_count": VOCAB - PREFIX_COUNT - (CONTROL_END - CONTROL_START),
        "candidate_prefixes": CANDIDATE_PREFIXES,
        "families": {},
    }

    for work_id, domain in WORKS:
        paras = E124.body(E124.fetch(work_id), 200)
        ids = tok.encode("\n\n".join(paras), add_special_tokens=False).ids
        total = len(ids)
        counts = Counter(ids)

        unprop = Counter({t: c for t, c in counts.items() if not proposable(t)})
        unprop_total = sum(unprop.values())

        # Decoded form of every observed unproposable id, once.
        decoded = {t: tok.decode([t]) for t in unprop}
        ascii_ids = {t for t, s in decoded.items() if s.isascii()}

        curve = []
        for p in CANDIDATE_PREFIXES:
            # Contiguous widening to bound `p` recovers every observed id in
            # [PREFIX_COUNT, p). Ids at or above `p` stay unproposable, and an
            # ASCII-scattered rule on top of `p` would add only the ASCII ones.
            recovered = sum(c for t, c in unprop.items() if t < p)
            ascii_above = sum(c for t, c in unprop.items() if t >= p and t in ascii_ids)
            curve.append(
                {
                    "prefix": p,
                    "recovered_pct": 100.0 * recovered / total,
                    "residual_pct": 100.0 * (unprop_total - recovered) / total,
                    "ascii_above_prefix_pct": 100.0 * ascii_above / total,
                    "padded_count": ((p + CONTROL_END - CONTROL_START + 7) // 8) * 8
                    if p < VOCAB
                    else VOCAB,
                }
            )

        # Smallest contiguous bound that recovers a given share of the census
        # mass. Rung 4 needs this shape: if the mass is spread over the whole
        # tail there is no cheap partial prefix.
        ordered = sorted(unprop.items())
        share_prefix = {}
        for share in (0.50, 0.80, 0.90, 0.95, 1.00):
            need = share * unprop_total
            run = 0
            bound = VOCAB
            for t, c in ordered:
                run += c
                if run >= need:
                    bound = min(VOCAB, t + 1)
                    break
            share_prefix[f"{share:.2f}"] = {
                "prefix": bound,
                "padded_count": ((bound + CONTROL_END - CONTROL_START + 7) // 8) * 8
                if bound < VOCAB
                else VOCAB,
            }

        ascii_mass = sum(c for t, c in unprop.items() if t in ascii_ids)
        top = [
            {
                "id": t,
                "count": c,
                "pct": 100.0 * c / total,
                "decoded": decoded[t],
                "piece": tok.id_to_token(t),
                "ascii": t in ascii_ids,
            }
            for t, c in unprop.most_common(args.top)
        ]

        report["families"][domain] = {
            "gutenberg_id": work_id,
            "medpair_weight": MEDPAIR[domain],
            "paragraphs": len(paras),
            "tokens": total,
            "distinct_ids": len(counts),
            "unproposable_tokens": unprop_total,
            "unproposable_pct": 100.0 * unprop_total / total,
            "unproposable_distinct_ids": len(unprop),
            "unproposable_ascii_pct": 100.0 * ascii_mass / total,
            "unproposable_nonascii_pct": 100.0 * (unprop_total - ascii_mass) / total,
            "cumulative_curve": curve,
            "share_prefix": share_prefix,
            "top_unproposable": top,
        }

    fam = report["families"]
    report["medpair"] = {
        "identity": "0.478 * beagle + 0.522 * essays",
        "recoverable_pct_full_vocabulary": sum(
            MEDPAIR[d] * fam[d]["unproposable_pct"] for d in MEDPAIR
        ),
        "curve": [
            {
                "prefix": p,
                "recovered_pct": sum(
                    MEDPAIR[d] * fam[d]["cumulative_curve"][i]["recovered_pct"]
                    for d in MEDPAIR
                ),
                "ascii_above_prefix_pct": sum(
                    MEDPAIR[d] * fam[d]["cumulative_curve"][i]["ascii_above_prefix_pct"]
                    for d in MEDPAIR
                ),
            }
            for i, p in enumerate(CANDIDATE_PREFIXES)
        ],
    }
    report["stop_rule"] = {
        "threshold_pct": 0.15,
        "measured_pct": report["medpair"]["recoverable_pct_full_vocabulary"],
        "continue": report["medpair"]["recoverable_pct_full_vocabulary"] >= 0.15,
    }

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    for domain, f in fam.items():
        print(f"\n== {domain} (pg{f['gutenberg_id']}, {f['tokens']} tokens) ==")
        print(
            f"  unproposable {f['unproposable_pct']:.4f} %  "
            f"ascii {f['unproposable_ascii_pct']:.4f} %  "
            f"non-ascii {f['unproposable_nonascii_pct']:.4f} %  "
            f"distinct {f['unproposable_distinct_ids']}"
        )
        for row in f["cumulative_curve"]:
            print(
                f"  P={row['prefix']:>7}  recovered {row['recovered_pct']:.4f} %"
                f"  residual {row['residual_pct']:.4f} %"
                f"  ascii-above {row['ascii_above_prefix_pct']:.4f} %"
            )
        print(f"  top {min(args.top, len(f['top_unproposable']))} unproposable:")
        for row in f["top_unproposable"]:
            print(
                f"    {row['id']:>6} x{row['count']:<5} {row['pct']:.4f} %  "
                f"ascii={int(row['ascii'])}  {row['decoded']!r}"
            )

    print("\n== medpair 0.478*beagle + 0.522*essays ==")
    for row in report["medpair"]["curve"]:
        print(
            f"  P={row['prefix']:>7}  recovered {row['recovered_pct']:.4f} %"
            f"  ascii-above {row['ascii_above_prefix_pct']:.4f} %"
        )
    print(
        f"\nrung 0 stop rule: medpair recoverable at full vocabulary "
        f"{report['stop_rule']['measured_pct']:.4f} % vs 0.15 % floor -> "
        f"{'CONTINUE' if report['stop_rule']['continue'] else 'CLOSE'}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
