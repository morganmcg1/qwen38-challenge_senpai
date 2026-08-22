#!/usr/bin/env python3
"""E124 stage 0.5: cut candidate seeds in the organizer's eight domain labels.

WHY. F92 showed the campaign's local prose corpus accepts 0.44-0.52 at depth
~2.5, while 100 % of the published median's marginal weight sits on hidden
prompts accepting 0.83-0.90 at depth 4.4-6.1. An arm comparison run on the
local corpus therefore measures the zero-weight regime. This builds candidate
seeds in the SAME published domain labels so stage 1 can be stratified.

LEGALITY. These are local research fixtures only. No prompt text, hash,
per-prompt table, detector or seed-derived constant may enter the candidate
surface. The domain labels are the organizer's published labels; nothing here
reconstructs, guesses or approaches a hidden seed. Two works Darwin wrote are
used because the published label is `beagle`, not because any hidden text is
known.

Seeds are cut to exactly 512 tokens with the target's own tokenizer, matching
the ranked seed length, and each window starts at a paragraph boundary well
inside the body so front matter never leaks in.

  python3 research/e124_corpus.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1/e124/corpus"))
TOKENIZER = Path("weights/tokenizer.json")
SEED_TOKENS = 512

# (gutenberg id, seed id stem, published domain label, seed count).
# www.gutenberg.org answers 503 from this host, so the pglaf mirror is used.
WORKS = [
    (944, "beagle", "beagle", 2),  # Darwin, The Voyage of the Beagle
    (72583, "medicine_hippoc", "medicine", 1),  # Genuine Works of Hippocrates
    (78966, "medicine_hist", "medicine", 1),  # Singer, A short history of medicine
    (3600, "essays_montaigne", "essays", 1),  # Montaigne, Essays, complete
    (575, "essays_bacon", "essays", 1),  # Bacon, Essays
    (78430, "botany_andrews", "botany", 1),  # Andrews, A practical course in botany
    (1497, "republic_jowett", "republic", 1),  # Plato, The Republic
    (14033, "plutarch_lives", "plutarch", 1),  # Plutarch's Lives, Vol. 1 of 4
    (15492, "drama_dollhouse", "drama", 1),  # Ibsen, A Doll's House
    (43684, "travel_eothen", "travel", 1),  # Kinglake, Eothen
]
URL = "https://gutenberg.pglaf.org/cache/generated/{id}/pg{id}.txt"
START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)


def fetch(work_id: int) -> str:
    raw_path = CACHE / f"pg{work_id}.txt"
    if not raw_path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            URL.format(id=work_id), headers={"User-Agent": "senpai-e124-research"}
        )
        with urllib.request.urlopen(req, timeout=180) as fh:
            raw_path.write_bytes(fh.read())
    return raw_path.read_text(encoding="utf-8", errors="replace")


def is_prose(para: str) -> bool:
    """Reject tables, indexes and price lists.

    A digit-dense paragraph is trivially predictable, so it would inflate
    acceptance without telling us anything about drafting on prose.
    """
    letters = sum(c.isalpha() or c.isspace() for c in para)
    return letters / len(para) > 0.9


def body(text: str, min_para: int) -> list[str]:
    m = START.search(text)
    if m:
        text = text[m.end() :]
    m = END.search(text)
    if m:
        text = text[: m.start()]
    # Collapse the hard-wrapped source into paragraphs: a seed that is mostly
    # newlines measures line-break prediction, not prose prediction.
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if len(p) > min_para and is_prose(p)]


def exact_cut(tok: Tokenizer, ids: list[int]) -> str:
    """Return text that re-encodes to exactly SEED_TOKENS tokens."""
    n = SEED_TOKENS
    got = -1
    for _ in range(16):
        text = tok.decode(ids[:n])
        got = len(tok.encode(text, add_special_tokens=False).ids)
        if got == SEED_TOKENS:
            return text
        n += SEED_TOKENS - got
        if not 0 < n <= len(ids):
            break
    raise SystemExit(f"could not cut a {SEED_TOKENS}-token seed (last try {got})")


def window(tok: Tokenizer, paras: list[str], start: int) -> tuple[str, int] | None:
    chunk: list[str] = []
    ids: list[int] = []
    for para in paras[start:]:
        chunk.append(para)
        ids = tok.encode("\n\n".join(chunk), add_special_tokens=False).ids
        if len(ids) >= SEED_TOKENS + 64:  # headroom for the exact cut search
            break
    if len(ids) < SEED_TOKENS:
        return None
    return exact_cut(tok, ids), start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="research")
    ap.add_argument("--manifest", default="research/e124-corpus-manifest.json")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(TOKENIZER))
    out_dir = Path(args.out_dir)
    manifest = {
        "seed_tokens": SEED_TOKENS,
        "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        "mirror": URL,
        "purpose": "E124 stage 0.5 median-regime candidate corpus (local research only)",
        "seeds": [],
    }

    for work_id, stem, domain, count in WORKS:
        raw = fetch(work_id)
        paras = body(raw, 200)
        # A play is short-line dialogue, so the prose paragraph floor starves
        # it. Fall back before giving up on the domain.
        if sum(len(p) for p in paras) < 40000:
            paras = body(raw, 80)
        for k in range(count):
            seed_id = stem if count == 1 else f"{stem}_{chr(ord('a') + k)}"
            start = int(len(paras) * (0.15 + 0.5 * k / max(1, count)))
            cut = window(tok, paras, start)
            if cut is None:
                print(f"{seed_id:20s} SKIPPED: not enough prose after paragraph {start}")
                continue
            text, used = cut
            path = out_dir / f"e124_prose_hi_{seed_id}_512.txt"
            path.write_text(text)
            manifest["seeds"].append(
                {
                    "id": seed_id,
                    "domain": domain,
                    "gutenberg_id": work_id,
                    "path": str(path),
                    "paragraph_start": used,
                    "chars": len(text),
                    "tokens": len(tok.encode(text, add_special_tokens=False).ids),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            print(f"{seed_id:20s} {domain:9s} {manifest['seeds'][-1]['tokens']:4d} tok  {path}")

    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{len(manifest['seeds'])} seeds -> {args.manifest}")


if __name__ == "__main__":
    main()
