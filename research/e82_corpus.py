#!/usr/bin/env python3
"""E82 rung 0, step 5: build the offline acceptance-screen corpus.

Constraints this satisfies, from the assignment:

  * public-domain prose the campaign has never evaluated on. The eight ranked
    pool prompts are a naturalist voyage narrative, a medical text, Plutarch, a
    drama, a travelogue, an essay collection, the Republic and a botany text,
    so none of those authors, works or genres appear here;
  * no public fixture prompt, no golden, no GPQA item;
  * at least two domains, because domain sensitivity is the recorded failure
    mode of every head retrain on the campaign stop list.

Seeds are cut to exactly 512 tokens with the target's own tokenizer, matching
the ranked seed length, and each seed starts at a paragraph boundary well
inside the book so front matter never leaks in.

  python3 research/e82_corpus.py --seeds-per-work 2
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

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1/e82/corpus"))
TOKENIZER = Path("weights/tokenizer.json")
SEED_TOKENS = 512

# (gutenberg id, slug, domain). Deliberately disjoint from the ranked pool's
# genres: no naturalist voyage, medicine, Plutarch, drama, travelogue, essay
# collection, Republic or botany.
WORKS = [
    (2701, "mobydick", "narrative"),
    (84, "frankenstein", "narrative"),
    (1342, "pride", "narrative"),
    (3300, "wealthofnations", "expository"),
    (5827, "problemsofphilosophy", "expository"),
    (14474, "candle", "expository"),
]
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)


def fetch(work_id: int) -> str:
    raw_path = CACHE / f"pg{work_id}.txt"
    if not raw_path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL.format(id=work_id), headers={"User-Agent": "senpai-e82-research"})
        with urllib.request.urlopen(req, timeout=120) as fh:
            raw_path.write_bytes(fh.read())
    return raw_path.read_text(encoding="utf-8", errors="replace")


def body(text: str) -> str:
    m = START.search(text)
    if m:
        text = text[m.end() :]
    m = END.search(text)
    if m:
        text = text[: m.start()]
    # Collapse the hard-wrapped source into paragraphs: a seed that is mostly
    # newlines measures line-break prediction, not prose prediction.
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paras if len(p) > 200 and is_prose(p))


def is_prose(para: str) -> bool:
    """Reject tables, indexes and price lists.

    A digit-dense paragraph is trivially predictable, so it would inflate the
    easiest tercile without telling us anything about drafting on prose.
    """
    letters = sum(c.isalpha() or c.isspace() for c in para)
    return letters / len(para) > 0.9


def exact_cut(tok: Tokenizer, ids: list[int]) -> str:
    """Return text that re-encodes to exactly SEED_TOKENS tokens.

    Decoding a token prefix and re-encoding it is not the identity, so the
    cut point is searched instead of assumed. Every arm reads the same seed
    file, so the only requirement is that the ranked 512-token seed length is
    reproduced exactly.
    """
    n = SEED_TOKENS
    for _ in range(16):
        text = tok.decode(ids[:n])
        got = len(tok.encode(text, add_special_tokens=False).ids)
        if got == SEED_TOKENS:
            return text
        n += SEED_TOKENS - got
        if not 0 < n <= len(ids):
            break
    raise SystemExit(f"could not cut a {SEED_TOKENS}-token seed (last try {got})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-per-work", type=int, default=2)
    ap.add_argument("--out-dir", default=str(CACHE / "seeds"))
    ap.add_argument("--manifest", default="research/e82-corpus-manifest.json")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(TOKENIZER))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed_tokens": SEED_TOKENS, "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(), "seeds": []}

    for work_id, slug, domain in WORKS:
        prose = body(fetch(work_id))
        paras = prose.split("\n\n")
        # Spread the seeds across the middle 80 % of each work.
        for k in range(args.seeds_per_work):
            start = int(len(paras) * (0.1 + 0.8 * k / max(1, args.seeds_per_work)))
            chunk, ids = [], []
            for para in paras[start:]:
                chunk.append(para)
                ids = tok.encode("\n\n".join(chunk), add_special_tokens=False).ids
                if len(ids) >= SEED_TOKENS + 64:  # headroom for the exact cut search
                    break
            if len(ids) < SEED_TOKENS:
                raise SystemExit(f"{slug}: not enough text after paragraph {start}")
            text = exact_cut(tok, ids)
            name = f"{slug}-{k}"
            path = out_dir / f"e82-{name}.txt"
            path.write_text(text)
            manifest["seeds"].append(
                {
                    "name": name,
                    "domain": domain,
                    "gutenberg_id": work_id,
                    "path": str(path),
                    "chars": len(text),
                    "tokens": len(tok.encode(text, add_special_tokens=False).ids),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            print(f"{name:28s} {domain:11s} {manifest['seeds'][-1]['tokens']:4d} tok  {path}")

    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest['seeds'])} seeds -> {args.manifest}")


if __name__ == "__main__":
    main()
