#!/usr/bin/env python3
"""Report target-tokenizer token counts for E17 prompt files.

The golden generator uses the first 512 tokens of each prompt, so every timed
prompt must tokenize to at least 512 tokens or the seed window is short.
"""
import sys
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER = (
    Path.home()
    / ".cache/huggingface/hub/models--EigenLabs--Qwen3.8-27B-4bit"
    / "snapshots/eda45ab47f465d08d6558f0353a2346e2eb9d5b3/tokenizer.json"
)


def main(paths: list[str]) -> int:
    tok = Tokenizer.from_file(str(TOKENIZER))
    bad = 0
    for p in sorted(paths):
        text = Path(p).read_text()
        n = len(tok.encode(text, add_special_tokens=False).ids)
        flag = "OK " if n >= 512 else "SHORT"
        if n < 512:
            bad += 1
        print(f"{flag} {n:5d} tokens  {len(text):5d} bytes  {p}")
    return 1 if bad else 0


if __name__ == "__main__":
    args = sys.argv[1:] or [str(p) for p in Path("research").glob("e17_prose_*_512.txt")]
    sys.exit(main(args))
