#!/usr/bin/env python3
"""Research-only (qwen38-r1-e215): build the E215 local prompt texts.

`mlxfast-swift generate-golden` tokenizes a prompt file with
`add_special_tokens=False` and then keeps the FIRST
`MLXFastConstants.correctnessPromptTokens` = 512 ids. A prompt that tokenizes
to more than 512 ids is therefore cut mid-sentence, and for a chat prompt the
cut would remove the assistant turn opener the answer has to start from. Each
text this script writes tokenizes to exactly 512 ids, so the seed the harness
uses is the whole text and nothing else.

Three prompts:

  e215-eos-short   one short factual answer, non-thinking form, so the turn
                   ends with `<|im_end|>` (248046) early inside a 512-token
                   decode window. This is the Q1 post-EOS exactness fixture.
  e215-code        a small deterministic coding task: a high-acceptance
                   workload for the Q2 schedule census.
  e215-story       open-ended narrative continuation: a low-acceptance
                   workload for the Q2 schedule census.

usage: research/e215_make_prompts.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from tokenizers import Tokenizer

TARGET_TOKENS = 512
REPO = pathlib.Path(__file__).resolve().parents[1]

# One filler line per padding step. The lines are ordinary English so a
# tokenization surprise is inspectable, and they carry no instruction that
# could change the answer the model gives after the assistant opener.
FILLER = [
    "The notes below are background material for the operator, not part of the question.",
    "The machine that runs this prompt is an Apple Silicon laptop with unified memory.",
    "The decode window is fixed in length and is counted by the trusted parent process.",
    "A stop token inside that window is ordinary data and does not end the window.",
    "The reference trajectory is greedy, so every run of the same build repeats it.",
    "The proposal head suggests tokens and the target model checks each of them.",
    "Rejected tokens are discarded and the accepted prefix is committed in order.",
    "Recurrent layers keep a small state that has to be repaired after a rejection.",
    "Full attention layers keep a key and value cache that grows with the window.",
    "The operator reads the counters after the run and never during the run.",
    "Temperature is recorded before and after each leg for the thermal record.",
    "The numbers in this block are illustrative and carry no measured meaning.",
    "Nothing in this block asks for an answer or changes the question asked below.",
    "The passage exists only to make the seed long enough for the fixed prompt width.",
    "Ordinary English keeps the tokenization stable across small edits to this file.",
    "The last line of the block is followed by the actual question for this turn.",
]

PROMPTS = {
    "e215-eos-short": {
        "system": (
            "You are a precise assistant. Answer in one short sentence. "
            "Do not explain your reasoning, do not add a preface, and do not "
            "add a closing note. Stop as soon as the sentence is complete."
        ),
        "question": "What is the capital city of France?",
    },
    "e215-eos-para": {
        "system": (
            "You are a precise assistant. Answer in three or four sentences of "
            "plain prose. Do not add a preface, a list, or a closing note. "
            "Stop as soon as the answer is complete."
        ),
        "question": (
            "Why does a lighthouse use a rotating lens instead of a brighter "
            "lamp?"
        ),
    },
    "e215-code": {
        "system": (
            "You are a careful Python engineer. Write plain, readable code. "
            "Answer with one code block and no commentary around it."
        ),
        "question": (
            "Write a Python function `merge_sorted(a, b)` that merges two "
            "sorted lists of integers into one sorted list without using "
            "`sorted`. Include a short docstring and simple type hints."
        ),
    },
    "e215-story": {
        "system": (
            "You are a novelist. Write flowing narrative prose. Do not use "
            "lists, headings, or code."
        ),
        "question": (
            "Continue this opening for as long as you like: the lighthouse "
            "keeper found a sealed brass tube in the shallows the morning "
            "after the storm."
        ),
    },
}

# Non-thinking assistant opener. The template's thinking form would spend the
# whole decode window inside a reasoning block, which would defeat the point of
# the EOS fixture, so the empty think block is written into the seed directly.
ASSISTANT_OPENER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_text(system: str, question: str, filler_lines: int, tail_words: int) -> str:
    lines = FILLER * ((filler_lines // len(FILLER)) + 1)
    block = "\n".join(lines[:filler_lines])
    if tail_words:
        block = block + "\n" + " ".join(["context"] * tail_words)
    return (
        "<|im_start|>system\n"
        + system
        + "<|im_end|>\n<|im_start|>user\n"
        + "Reference notes:\n"
        + block
        + "\n\nQuestion: "
        + question
        + "<|im_end|>\n"
        + ASSISTANT_OPENER
    )


def fit(tokenizer: Tokenizer, system: str, question: str) -> str:
    def length(filler_lines: int, tail_words: int) -> int:
        text = build_text(system, question, filler_lines, tail_words)
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    filler_lines = 0
    while length(filler_lines + 1, 0) <= TARGET_TOKENS:
        filler_lines += 1
    tail_words = 0
    while length(filler_lines, tail_words + 1) <= TARGET_TOKENS:
        tail_words += 1
    text = build_text(system, question, filler_lines, tail_words)
    got = length(filler_lines, tail_words)
    if got != TARGET_TOKENS:
        raise SystemExit(
            f"could not hit {TARGET_TOKENS} tokens exactly: closest is {got}"
        )
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "research/e215-artifacts/prompts"))
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(REPO / "weights/tokenizer.json"))
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, spec in PROMPTS.items():
        text = fit(tokenizer, spec["system"], spec["question"])
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        assert len(ids) == TARGET_TOKENS
        path = out_dir / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        manifest[name] = {
            "path": str(path.relative_to(REPO)),
            "prompt_tokens": len(ids),
            "last_ids": ids[-8:],
            "question": spec["question"],
        }
        print(f"{name}: {len(ids)} tokens -> {path.relative_to(REPO)}")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
