#!/usr/bin/env python3
"""E133 rung 1: add capture windows where the ranked score actually is.

WHY. F1 section 3 applies Finding 83's marginal prompt weights: `beagle`
carries the first 0.5 term outright, `medicine`/`essays`/`republic`/`botany`
share the second, and `plutarch`/`drama`/`travel` carry exactly zero. F1
section 4 then requires at least 4,000 captured samples in each of the two
GATING strata, which is about eight 512-token windows each at E87's observed
489 samples per window. E124 cut only two beagle windows and six min-carrier
windows, so this script cuts the missing ones at fresh paragraph offsets in
the SAME pinned Gutenberg works.

It reuses `research/e124_corpus.py`'s fetch, body, window and exact-cut rules
verbatim, so a new window differs from an E124 window only in its paragraph
offset. Every window is 512 tokens under the target's own tokenizer.

LEGALITY. Local research fixtures only, exactly as E124. No prompt text, hash,
per-prompt table, detector or seed-derived constant may reach the candidate
surface. The domain labels are the organizer's published labels. Nothing here
reconstructs or approaches a hidden seed.

  python3 research/e133_corpus.py
  python3 research/e133_corpus.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e124_corpus as E124  # noqa: E402

E124_MANIFEST = Path("research/e124-corpus-manifest.json")
OUT_MANIFEST = Path("research/e133-corpus-manifest.json")
OUT_DIR = Path("research")

# Finding 83 marginal weight on the ranked median. `zero_weight` is captured
# for an out-of-band sanity reading and never gates.
STRATUM = {
    "beagle": "beagle",
    "medicine": "min_carriers",
    "essays": "min_carriers",
    "republic": "min_carriers",
    "botany": "min_carriers",
    "plutarch": "zero_weight",
    "drama": "zero_weight",
    "travel": "zero_weight",
}
GATING = ("beagle", "min_carriers")
TARGET_SAMPLES_PER_GATING_STRATUM = 4000

# (gutenberg id, seed stem, domain, paragraph fraction). E124 cut every work
# at 0.15 and beagle also at 0.40, so these fractions are all fresh. A window
# spans roughly ten paragraphs of a body of several hundred, so fractions
# 0.06 apart cannot overlap.
NEW_WINDOWS = [
    (944, "beagle_c", "beagle", 0.22),
    (944, "beagle_d", "beagle", 0.29),
    (944, "beagle_e", "beagle", 0.47),
    (944, "beagle_f", "beagle", 0.55),
    (944, "beagle_g", "beagle", 0.63),
    (944, "beagle_h", "beagle", 0.71),
    # A window yields one sample per draft row, and the measured yield per
    # window ranges from 459 to 563. Eight beagle windows would clear the
    # 4,000-sample floor only if the yield stays near its upper end, so cut two
    # more rather than discover a short stratum after the capture.
    (944, "beagle_i", "beagle", 0.79),
    (944, "beagle_j", "beagle", 0.87),
    (72583, "medicine_hippoc_b", "medicine", 0.40),
    (78966, "medicine_hist_b", "medicine", 0.40),
    (3600, "essays_montaigne_b", "essays", 0.40),
]


def paragraphs(work_id: int) -> list[str]:
    paras = E124.body(E124.fetch(work_id), 200)
    if sum(len(p) for p in paras) < 40000:
        paras = E124.body(E124.fetch(work_id), 80)
    return paras


def cut(tok: Tokenizer, work_id: int, stem: str, domain: str,
        fraction: float) -> dict:
    paras = paragraphs(work_id)
    start = int(len(paras) * fraction)
    made = E124.window(tok, paras, start)
    if made is None:
        raise SystemExit(f"{stem}: not enough prose after paragraph {start}")
    text, used = made
    path = OUT_DIR / f"e133_prose_hi_{stem}_512.txt"
    path.write_text(text)
    ids = tok.encode(text, add_special_tokens=False).ids
    assert len(ids) == E124.SEED_TOKENS, (stem, len(ids))
    return {
        "id": stem,
        "domain": domain,
        "stratum": STRATUM[domain],
        "gutenberg_id": work_id,
        "path": str(path),
        "paragraph_start": used,
        "paragraph_fraction": fraction,
        "chars": len(text),
        "tokens": len(ids),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": "e133",
    }


def inherited() -> list[dict]:
    seeds = []
    for seed in json.loads(E124_MANIFEST.read_text())["seeds"]:
        seeds.append({**seed, "stratum": STRATUM[seed["domain"]],
                      "source": "e124"})
    return seeds


def verify(seeds: list[dict], tok: Tokenizer) -> None:
    seen_text: dict[str, str] = {}
    for seed in seeds:
        raw = Path(seed["path"]).read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != seed["sha256"]:
            raise SystemExit(f"{seed['id']}: sha256 {got} != {seed['sha256']}")
        ids = tok.encode(raw.decode(), add_special_tokens=False).ids
        if len(ids) != E124.SEED_TOKENS:
            raise SystemExit(f"{seed['id']}: {len(ids)} tokens")
        # Two windows that share text would double-count one regime.
        if got in seen_text:
            raise SystemExit(f"{seed['id']} duplicates {seen_text[got]}")
        seen_text[got] = seed["id"]
    print(f"verified {len(seeds)} windows: sha256, 512 tokens, all distinct")


def report(seeds: list[dict], per_window: int) -> None:
    counts: dict[str, int] = {}
    for seed in seeds:
        counts[seed["stratum"]] = counts.get(seed["stratum"], 0) + 1
    print(f"{'stratum':14s}{'windows':>8s}{'projected':>11s}  meets 4,000")
    for stratum in ("beagle", "min_carriers", "zero_weight"):
        n = counts.get(stratum, 0)
        projected = n * per_window
        flag = "" if stratum not in GATING else (
            "yes" if projected >= TARGET_SAMPLES_PER_GATING_STRATUM else "NO")
        print(f"{stratum:14s}{n:8d}{projected:11d}  {flag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the manifest already on disk and stop")
    ap.add_argument("--per-window", type=int, default=489,
                    help="E87's observed samples per 512-token window")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(E124.TOKENIZER))
    if args.check:
        blob = json.loads(OUT_MANIFEST.read_text())
        verify(blob["seeds"], tok)
        report(blob["seeds"], args.per_window)
        return

    seeds = inherited()
    for work_id, stem, domain, fraction in NEW_WINDOWS:
        seed = cut(tok, work_id, stem, domain, fraction)
        seeds.append(seed)
        print(f"  {seed['id']:22s}{seed['domain']:10s}"
              f"para {seed['paragraph_start']:5d}  {seed['chars']:5d} chars")

    verify(seeds, tok)
    OUT_MANIFEST.write_text(json.dumps({
        "seed_tokens": E124.SEED_TOKENS,
        "tokenizer_sha256": hashlib.sha256(E124.TOKENIZER.read_bytes()).hexdigest(),
        "mirror": E124.URL,
        "purpose": "E133 C1 offline screen corpus, stratified by Finding 83 weight",
        "inherits": str(E124_MANIFEST),
        "strata": STRATUM,
        "gating_strata": list(GATING),
        "seeds": seeds,
    }, indent=2) + "\n")
    print(f"wrote {OUT_MANIFEST}")
    report(seeds, args.per_window)


if __name__ == "__main__":
    main()
