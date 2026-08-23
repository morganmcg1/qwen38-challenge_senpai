#!/usr/bin/env python3
"""E148 R-D: filter the corrected board for mechanisms we can still claim.

R-C produced a corrected, cohort-local price for every priceable rival row.
R-D turns that priced list into an actionable one by removing every row whose
mechanism we cannot or should not take:

  F1 integrity    the note must not declare a benchmark-escape mechanism
  F2 scope        every source file the note names must lie inside the
                  benchmark.json editable surface
  F3 not-landed   the mechanism's own code identifiers must be absent from our
                  maintained tree (if they are present, we already have it)
  F4 unowned      the named files must be outside another student's live area
  F5 novel-to-us  the mechanism must not already appear in the campaign ledger

Every percentage is harness=ranked, cohort-local, total-leg frame, weighted
five prompts, corrected by the E148 state-step corrector.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
REPO = HERE.parent

RC = Path(__file__).resolve().parent / "e148-rc.json"
OUT = Path(__file__).resolve().parent / "e148-rd.json"
BOARD = Path("/tmp/yukon-board/full.json")
LEDGER = REPO / "senpai" / "campaign-ledger.md"

# Rows this campaign has already claimed, by E-number, so they cannot be
# reported as unclaimed opportunities.
CLAIMED_ROWS = {
    "43925f29": "E147 (double-buffer the affine NAX transposed GEMM)",
    "5cdc9c17": "E145/E147 prefill lineage (BitWonka NAX retile)",
    "9b241879": "our own senpai falsification arm",
}

# Live per-student areas in this launch. A mechanism landing here is a hand-off,
# not an unclaimed candidate for E148.
OWNERSHIP = [
    (re.compile(r"Qwen36MTPBlockSession"), "edward"),
    (re.compile(r"quantized_nax\.h"), "alphonse"),
    (re.compile(r"quantized\.(h|cpp)"), "alphonse"),
    (re.compile(r"mlx-generated/quantized"), "alphonse"),
    (re.compile(r"Qwen35\.swift"), "thorfinn"),
    (re.compile(r"Qwen35Model|Qwen35TextModel"), "thorfinn"),
]

# A note can describe a student's live area without naming its file, so match
# the mechanism vocabulary too.
OWNERSHIP_BY_MECHANISM = [
    (re.compile(r"block session|draft depth|drafting depth|acceptance rule|"
                r"streak gate|rollback|state journal|cache journal|repair",
                re.IGNORECASE), "edward"),
    (re.compile(r"\bqmv\b|\bqmm\b|\bNAX\b|\bIPG\b|nibble|dequant|affine-[24]|"
                r"group-64|quantized kernel", re.IGNORECASE), "alphonse"),
    (re.compile(r"gated[- ]delta|\bGDN\b|RMSNorm|SDPA|full[- ]attention|"
                r"recurrent state", re.IGNORECASE), "thorfinn"),
]

# Terms the correctness and work-honesty section of program.md forbids.
INTEGRITY_TERMS = [
    "prompt lookup",
    "n-gram",
    "ngram",
    "suffix automaton",
    "suffix array",
    "suffix tree",
    "hidden prompt",
    "prompt pool",
    "across requests",
    "cross-request",
    "benchmark phase",
    "phase detection",
    "detect the reference",
    "token-history",
    "prompt-specific",
]

PATH_RE = re.compile(r"[A-Za-z0-9_./+-]+\.(?:swift|h|hpp|cpp|metal|json|py|sh)\b")
TICK_RE = re.compile(r"`([^`\n]{3,120})`")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{5,}$")
HEX_RE = re.compile(r"^[0-9a-f]{7,}$")

STOPWORDS = {
    "submission", "submissions", "benchmark", "candidate", "baseline", "official",
    "promoted", "rejected", "accepted", "frontier", "decode", "prefill", "serial",
    "speedup", "seconds", "tokens", "prompts", "median", "kernel", "kernels",
    "record", "records", "current", "release", "archive", "archives", "mechanism",
    "editable", "editablePaths", "origin", "main", "commit", "commits", "source",
    "target", "proposal", "verify", "accept", "reject", "rollback", "residency",
    "geometry", "profile", "profiles", "buffer", "buffers", "warmup", "warmups",
}


def load_source_corpus() -> str:
    """Concatenate every tracked source byte the candidate may ship."""
    listed = subprocess.run(
        ["git", "ls-files", "--", "Sources", "Vendor", "mtp-head.manifest.json"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split()
    parts = []
    for rel in listed:
        p = REPO / rel
        try:
            parts.append(p.read_text(errors="ignore"))
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(parts)


def load_ledger() -> str:
    return LEDGER.read_text(errors="ignore")


def editable_prefixes() -> list[str]:
    bench = json.loads((REPO / "benchmark.json").read_text())
    paths = list(bench["editablePaths"]) + list(bench["optionalEditablePaths"])
    return [p.rstrip("/") for p in paths]


def note_identifiers(note: str) -> list[str]:
    """Distinctive code identifiers the note quotes in backticks."""
    out: list[str] = []
    for raw in TICK_RE.findall(note):
        # Drop template arguments and call parentheses; keep the symbol.
        sym = re.split(r"[<(\[]", raw.strip())[0].strip()
        sym = sym.split("::")[-1].split(".")[-1]
        if not IDENT_RE.match(sym):
            continue
        if HEX_RE.match(sym):
            continue
        low = sym.lower()
        if low in STOPWORDS:
            continue
        has_underscore = "_" in sym
        has_camel = bool(re.search(r"[a-z][A-Z]", sym))
        if not (has_underscore or has_camel):
            continue
        if sym not in out:
            out.append(sym)
    return out


def note_paths(note: str) -> list[str]:
    out: list[str] = []
    for m in PATH_RE.finditer(note):
        p = m.group(0).lstrip("./")
        if p.endswith((".py", ".sh")):
            continue
        if p not in out:
            out.append(p)
    return out


def owners_for(paths: list[str], note: str) -> tuple[list[str], list[str]]:
    """Return (file-based owners, prose-only owners).

    File-based ownership is the hard signal: taking the mechanism means editing
    that student's live file. Mechanism vocabulary is advisory only, because
    almost every serious kernel note says "rollback", "GDN" or "qmv" in passing.
    """
    by_file: list[str] = []
    hay_paths = " ".join(paths)
    for rx, who in OWNERSHIP:
        if rx.search(hay_paths) and who not in by_file:
            by_file.append(who)
    by_prose: list[str] = []
    for rx, who in OWNERSHIP_BY_MECHANISM:
        if who in by_file or who in by_prose:
            continue
        if rx.search(note):
            by_prose.append(who)
    return by_file, by_prose


def scope_report(paths: list[str], prefixes: list[str], tracked: dict[str, list[str]]) -> dict:
    """Resolve each named path against the real tree, then test the editable surface."""
    outside, unresolved, inside = [], [], []
    for p in paths:
        cands = tracked.get(p.split("/")[-1], [])
        matched = [t for t in cands if t.endswith(p) or t == p]
        if not matched:
            unresolved.append(p)
            continue
        if any(any(t == pre or t.startswith(pre + "/") for pre in prefixes) for t in matched):
            inside.append(p)
        else:
            outside.append(p)
    return {
        "paths_inside_editable_surface": inside[:10],
        "paths_outside_editable_surface": outside[:10],
        "paths_not_in_this_tree": unresolved[:10],
    }


NEGATION = re.compile(
    r"\b(no|not|never|without|neither|nor|excludes?|forbids?|prohibits?|free of)\b",
    re.IGNORECASE,
)


def integrity_flags(note: str) -> list[dict]:
    """Report only claims that are NOT part of a disclaimer sentence.

    Rival notes routinely recite the banned-mechanism list to declare that they
    use none of it, so a bare term match is almost always a false positive.
    """
    flags = []
    low = note.lower()
    for term in INTEGRITY_TERMS:
        start_at = 0
        while True:
            i = low.find(term, start_at)
            if i < 0:
                break
            start_at = i + len(term)
            s = max(0, low.rfind(".", 0, i) + 1)
            e = low.find(".", i)
            e = len(note) if e < 0 else e + 1
            sentence = note[s:e].strip()
            prefix = sentence[: max(0, i - s)]
            if NEGATION.search(prefix):
                continue
            flags.append({"term": term, "sentence": sentence[:300], "negated": False})
    return flags


def landed_report(idents: list[str], corpus: str) -> dict:
    found = [s for s in idents if s in corpus]
    missing = [s for s in idents if s not in corpus]
    n = len(idents)
    return {
        "identifiers": n,
        "identifiers_in_our_tree": len(found),
        "landed_fraction": round(len(found) / n, 4) if n else None,
        "found_sample": found[:8],
        "missing_sample": missing[:8],
    }


def ledger_report(idents: list[str], ledger: str) -> dict:
    found = [s for s in idents if s in ledger]
    n = len(idents)
    return {
        "identifiers_in_ledger": len(found),
        "ledger_fraction": round(len(found) / n, 4) if n else None,
        "ledger_sample": found[:8],
    }


def resolver_failure_modes() -> dict:
    """F2.6: how the note-text parent resolver fails, not only how often it hits.

    `promotedSourceRef` is the row's own promoted tree, never a parent pointer,
    so for the rejected rows that hold the unclaimed mechanisms the note text is
    the only channel. A hit rate alone hides which way it breaks.
    """
    import e146_lib as L  # noqa: PLC0415
    import e148_lib as E  # noqa: PLC0415

    _, rows = E.load_rows()
    by_id = {r.id8: r for r in rows}
    trees = E.tree_index(rows)
    _, cohorts = E.build_cohorts(rows)
    hex_token = E.HEX_TOKEN

    census = {
        "rows": len(rows),
        "names_no_hex_token_at_all": 0,
        "names_hex_but_none_resolves": 0,
        "names_exactly_one_resolvable_tree": 0,
        "names_more_than_one_resolvable_tree": 0,
        "resolved_same_cohort": 0,
        "resolved_cross_schedule": 0,
        "unresolved": 0,
        "ambiguity_broken_by_score": 0,
    }
    multi_examples = []
    for row in rows:
        note = row.note or ""
        tokens = set(hex_token.findall(note))
        cited = E.cited_parents(row, trees, by_id)
        if not tokens:
            census["names_no_hex_token_at_all"] += 1
        elif not cited:
            census["names_hex_but_none_resolves"] += 1
        elif len(cited) == 1:
            census["names_exactly_one_resolvable_tree"] += 1
        else:
            census["names_more_than_one_resolvable_tree"] += 1
            if len(multi_examples) < 8:
                multi_examples.append({
                    "row": row.id8, "candidates": [c.id8 for c in cited[:6]],
                    "chosen": None,
                })
        parent, kind = E.resolve_parent(row, trees, by_id, cohorts)
        if kind == "cited-same-cohort":
            census["resolved_same_cohort"] += 1
        elif kind == "cited-cross-schedule":
            census["resolved_cross_schedule"] += 1
        else:
            census["unresolved"] += 1
        if parent is not None and len(cited) > 1:
            same = [c for c in cited if E.cohort_key(c) == E.cohort_key(row)]
            if len(same) > 1:
                census["ambiguity_broken_by_score"] += 1
            for ex in multi_examples:
                if ex["row"] == row.id8:
                    ex["chosen"] = parent.id8
    census["multi_citation_examples"] = multi_examples
    census["note"] = (
        "A note that names more than one resolvable tree is the resolver's main "
        "exposure. The tie is broken by preferring a same-cohort citation and "
        "then the highest published score, which is the tree a solver is most "
        "likely to have forked. That heuristic is unverifiable per row. It was "
        "checked on the two hardest control cases in R-B, a4ac742e -> 48423d09 "
        "and e987b29b -> 51b9bf85, and it got both right, but "
        "%d rows depend on it and no per-row proof exists."
        % census["ambiguity_broken_by_score"])
    assert len(L.PROMPT_ORDER) == 8
    return census


# Hand adjudication. Every entry below was decided by reading this checkout's
# source at the cited line, not by matching note prose. `verdict` is one of
# landed / precondition-absent / claimed / open. The automated filters stay in
# place for every row this table does not name.
HAND: dict[str, dict[str, object]] = {
    "f5830389": {
        "verdict": "landed",
        "mechanism": "M=8 affine-4 g64 crossrow QMV at IPG 4",
        "source": ["Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/"
                   "kernels/quantized.h:1968"],
        "evidence": "case 8 dispatches qmv_fast_crossrow_affine4_g64_m"
                    "<T, 8, 4, true>; the IPG 3 -> 4 change is already the "
                    "live template argument. The comment above it still says "
                    "'3+3+2, not 4+4' and is stale against its own code.",
    },
    "444f9767": {
        "verdict": "landed",
        "mechanism": "top-32 shortlist kernels plus the boundary residual/"
                     "RMSNorm fusion",
        "source": ["Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/"
                   "Qwen35.swift:4408", "Vendor/mlx-swift-lm/Libraries/MLXLLM/"
                   "Models/Qwen35.swift:2529"],
        "evidence": "qwen35DraftTop32PartialKernel and "
                    "qwen35FusedResidualRMSNormKernel both exist and are live "
                    "at Qwen35.swift:3715, :3746 and :3762.",
    },
    "06e8c9d4": {
        "verdict": "landed",
        "mechanism": "lazy exact prefix replay for multi-draft GDN rollback",
        "source": ["Sources/MLXFastModel/Qwen36MTPBlockSession.swift:1927",
                   "Sources/MLXFastModel/Qwen36MTPBlockSession.swift:1994",
                   "Sources/MLXFastModel/Qwen36MTPBlockSession.swift:2020"],
        "evidence": "prefixReplayTape is a live field on the round arrays and "
                    "is released at all three rollback exits.",
    },
    "578535f7": {
        "verdict": "landed",
        "mechanism": "M=8 4+4 QMV combine",
        "source": ["Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/"
                   "kernels/quantized.h:1968"],
        "evidence": "Same landed cell as f5830389: the even 4+4 split is what "
                    "IPG 4 selects inside qmv_fast_crossrow_affine4_g64_wide.",
    },
    "3ec77796": {
        "verdict": "landed",
        "mechanism": "residency wiring plus command-buffer geometry",
        "source": ["Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift"],
        "evidence": "residencySet appears in 18 files and MTLResidency in 6; "
                    "ledger item 12 records our variant as the stronger one "
                    "(setenv overwrite 1 against the rival's 0).",
    },
    "070f1189": {
        "verdict": "landed",
        "mechanism": "boundary-fused residual/RMSNorm chain",
        "source": ["Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/"
                   "Qwen35.swift:2529"],
        "evidence": "qwen35FusedResidualRMSNorm is live at :3715, :3746, "
                    ":3762.",
    },
    "942e5ab2": {
        "verdict": "landed",
        "mechanism": "fused residual/RMSNorm",
        "source": ["Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/"
                   "Qwen35.swift:2529"],
        "evidence": "Same landed kernel as 070f1189.",
    },
    "22ce3162": {
        "verdict": "landed",
        "mechanism": "packed GDN prework mixer, five outputs, one launch, "
                     "verify widths 3...9",
        "source": ["Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/"
                   "Qwen35.swift:276", "Vendor/mlx-swift-lm/Libraries/MLXLLM/"
                   "Models/Qwen35.swift:967"],
        "evidence": "qwen35PackedGDNPreworkKernel is built at :276, named "
                    "qwen35_packed_gdn_prework at :419, and dispatched at "
                    ":967 behind the fail-closed gate at :948.",
    },
    "b28b0993": {
        "verdict": "landed",
        "mechanism": "island fast path plus the affine-4 embedding dequant "
                     "folded into the dual-RMSNorm-concat launch",
        "source": ["Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/"
                   "Qwen35.swift:3266", "Vendor/mlx-swift-lm/Libraries/MLXLLM/"
                   "Models/Qwen35.swift:2893"],
        "evidence": "Its note names one fusion, the embedding fold, and we "
                    "ship it as qwen35_embed_dual_rms_norm_concat_bf16_v1. "
                    "The island fast path is live at :3266 and :3339 behind "
                    "islandFastPathReady() at :3424.",
    },
    "5c523482": {
        "verdict": "landed",
        "mechanism": "emit exact float32 recurrence beta from the packed GDN "
                     "prework kernel, removing the separate [1,S,48] graph "
                     "sigmoid launch at verify widths 3...9",
        "source": ["Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/"
                   "Qwen35.swift:424", "Vendor/mlx-swift-lm/Libraries/MLXLLM/"
                   "Models/Qwen35.swift:280"],
        "evidence": "Our kernel declares six outputs including beta_out, and "
                    "the header at :265 states it removes the final [1,S,48] "
                    "elementwise launch. We are ahead of the rival here: "
                    "22ce3162 kept beta OUT of its kernel because the "
                    "in-kernel sigmoid diverges from MLX by 1 ulp on the "
                    "single bf16 input 0xC0DB. qwen35_prework_beta at :280 "
                    "maps that input to MLX's output word 0x3A8B and keeps "
                    "the inherited expression everywhere else, so the "
                    "rival's stated blocker is resolved in our source. The "
                    "eager sigmoid sites left at :183 and :746 are the S=1 "
                    "and S=2 fallbacks the S>=3 gate at :948 excludes.",
        "caveat": "Its -2.69 % prefill reading is the prefill channel's own "
                  "dispersion on a 2.93 cohort, not a mechanism (advisor "
                  "F3.4). Dropped from the prefill list. Banked negative from "
                  "the same row: reducing eval(cache state + bundle) to "
                  "eval(bundle) passed parity and was officially negative.",
    },
    "4debb1df": {
        "verdict": "landed",
        "mechanism": "declared same-content resample, no mechanism",
        "source": [],
        "evidence": "R-B verified an empty git diff of Vendor/ and Sources/ "
                    "against the promoted tip eb5eadc7. It is a null control, "
                    "not a candidate.",
    },
    "142c41bd": {
        "verdict": "landed",
        "mechanism": "isolated eval-root trim",
        "source": ["Sources/MLXFastModel/Qwen36MTPBlockSession.swift:1648"],
        "evidence": "The round already carries a single blocking eval whose "
                    "bundle is the minimal root set.",
    },
    "d20d1c13": {
        "verdict": "precondition-absent",
        "mechanism": "skip the unused probe-sort JIT while E87 is the live "
                     "select arm",
        "source": ["Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/"
                   "Qwen35.swift:6245", "Vendor/mlx-swift-lm/Libraries/MLXLLM/"
                   "Models/Qwen35.swift:6269"],
        "evidence": "clusterCandidateIDs builds _draftProbeSort at :6245 and "
                    "READS it at :6269 to reorder the probe indices before "
                    "the row score. The dead-work precondition the rival "
                    "measured does not hold in this tree.",
    },
    "7fbb504f": {
        "verdict": "claimed",
        "mechanism": "complete the later-window SDPA warm set to "
                     "qL in {1,2,3,4,5}",
        "source": ["Sources/MLXFastModel/Qwen36MTPBlockSession.swift:638",
                   "Sources/MLXFastModel/Qwen36MTPBlockSession.swift:664"],
        "evidence": "The gap is real: both warm loops read `for qL in "
                    "[1, 5, 4]` and segmentedVerifyDepthCap = 7 makes verify "
                    "qL 7 and 8 live, whose chunk B is qL 2 and 3. But "
                    "ledger 179(E) (line 8561) already found this from "
                    "source and ledger line 32536 CLOSED it: an ABBA pair put "
                    "the candidate leg at +0.0057 %, sd 0.0875, a null, and "
                    "put the -0.61 % published headline down to serial "
                    "lottery. STOP LIST. Reopening condition unchanged.",
    },
    "214d92aa": {
        "verdict": "open",
        "mechanism": "combine the coarse top-32 finalization and the selected "
                     "affine-4 rerank into one dispatch",
        "source": ["Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/"
                   "Qwen35.swift:4508", "Vendor/mlx-swift-lm/Libraries/MLXLLM/"
                   "Models/Qwen35.swift:6374"],
        "evidence": "The rival note lists TWO exact proposal-dispatch "
                    "reductions. Reduction 1, the affine-4 embedding dequant "
                    "fold, is ours at Qwen35.swift:2893. Reduction 2 is not: "
                    "draftTokenIDWithDeclaredRerank still runs a finalize "
                    "dispatch and then a separate "
                    "qwen35DraftSelectedAffine4RerankKernel at :6374. Ledger "
                    "236.3 (line 25720) priced this row as reduction 1 alone "
                    "and concluded 'We already ship this', so reduction 2 was "
                    "never adjudicated and ADVISOR ERROR 30's 1.9x repricing "
                    "of E85 may be a two-mechanism total.",
        "caveat": "On our declared cluster head the live arm is "
                  "clusterCandidateIDs -> Qwen35RowTop32 -> rerank, so the "
                  "rival's exact dense-coarse finalize is dead here. The "
                  "fusion CLASS applies to the clustered finalize; the exact "
                  "kernel does not port unchanged.",
    },
    "d5e94249": {
        "verdict": "open",
        "mechanism": "fold the GDN q/k scale constant into the RMSNorm weight "
                     "tensor, collapsing two launches into one",
        "source": ["Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/"
                   "Qwen35.swift:910", "Vendor/mlx-swift-lm/Libraries/MLXLLM/"
                   "Models/Qwen35.swift:1009", "Vendor/mlx-swift-lm/Libraries/"
                   "MLXLLM/Models/Qwen35.swift:1208"],
        "evidence": "All three eager sites read qScaleConst * "
                    "MLXFast.rmsNorm(q, weight: .mlxNone, eps: 1e-6), so each "
                    "pays a scalar-multiply launch after the norm launch. "
                    "Ledger item 9 (line 9476) closed the SIBLING form, "
                    "3ac231d5's baked bf16 compile-time immediates, as a "
                    "measured M5 null at -0.0003 for two reasons: only two "
                    "encoder binds were left as prize, and a compile-time "
                    "constant licenses reassociation and 2^-7 strength "
                    "reduction. Neither reason survives the weight-fold form, "
                    "which keeps the scale in a buffer and removes a whole "
                    "launch. Reopening condition CHANGED.",
        "caveat": "The packed GDN prework mixer already fuses q/k "
                  "rmsNorm-and-scale for S in 3...9, so only S=1, S=2 and "
                  "prefill reach these sites. The carrier row is decode-"
                  "REFUSED at steps_exact 1.026; its only priceable channel "
                  "is prefill at -0.4422 %, which is roughly -0.02 to -0.04 % "
                  "of the total leg and sits under the 0.1154 pp MDE.",
    },
}



def main() -> None:
    rc = json.loads(RC.read_text())
    board = json.loads(BOARD.read_text())
    board = board["submissions"] if isinstance(board, dict) else board
    notes = {r["id"][:8]: (r.get("note") or "") for r in board}

    corpus = load_source_corpus()
    ledger = load_ledger()
    prefixes = editable_prefixes()

    tracked: dict[str, list[str]] = {}
    for rel in subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.split():
        tracked.setdefault(rel.split("/")[-1], []).append(rel)

    threshold = -0.10
    priced = [r for r in rc["decode_ranked_table"]
              if r["corrected_total_pct"] <= threshold]
    priced.sort(key=lambda r: r["corrected_total_pct"])

    records = []
    for r in priced:
        row = r["row"]
        note = notes.get(row, "")
        idents = note_identifiers(note)
        paths = note_paths(note)
        landed = landed_report(idents, corpus)
        ledg = ledger_report(idents, ledger)
        owners, owners_prose = owners_for(paths, note)
        scope = scope_report(paths, prefixes, tracked)
        flags = integrity_flags(note)

        # Hard blockers only. Scope prose, ledger word overlap and disclaimer
        # sentences are advisory signals and are reported, not used to reject.
        verdict = []
        if row in CLAIMED_ROWS:
            verdict.append("claimed")
        if scope["paths_outside_editable_surface"]:
            verdict.append("out-of-scope-path")
        if landed["landed_fraction"] is None or landed["identifiers"] < 3:
            verdict.append("undetermined-landing")
        elif landed["landed_fraction"] >= 0.60:
            verdict.append("already-in-our-tree")
        if owners:
            verdict.append("owned:" + "+".join(owners))

        records.append({
            "row": row,
            "solver": r["solver"],
            "score": r["score"],
            "promotion": r["promotion"],
            "parent": r["parent"],
            "parent_score": r["parent_score"],
            "corrected_total_pct": round(r["corrected_total_pct"], 4),
            "median_pair_pct": (round(r["median_pair_pct"], 4)
                                if r["median_pair_pct"] is not None else None),
            "raw_total_pct": round(r["raw_total_pct"], 4),
            "price_sd_pp": round(r["price_sd_pp"], 4) if r["price_sd_pp"] else None,
            "title": r["title"][:180],
            "note_paths": paths[:10],
            "owners_by_file": owners,
            "owners_by_mechanism_prose": owners_prose,
            "scope": scope,
            "integrity_flags": flags,
            "landing": landed,
            "ledger": ledg,
            "blockers": verdict,
            "unclaimed": not verdict,
        })

    for x in records:
        h = HAND.get(x["row"])
        if h is None:
            continue
        x["hand_adjudication"] = h
        x["unclaimed"] = h["verdict"] == "open"
        x["blockers"] = [] if x["unclaimed"] else [f'hand:{h["verdict"]}']

    unclaimed = [x for x in records if x["unclaimed"]]
    best = min((x["corrected_total_pct"] for x in unclaimed), default=0.0)

    # d5e94249 is decode-REFUSED (steps_exact 1.026) so it never reaches the
    # priced table, but its mechanism is open and its prefill channel is
    # priceable. Carry it as an out-of-band candidate with a null decode price.
    refused = {r["row"]: r for r in rc["decode_refused_table"]}
    extra = []
    for row, h in HAND.items():
        if h["verdict"] != "open" or any(x["row"] == row for x in unclaimed):
            continue
        src = refused.get(row)
        extra.append({
            "row": row,
            "priced_channel": "prefill only (decode refused)",
            "score": src["score"] if src else None,
            "parent": src["parent"] if src else None,
            "steps_exact": round(src["steps_exact"], 4) if src else None,
            "corrected_total_pct": None,
            "prefill_pct": round(src["prefill_pct"], 4) if src else None,
            "hand_adjudication": h,
        })

    blocker_census: dict[str, int] = {}
    for x in records:
        for b in x["blockers"]:
            key = b.split(":")[0]
            blocker_census[key] = blocker_census.get(key, 0) + 1

    out = {
        "harness": "ranked",
        "frame": "cohort-local, total-leg, weighted-five, state-corrected",
        "resolver_failure_modes": resolver_failure_modes(),
        "threshold_pct": threshold,
        "rows_considered": len(records),
        "e148_unclaimed_candidates": len(unclaimed) + len(extra),
        "e148_best_unclaimed_corrected_pct": round(best, 4),
        "hand_adjudicated_rows": len(HAND),
        "unclaimed_priced": [x for x in unclaimed],
        "unclaimed_out_of_band": extra,
        "blocker_census": blocker_census,
        "records": records,
        "editable_prefix_count": len(prefixes),
        "source_corpus_bytes": len(corpus),
        "ledger_bytes": len(ledger),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"rows considered (<= {threshold} pp): {len(records)}")
    print(f"unclaimed after all filters: {len(unclaimed)}")
    print(f"e148_best_unclaimed_corrected_pct = {best:.4f}")
    print("blocker census:", json.dumps(blocker_census, sort_keys=True))
    print()
    for x in records[:40]:
        mark = "CANDIDATE" if x["unclaimed"] else ",".join(x["blockers"])
        lf = x["landing"]["landed_fraction"]
        lfs = "n/a" if lf is None else f"{lf:.2f}"
        print(f'{x["row"]}  {x["corrected_total_pct"]:+.4f}  '
              f'land={lfs}({x["landing"]["identifiers"]})  {mark}')


if __name__ == "__main__":
    main()
