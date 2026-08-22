#!/usr/bin/env python3
"""E133 rungs 2-4: offline screen of the sketch-first draft readout (C1).

THE CHAIN THIS FILE MODELS. At 197e0550 the scored draft readout is:

  A  affine-2 g64 score of 12,292 derived centroids
  B  probe the best 25 %, then affine-2 g64 score of the 3,073 * 8 rows those
     leaves own
  C  top-32 of those rows by the SAME affine-2 score
  D  exact affine-4 g64 rerank of the 32, whose argmax is the proposal

C1 replaces the affine-2 matvec in A and B with a compact sketch, keeps the
best `N` rows, rescores those `N` exactly with affine-2, and then takes the
top-32 as before. D never changes, so the proposal is correct exactly when the
global affine-4 argmax survives into the top-32.

  miss(arm) = the exact affine-4 argmax over all 98,330 reachable rows is not
              in the arm's final 32.

The decision quantity is the PAIRED net miss rate against the shipped chain,
because the shipped chain already misses at some rate and that rate is already
priced into the baseline score.

WHY THIS IS NOT research/e87_screen.py. E87's baseline is the DENSE coarse
readout and its arm-C tables come from its own k-means. Neither describes what
ships now. Screening C1 against E87's baseline would price the sketch against
a readout the runtime stopped using when E121 landed, so the whole net-miss
column would be wrong. research/e133_index.py rebuilds the live derived index
and this file screens against it.

Sub-commands:
  corpus     inventory the captured hidden states, with per-seed acceptance
  validate   positive control: offline argmax vs the runtime's own proposal,
             shipped-chain absolute miss, and a deliberately damaged sketch
  screen     the full family x size x N x probe-fraction sweep

  python3 research/e133_screen.py corpus
  python3 research/e133_screen.py validate --limit 2048
  python3 research/e133_screen.py screen --out research/e133-screen.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e87_head as H  # noqa: E402
import e133_index as IX  # noqa: E402

CACHE = Path(os.path.expanduser("~/.cache/mlxfast/qwen3.8-27b-mtp-v1/e133"))
DUMP_DIR = CACHE / "screen/hidden"
VERIFY_DIR = CACHE / "screen/verify"
MANIFEST = Path(__file__).resolve().parent / "e124-corpus-manifest.json"
PCA_CACHE = CACHE / "pca-basis.npy"

LEAVES = H.PADDED_COUNT // IX.ROWS_PER_LEAF          # 12,292
SHIPPED_PROBE_FRACTION = IX.PROBE_FRACTION           # 0.25
SHORTLIST = 32                                       # draftRerankCandidateCount
Z = 1.959963984540054

# --- ranked byte model (assignment section D3, all values in the ranked frame)
AFFINE2_ROW_BYTES = 320 * 4 + 80 * 2 + 80 * 2        # 1,600
AFFINE4_ROW_BYTES = 640 * 4 + 80 * 2 + 80 * 2        # 2,880
STEP_BYTES = 323.59e6                # reconciled per-draft-step byte budget
BANDWIDTH = 462.2e9                  # ranked effective read rate
DRAFT_STEPS_PER_ROUND = 4.3818       # ranked mean draft steps per beagle round
BEAGLE_ROUND_US = 55_870.0           # ranked beagle round time
HEAD_SHARE_LO = 0.07
HEAD_SHARE_HI = 0.09
MU_BYTES = H.HIDDEN * 2              # bf16 mean vector
# Finding 69 exchange rate: one unit of net argmax miss rate costs 203 % of
# score. `e87_head.MISS_TO_SCORE_PCT` still holds the superseded 206.6 from
# E82 rung 0, so E133 states its own rate rather than editing E87's record.
MISS_TO_SCORE_PCT = 203.0

SKETCH_SEED = 133                    # fixed seed of R, stated here and in the brief


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------
# corpus


def seed_domains() -> dict[str, str]:
    return {s["id"]: s["domain"] for s in json.load(MANIFEST.open())["seeds"]}


def seed_shards() -> dict[str, list[tuple[Path, Path]]]:
    by_seed: dict[str, list[tuple[Path, Path]]] = {}
    for x_path in sorted(DUMP_DIR.glob("*.pid*.x.f32")):
        stem = x_path.name[: -len(".x.f32")]
        tok_path = DUMP_DIR / f"{stem}.tok.i32"
        if not tok_path.exists():
            continue
        by_seed.setdefault(stem.rsplit(".pid", 1)[0], []).append((x_path, tok_path))
    return by_seed


def chunks(batch: int, limit: int = 0, only_seeds: set[str] | None = None):
    """Yield `(seed, domain, x_bf16, proposal_tokens)`."""
    dom = seed_domains()
    total = 0
    for seed, shards in sorted(seed_shards().items()):
        if only_seeds and seed not in only_seeds:
            continue
        for x_path, tok_path in shards:
            x = np.memmap(x_path, dtype=np.float32, mode="r").reshape(-1, H.HIDDEN)
            tok = np.memmap(tok_path, dtype=np.int32, mode="r")
            n = min(x.shape[0], tok.shape[0])
            for start in range(0, n, batch):
                if limit and total >= limit:
                    return
                stop = min(start + batch, n)
                if limit:
                    stop = min(stop, start + (limit - total))
                xb = mx.array(np.ascontiguousarray(x[start:stop])).astype(mx.bfloat16)
                yield seed, dom.get(seed, "?"), xb, np.asarray(tok[start:stop])
                total += stop - start


def cmd_corpus(args) -> None:
    dom = seed_domains()
    rows = []
    for seed, shards in sorted(seed_shards().items()):
        samples = sum(
            np.memmap(t, dtype=np.int32, mode="r").shape[0] for _, t in shards)
        verify = VERIFY_DIR / f"{seed}.json"
        meta = json.loads(verify.read_text()) if verify.exists() else {}
        rows.append({
            "seed": seed,
            "domain": dom.get(seed, "?"),
            "samples": int(samples),
            "shards": len(shards),
            "accepted_draft_rate": meta.get("accepted_draft_rate"),
            "effective_mean_draft_len": meta.get("effective_mean_draft_len"),
            "round_count": meta.get("round_count"),
            "parity_all_ok": meta.get("parity_all_ok"),
            "all_tokens_matched": meta.get("all_tokens_matched"),
        })
    total = sum(r["samples"] for r in rows)
    by_domain: dict[str, int] = {}
    for r in rows:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + r["samples"]
    print(f"{'seed':20s}{'domain':10s}{'samples':>9s}{'accept':>9s}{'meanM':>8s}"
          f"{'rounds':>8s}  parity")
    for r in rows:
        acc = r["accepted_draft_rate"]
        mdl = r["effective_mean_draft_len"]
        print(f"{r['seed']:20s}{r['domain']:10s}{r['samples']:9d}"
              f"{acc if acc is None else round(acc, 4):>9}"
              f"{mdl if mdl is None else round(mdl, 3):>8}"
              f"{str(r['round_count']):>8}  {r['parity_all_ok']}")
    print(f"\ntotal samples {total}")
    print("by domain " + json.dumps(by_domain))
    out = {"samples": total, "by_domain": by_domain, "seeds": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


# --------------------------------------------------------------------------
# the live index, loaded once


class Index:
    def __init__(self, path: Path):
        blob = np.load(path)
        self.order = blob["order"].astype(np.int32)
        self.cluster_perm = blob["cluster_perm"].astype(np.int32)
        self.order_mx = mx.array(self.order)
        self.centroid = {
            "weight": mx.array(blob["centroid_weight"]),
            "scales": mx.array(blob["centroid_scales"]).view(mx.bfloat16),
            "biases": mx.array(blob["centroid_biases"]).view(mx.bfloat16),
            "group_size": 64,
            "bits": int(blob["centroid_bits"]),
        }
        # Canonical permuted position of every REAL compact row. Values at or
        # above REAL_COUNT are the six padding copies and are not canonical.
        position = np.full(H.PADDED_COUNT, -1, dtype=np.int32)
        position[self.order] = np.arange(H.PADDED_COUNT, dtype=np.int32)
        self.position_of_row = mx.array(position[: H.REAL_COUNT])
        assert int(np.min(position[: H.REAL_COUNT])) >= 0


# --------------------------------------------------------------------------
# sketch families


def pca_basis(rows: mx.array, mu: mx.array, rank: int) -> mx.array:
    """Top-`rank` eigenvectors of the row covariance, cached on disk.

    A PCA basis is a fixed function of the checkpoint, in the same class as the
    derived cluster index the runtime already builds at warm.
    """
    if PCA_CACHE.exists():
        basis = np.load(PCA_CACHE)
        if basis.shape[1] >= rank:
            return mx.array(basis[:, :rank])
    t0 = time.time()
    cov = mx.zeros((H.HIDDEN, H.HIDDEN), mx.float32)
    step = 8192
    for a in range(0, rows.shape[0], step):
        block = (rows[a: a + step].astype(mx.float32) - mu)
        cov = cov + mx.matmul(block.T, block)
        mx.eval(cov)
    values, vectors = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    basis = np.ascontiguousarray(vectors[:, ::-1][:, :256]).astype(np.float32)
    PCA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(PCA_CACHE, basis)
    kept = float(np.sum(values[::-1][:rank]) / np.sum(values))
    print(f"  pca basis in {time.time() - t0:.1f}s, top-{rank} keeps "
          f"{kept:.3f} of the variance", flush=True)
    return mx.array(basis[:, :rank])


class Sketch:
    """One (family, size) cell: row codes, centroid codes and the query rule."""

    def __init__(self, family: str, size: int, rows: mx.array, centroids: mx.array,
                 mu: mx.array, basis: mx.array | None):
        self.family = family
        self.size = size
        self.key = f"{family}{size}"
        if family == "simhash":
            rng = np.random.default_rng(SKETCH_SEED)
            proj = np.where(
                rng.standard_normal((H.HIDDEN, size)) >= 0.0, 1.0, -1.0).astype(np.float32)
            self.proj = mx.array(proj)
            self.row_codes, self.row_scale = self._simhash(rows, mu)
            self.cent_codes, self.cent_scale = self._simhash(centroids, mu)
            self.bytes_per_row = size // 8 + 4          # packed bits + norm
            self.proj_bytes = H.HIDDEN * size // 8      # 1-bit +-1 R
        elif family == "lowrank":
            assert basis is not None
            self.proj = basis[:, :size]
            self.row_codes, self.row_scale = self._lowrank(rows, mu)
            self.cent_codes, self.cent_scale = self._lowrank(centroids, mu)
            self.bytes_per_row = size + 4               # int8 codes + fp32 scale
            self.proj_bytes = H.HIDDEN * size * 2       # bf16 basis
        else:
            raise SystemExit(f"unknown sketch family {family}")
        mx.eval(self.row_codes, self.row_scale, self.cent_codes, self.cent_scale)

    def _simhash(self, table: mx.array, mu: mx.array):
        parts_c, parts_s = [], []
        step = 8192
        for a in range(0, table.shape[0], step):
            block = table[a: a + step].astype(mx.float32) - mu
            parts_c.append(mx.where(mx.matmul(block, self.proj) >= 0.0, 1.0, -1.0))
            parts_s.append(mx.linalg.norm(block, axis=1))
            mx.eval(parts_c[-1], parts_s[-1])
        return mx.concatenate(parts_c, axis=0), mx.concatenate(parts_s, axis=0)

    def _lowrank(self, table: mx.array, mu: mx.array):
        parts_c, parts_s = [], []
        step = 8192
        for a in range(0, table.shape[0], step):
            block = table[a: a + step].astype(mx.float32) - mu
            values = mx.matmul(block, self.proj)
            scale = mx.maximum(mx.max(mx.abs(values), axis=1), 1e-12) / 127.0
            codes = mx.round(values / scale[:, None])
            parts_c.append(mx.clip(codes, -127.0, 127.0))
            parts_s.append(scale)
            mx.eval(parts_c[-1], parts_s[-1])
        return mx.concatenate(parts_c, axis=0), mx.concatenate(parts_s, axis=0)

    def query(self, x: mx.array, mu: mx.array) -> mx.array:
        centred = x.astype(mx.float32) - mu
        if self.family == "simhash":
            return mx.where(mx.matmul(centred, self.proj) >= 0.0, 1.0, -1.0)
        return mx.matmul(centred, self.proj)

    def score_rows(self, q: mx.array) -> mx.array:
        dot = mx.matmul(q, self.row_codes.T)
        if self.family == "simhash":
            return self.row_scale[None, :] * mx.sin(0.5 * math.pi * dot / self.size)
        return self.row_scale[None, :] * dot

    def score_centroids(self, q: mx.array) -> mx.array:
        dot = mx.matmul(q, self.cent_codes.T)
        if self.family == "simhash":
            return self.cent_scale[None, :] * mx.sin(0.5 * math.pi * dot / self.size)
        return self.cent_scale[None, :] * dot


# --------------------------------------------------------------------------
# byte and score model, all in the ranked frame


def shipped_stage_bytes(probe_fraction: float = SHIPPED_PROBE_FRACTION) -> int:
    clusters = max(1, math.ceil(probe_fraction * LEAVES))
    return (LEAVES * AFFINE2_ROW_BYTES
            + clusters * IX.ROWS_PER_LEAF * AFFINE2_ROW_BYTES
            + SHORTLIST * AFFINE4_ROW_BYTES)


def arm_stage_bytes(bytes_per_row: int, proj_bytes: int, survivors: int,
                    probe_fraction: float, stage_a: str) -> int:
    """`stage_a` selects who orders the leaves: the sketch, or today's affine-2
    centroid readout. The hybrid keeps stage A exact and sketches only the
    24,584-row stage, which is 39.33 MB of the 59.09 MB stage."""
    clusters = max(1, math.ceil(probe_fraction * LEAVES))
    centroid = bytes_per_row if stage_a == "sketch" else AFFINE2_ROW_BYTES
    return (LEAVES * centroid
            + clusters * IX.ROWS_PER_LEAF * bytes_per_row
            + survivors * AFFINE2_ROW_BYTES
            + proj_bytes + MU_BYTES
            + SHORTLIST * AFFINE4_ROW_BYTES)


def price(removed_bytes: int) -> dict:
    """The assignment's three ranked prices, recomputed from actual bytes."""
    seconds = removed_bytes / BANDWIDTH
    byte_rate_pct = 100.0 * (seconds * 1e6 * DRAFT_STEPS_PER_ROUND) / BEAGLE_ROUND_US
    fraction = removed_bytes / STEP_BYTES
    return {
        "removed_bytes": removed_bytes,
        "removed_step_fraction": fraction,
        "pct_byte_rate": byte_rate_pct,
        "pct_head_share_7": 100.0 * HEAD_SHARE_LO * fraction,
        "pct_head_share_9": 100.0 * HEAD_SHARE_HI * fraction,
    }


# --------------------------------------------------------------------------
# counters


class Cell:
    __slots__ = ("miss", "worse", "better", "n", "recall_hit", "probe_hit",
                 "survivor_hit", "by_domain")

    def __init__(self) -> None:
        self.miss = self.worse = self.better = self.n = 0
        self.recall_hit = self.probe_hit = self.survivor_hit = 0
        self.by_domain: dict[str, list[int]] = {}

    def add(self, domain: str, miss: np.ndarray, base: np.ndarray,
            recall: np.ndarray, probe: np.ndarray, survivor: np.ndarray) -> None:
        n = int(miss.size)
        hit = int(miss.sum())
        worse = int((miss & ~base).sum())
        better = int((base & ~miss).sum())
        self.miss += hit
        self.worse += worse
        self.better += better
        self.n += n
        self.recall_hit += int(recall.sum())
        self.probe_hit += int(probe.sum())
        self.survivor_hit += int(survivor.sum())
        slot = self.by_domain.setdefault(domain, [0, 0, 0, 0, 0])
        slot[0] += hit
        slot[1] += worse
        slot[2] += better
        slot[3] += n
        slot[4] += int(recall.sum())


# --------------------------------------------------------------------------
# the shared per-batch front end


class Screen:
    def __init__(self, index: Index):
        self.index = index
        self.exact = H.load_exact()
        self.coarse = H.load_coarse()
        rows = H.dequantized(self.exact)
        mx.eval(rows)
        self.mu = mx.mean(rows[: H.REAL_COUNT].astype(mx.float32), axis=0)
        mx.eval(self.mu)
        self.centroids = (
            mx.take(rows, index.order_mx, axis=0)
            .reshape(LEAVES, IX.ROWS_PER_LEAF, H.HIDDEN)
            .astype(mx.float32)
            .mean(axis=1)
        )
        mx.eval(self.centroids)
        self.rows = rows

    def front(self, x: mx.array):
        """Everything both the shipped chain and every arm need."""
        b = x.shape[0]
        exact_scores = H.scores(self.exact, x)
        argmax_row = mx.argmax(exact_scores, axis=1)
        coarse_all = H.scores_all(self.coarse, x)
        coarse_perm = mx.take(coarse_all, self.index.order_mx, axis=1)
        at = mx.take_along_axis(
            coarse_all, argmax_row[:, None].astype(mx.int32), axis=1)
        gt_perm = coarse_perm > at
        pos_r = mx.take(self.index.position_of_row, argmax_row, axis=0).astype(mx.int32)
        leaf_r = pos_r // IX.ROWS_PER_LEAF
        gt_leaf = mx.sum(gt_perm.reshape(b, LEAVES, IX.ROWS_PER_LEAF).astype(mx.int32),
                         axis=2)
        # Stage A as it ships: affine-2 g64 over the 12,292 derived centroids.
        centroid_scores = H.scores_all(self.index.centroid, x)
        order_a2 = mx.argsort(-centroid_scores, axis=1)
        crank_a2 = mx.take_along_axis(
            mx.argsort(order_a2, axis=1), leaf_r[:, None], axis=1)[:, 0]
        mx.eval(argmax_row, coarse_perm, gt_perm, pos_r, leaf_r, gt_leaf,
                order_a2, crank_a2)
        return {
            "b": b,
            "argmax_row": argmax_row,
            "coarse_perm": coarse_perm,
            "gt_perm": gt_perm,
            "gt_leaf": gt_leaf,
            "pos_r": pos_r,
            "leaf_r": leaf_r,
            "order_a2": order_a2,
            "crank_a2": crank_a2,
        }

    def shipped_miss(self, f) -> np.ndarray:
        """The live chain: affine-2 centroids, 25 % probe, affine-2 top-32."""
        clusters = max(1, math.ceil(SHIPPED_PROBE_FRACTION * LEAVES))
        cum_gt = mx.cumsum(
            mx.take_along_axis(f["gt_leaf"], f["order_a2"], axis=1), axis=1)
        miss = mx.logical_or(mx.logical_not(f["crank_a2"] < clusters),
                             cum_gt[:, clusters - 1] >= SHORTLIST)
        mx.eval(miss)
        return np.asarray(miss).astype(bool)


def run_arm(screen: Screen, sketch: Sketch, f, x: mx.array, probe_fractions,
            survivor_widths, base_miss: np.ndarray, domain: str,
            cells: dict[tuple, Cell], stage_a: str = "sketch") -> None:
    """One sketch cell family. `stage_a` chooses who orders the leaves.

    `sketch` sketches both stages, which is C1 as designed. `affine2` keeps
    today's exact centroid readout and sketches only the 24,584-row stage, so
    a stage-A failure and a stage-B failure can be told apart and priced
    apart.
    """
    b = f["b"]
    q = sketch.query(x, screen.mu)
    if stage_a == "sketch":
        cent = sketch.score_centroids(q)
        order_c = mx.argsort(-cent, axis=1)
        crank_r = mx.take_along_axis(
            mx.argsort(order_c, axis=1), f["leaf_r"][:, None], axis=1)[:, 0]
    else:
        order_c = f["order_a2"]
        crank_r = f["crank_a2"]
    mx.eval(order_c, crank_r)

    row_scores = sketch.score_rows(q)
    max_clusters = max(max(1, math.ceil(p * LEAVES)) for p in probe_fractions)
    positions = (order_c[:, :max_clusters, None] * IX.ROWS_PER_LEAF
                 + mx.arange(IX.ROWS_PER_LEAF, dtype=mx.int32)[None, None, :])
    positions = positions.reshape(b, max_clusters * IX.ROWS_PER_LEAF).astype(mx.int32)
    row_perm = mx.take(row_scores, screen.index.order_mx, axis=1)
    sk_probe = mx.take_along_axis(row_perm, positions, axis=1)
    co_probe = mx.take_along_axis(f["coarse_perm"], positions, axis=1)
    gt_probe = mx.take_along_axis(f["gt_perm"], positions, axis=1)
    is_r = positions == f["pos_r"][:, None]
    mx.eval(sk_probe, co_probe, gt_probe, is_r)

    max_width = max(survivor_widths)
    for p in probe_fractions:
        clusters = max(1, math.ceil(p * LEAVES))
        width = clusters * IX.ROWS_PER_LEAF
        sk = sk_probe[:, :width]
        top = mx.argpartition(-sk, kth=max_width - 1, axis=1)[:, :max_width]
        top = mx.take_along_axis(
            top, mx.argsort(-mx.take_along_axis(sk, top, axis=1), axis=1), axis=1)
        exact_top1 = mx.argmax(co_probe[:, :width], axis=1).astype(mx.int32)
        probe_hit = np.asarray(crank_r < clusters).astype(bool)
        mx.eval(top, exact_top1)
        for n_keep in survivor_widths:
            sel = top[:, :n_keep]
            beaten = mx.sum(
                mx.take_along_axis(gt_probe[:, :width], sel, axis=1).astype(mx.int32),
                axis=1)
            survivor = mx.any(
                mx.take_along_axis(is_r[:, :width], sel, axis=1), axis=1)
            recall = mx.any(sel == exact_top1[:, None], axis=1)
            miss = mx.logical_or(mx.logical_not(survivor), beaten >= SHORTLIST)
            mx.eval(miss, survivor, recall)
            key = (sketch.key, stage_a, n_keep, p)
            cells.setdefault(key, Cell()).add(
                domain,
                np.asarray(miss).astype(bool), base_miss,
                np.asarray(recall).astype(bool), probe_hit,
                np.asarray(survivor).astype(bool))
        del sk, top, exact_top1
    del row_scores, row_perm, sk_probe, co_probe, gt_probe, is_r, positions


# --------------------------------------------------------------------------
# commands


def build_sketches(screen: Screen, families: list[tuple[str, int]]) -> list[Sketch]:
    need_basis = max([s for f, s in families if f == "lowrank"], default=0)
    basis = pca_basis(screen.rows, screen.mu, need_basis) if need_basis else None
    out = []
    for family, size in families:
        t0 = time.time()
        out.append(Sketch(family, size, screen.rows, screen.centroids, screen.mu, basis))
        print(f"  sketch {family}{size}: {out[-1].bytes_per_row} B/row, "
              f"R {out[-1].proj_bytes} B, built in {time.time() - t0:.1f}s", flush=True)
    return out


def parse_families(spec: str) -> list[tuple[str, int]]:
    families = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        family = "simhash" if token.startswith("simhash") else "lowrank"
        families.append((family, int(token[len(family):])))
    return families


def cmd_validate(args) -> None:
    index = Index(Path(args.index))
    screen = Screen(index)
    # A control that cannot fail is not a control: an 8-bit SimHash must lose
    # the argmax constantly, or the miss column below proves nothing.
    damaged = Sketch("simhash", 8, screen.rows, screen.centroids, screen.mu, None)

    n = tok_ok = shipped_miss = damaged_miss = 0
    ranks: list[int] = []
    for _, _, x, proposal in chunks(args.batch, args.limit):
        f = screen.front(x)
        vocab = np.asarray(H.compact_to_vocab(f["argmax_row"]))
        tok_ok += int(np.sum(vocab == proposal))
        bad = np.nonzero(vocab != proposal)[0]
        if bad.size:
            exact_scores = H.scores(screen.exact, x)
            sub = mx.take(exact_scores, mx.array(bad.astype(np.int32)), axis=0)
            got = mx.array(H.vocab_to_compact(proposal[bad]).astype(np.int32))
            mine = mx.take_along_axis(sub, got[:, None], axis=1)[:, 0]
            ranks += [int(v) for v in np.asarray(mx.sum(sub > mine[:, None], axis=1))]
        base = screen.shipped_miss(f)
        shipped_miss += int(base.sum())
        cells: dict[tuple, Cell] = {}
        run_arm(screen, damaged, f, x, [SHIPPED_PROBE_FRACTION], [256], base, "all", cells)
        damaged_miss += cells[
            (damaged.key, "sketch", 256, SHIPPED_PROBE_FRACTION)].miss
        n += f["b"]

    p, lo, hi = wilson(shipped_miss, n)
    report = {
        "samples": n,
        "proposal_match": tok_ok / n if n else float("nan"),
        "proposal_mismatch": {"count": len(ranks), "rank_max": max(ranks, default=0)},
        "m_shipped_live_chain": {"misses": shipped_miss, "p": p, "lo": lo, "hi": hi},
        "m_damaged_simhash8_control": {
            "misses": damaged_miss, "p": damaged_miss / n if n else float("nan")},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


def cmd_screen(args) -> None:
    index = Index(Path(args.index))
    screen = Screen(index)
    families = parse_families(args.families)
    widths = [int(v) for v in args.widths.split(",")]
    fractions = [float(v) for v in args.probes.split(",")]
    stages = [v for v in args.stage_a.split(",") if v]
    sketches = build_sketches(screen, families)

    cells: dict[tuple, Cell] = {}
    base_cell = Cell()
    n = 0
    t0 = time.time()
    for seed, domain, x, _ in chunks(args.batch, args.limit):
        f = screen.front(x)
        base = screen.shipped_miss(f)
        zeros = np.zeros_like(base)
        base_cell.add(domain, base, base, ~zeros, ~zeros, ~zeros)
        for sketch in sketches:
            for stage_a in stages:
                run_arm(screen, sketch, f, x, fractions, widths, base, domain,
                        cells, stage_a)
        n += f["b"]
        if n % (args.batch * 10) == 0:
            print(f"  {n} samples  {time.time() - t0:.0f}s", flush=True)

    out = {
        "samples": n,
        "base_sha": args.base_sha,
        "shipped": summarize("shipped", base_cell, price(0), {}),
        "cells": [],
    }
    shipped_bytes = shipped_stage_bytes()
    for (key, stage_a, n_keep, p), cell in cells.items():
        sketch = next(s for s in sketches if s.key == key)
        arm_bytes = arm_stage_bytes(sketch.bytes_per_row, sketch.proj_bytes,
                                    n_keep, p, stage_a)
        model = price(shipped_bytes - arm_bytes)
        tag = "" if stage_a == "sketch" else "-hybridA"
        out["cells"].append(summarize(
            f"{key}{tag}-N{n_keep}-p{p:g}", cell, model,
            {"family": sketch.family, "size": sketch.size, "stage_a": stage_a,
             "bytes_per_row": sketch.bytes_per_row, "proj_bytes": sketch.proj_bytes,
             "survivors": n_keep, "probe_fraction": p,
             "arm_stage_bytes": arm_bytes, "shipped_stage_bytes": shipped_bytes}))
    out["cells"].sort(key=lambda c: -c["predicted_pct_worst_domain"])

    print(f"\n{'arm':28s}{'B/row':>7s}{'m':>10s}{'net':>11s}{'netBeagle':>11s}"
          f"{'recall':>9s}{'gain%':>8s}{'pred%':>8s}{'predWD%':>9s}")
    for cell in out["cells"][: args.top]:
        print(f"{cell['arm']:28s}{cell['bytes_per_row']:7d}{cell['m']:10.5f}"
              f"{cell['net_miss']:11.3e}{cell['net_miss_low_acceptance']:11.3e}"
              f"{cell['recall_affine2_top1']:9.5f}{cell['pct_head_share_7']:8.3f}"
              f"{cell['predicted_pct']:8.3f}{cell['predicted_pct_worst_domain']:9.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


LOW_ACCEPTANCE_DOMAIN = "beagle"


def summarize(arm: str, cell: Cell, model: dict, extra: dict) -> dict:
    m, m_lo, m_hi = wilson(cell.miss, cell.n)
    net = (cell.worse - cell.better) / cell.n if cell.n else float("nan")
    disc = cell.worse + cell.better
    _, dlo, dhi = wilson(cell.worse, disc)
    net_lo = (2 * dlo - 1) * disc / cell.n if disc and cell.n else 0.0
    net_hi = (2 * dhi - 1) * disc / cell.n if disc and cell.n else 0.0
    by_domain = {}
    for domain, (hit, worse, better, total, recall) in cell.by_domain.items():
        # A per-domain stratum is small, so the discordant-pair interval is
        # reported with the point estimate. One extra miss in a 600-sample
        # stratum already reads as 1.7e-3 against a 3.0e-3 gate.
        d = worse + better
        _, lo, hi = wilson(worse, d)
        by_domain[domain] = {
            "n": total,
            "m": hit / total,
            "net": (worse - better) / total,
            "net_lo": (2 * lo - 1) * d / total if d else 0.0,
            "net_hi": (2 * hi - 1) * d / total if d else 0.0,
            "discordant": d,
            "recall": recall / total,
        }
    worst = max((v["net"] for v in by_domain.values()), default=net)
    low = by_domain.get(LOW_ACCEPTANCE_DOMAIN, {}).get("net", float("nan"))
    penalty = MISS_TO_SCORE_PCT
    return {
        "arm": arm,
        **extra,
        "n": cell.n,
        "misses": cell.miss,
        "m": m,
        "m_lo": m_lo,
        "m_hi": m_hi,
        "worse_than_shipped": cell.worse,
        "better_than_shipped": cell.better,
        "net_miss": net,
        "net_miss_lo": net_lo,
        "net_miss_hi": net_hi,
        "net_miss_worst_domain": worst,
        "net_miss_low_acceptance": low,
        "recall_affine2_top1": cell.recall_hit / cell.n if cell.n else float("nan"),
        "probe_hit_rate": cell.probe_hit / cell.n if cell.n else float("nan"),
        "survivor_hit_rate": cell.survivor_hit / cell.n if cell.n else float("nan"),
        **model,
        "predicted_pct": model["pct_head_share_7"] - penalty * net,
        "predicted_pct_worst_domain": model["pct_head_share_7"] - penalty * worst,
        "predicted_pct_low_acceptance":
            model["pct_head_share_7"] - penalty * (low if low == low else 0.0),
        "by_domain": by_domain,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus")
    c.add_argument("--out", default="research/e133-corpus.json")
    c.set_defaults(func=cmd_corpus)

    v = sub.add_parser("validate")
    v.add_argument("--limit", type=int, default=0)
    v.add_argument("--batch", type=int, default=64)
    v.add_argument("--index", default=str(IX.DEFAULT_OUT))
    v.add_argument("--out", default="research/e133-validate.json")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("screen")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--index", default=str(IX.DEFAULT_OUT))
    s.add_argument("--families",
                   default="simhash256,simhash512,simhash1024,simhash2048,"
                           "lowrank32,lowrank64,lowrank128,lowrank256")
    s.add_argument("--widths", default="64,128,256,512,1024")
    s.add_argument("--probes", default="0.25,0.35,0.50")
    s.add_argument("--stage-a", default="sketch,affine2",
                   help="sketch = C1 as assigned; affine2 = keep the exact "
                        "centroid readout and sketch only the row stage")
    s.add_argument("--top", type=int, default=25)
    s.add_argument("--base-sha", default="197e0550ab46842b639a4ff4fe3f4889ca3b01ec")
    s.add_argument("--out", default="research/e133-screen.json")
    s.set_defaults(func=cmd_screen)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
