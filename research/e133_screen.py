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

WHY THIS IS NOT research/e87_screen.py. E87's baseline is the DENSE coarse
readout and its arm-C tables come from its own k-means. Neither describes what
ships now. Screening C1 against E87's baseline would price the sketch against a
readout the runtime stopped using when E121 landed. research/e133_index.py
rebuilds the live derived index and this file screens against it.

WHAT F1 CHANGED, AND WHY EACH CHANGE IS HERE.

  F1.3  Finding 83 gives three strata, not ten domains. `beagle` carries the
        first 0.5 term of the median outright; medicine, essays, republic and
        botany share the second; plutarch, drama and travel carry exactly
        zero. The kill uses the WORSE OF THE TWO GATING STRATA. A maximum over
        ten noisy near-zero domain estimates is biased upward hard enough to
        close C1 on noise alone.
  F1.4c Record the RANK of the exact argmax in the sketch ordering, not a
        boolean at one `N`. The net miss at width `N` IS `P(rank >= N)`, so one
        rank per sample gives the whole survival curve, and the curve's shape
        constrains the `N = 256` tail far better than a dozen raw events. Every
        count carries an exact Clopper-Pearson interval, and the tail-fit
        estimate is reported beside the raw count, never instead of it.
  F1.5  Two miss rates. `m_absolute` is against the exact affine-4 argmax and
        carries the kill. `m_incremental` is against the SHIPPED chain output
        and carries the price, because the shipped chain is already
        approximate and its own misses are already in the baseline score.
  F1.5  A miss costs `m * (p - q)`, not `m`. `p` is the shipped chain's
        per-step acceptance and `q` is the chance the substituted row is
        itself the target's argmax. Both come from the trusted parent's
        `row_ledger`, which records, per draft row, the token the runtime
        proposed and the token the fixed target produced there. The dumped
        rows are aligned to that ledger and the alignment is CHECKED token by
        token, because the instrument fires once per draft-head call, not once
        per emitted token.

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
REFERENCE_DIR = CACHE / "screen/reference"
MANIFEST = Path(__file__).resolve().parent / "e133-corpus-manifest.json"
PCA_CACHE = CACHE / "pca-basis.npy"
QCOV_CACHE = CACHE / "query-second-moment.npz"
QBASIS_RANK = 512

LEAVES = H.PADDED_COUNT // IX.ROWS_PER_LEAF          # 12,292
SHIPPED_PROBE_FRACTION = IX.PROBE_FRACTION           # 0.25
SHORTLIST = 32                                       # draftRerankCandidateCount
Z = 1.959963984540054

# F1.3. `zero_weight` is reported and never gates.
GATING_STRATA = ("beagle", "min_carriers")
ALL_STRATA = ("beagle", "min_carriers", "zero_weight")
# F2.3. `essays_bacon` is the hardest carrier in the corpus: acceptance 0.3989
# at mean draft 1.996. It gates inside `min_carriers` and is ALSO reported on
# its own line. A watch line duplicates rows that a real stratum already
# counts, so it never enters a total and never gates.
WATCH_STRATA = ("essays_bacon",)

# F1.4c. The survival curve is read at these widths from one stored rank.
RANK_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
             16384)

T0_NET_MISS = 3.0e-3
T0B_RECALL = 0.997

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
MISS_TO_SCORE_PCT = H.MISS_TO_SCORE_PCT      # Finding 69, 203.0 after F1.7
assert MISS_TO_SCORE_PCT == 203.0, MISS_TO_SCORE_PCT

SKETCH_SEED = 133                    # fixed seed of R, stated here and in the brief


# --------------------------------------------------------------------------
# interval estimates


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1.0 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta by the Lentz continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log1p(-x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1.0 - c * d) < 1e-15:
            break
    return front * (f - 1.0)


def _beta_quantile(a: float, b: float, target: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if _betainc(a, b, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Exact binomial interval. F1.4c needs this for the rare-event counts;
    Wilson is not trustworthy at a handful of events."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    lo = 0.0 if k == 0 else _beta_quantile(k, n - k + 1, alpha / 2)
    hi = 1.0 if k == n else _beta_quantile(k + 1, n - k, 1 - alpha / 2)
    return k / n, lo, hi


def tail_fit(counts: dict[int, int], n: int, target: int,
             bootstrap: int = 200) -> dict:
    """Power-law fit of the survival curve `P(rank >= N)`, read at `target`.

    F1.4c: at small `N` the event is common and the estimate is tight, and the
    slope constrains the tail far better than a dozen raw events. The fit is
    reported BESIDE the raw count, never instead of it.
    """
    grid = [g for g in sorted(counts) if 0 < g < target and counts[g] >= 20]
    if len(grid) < 3 or n == 0:
        return {"usable": False, "reason": "fewer than three populated grid points"}
    x = np.log(np.asarray(grid, dtype=np.float64))
    obs = np.asarray([counts[g] for g in grid], dtype=np.float64)
    rng = np.random.default_rng(SKETCH_SEED)

    def fit(k: np.ndarray) -> float:
        y = np.log(np.maximum(k, 0.5) / n)
        slope, intercept = np.polyfit(x, y, 1)
        return float(np.exp(intercept + slope * math.log(target)))

    draws = np.array([fit(rng.binomial(n, obs / n).astype(np.float64))
                      for _ in range(bootstrap)])
    return {
        "usable": True,
        "grid": grid,
        "p": fit(obs),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
    }


# --------------------------------------------------------------------------
# the captured corpus


def seed_meta() -> dict[str, dict]:
    blob = json.load(MANIFEST.open())
    return {s["id"]: {"domain": s["domain"], "stratum": s["stratum"]}
            for s in blob["seeds"]}


def seed_shards() -> dict[str, list[tuple[Path, Path]]]:
    by_seed: dict[str, list[tuple[Path, Path]]] = {}
    for x_path in sorted(DUMP_DIR.glob("*.pid*.x.f32")):
        stem = x_path.name[: -len(".x.f32")]
        tok_path = DUMP_DIR / f"{stem}.tok.i32"
        if not tok_path.exists():
            continue
        by_seed.setdefault(stem.rsplit(".pid", 1)[0], []).append((x_path, tok_path))
    return by_seed


def draft_ledger(seed: str) -> dict[str, np.ndarray] | None:
    """The trusted parent's own record of every draft row it evaluated.

    `row_ledger` carries, per draft row, the token the shipped runtime proposed
    (`token`), the token the fixed target actually produced there
    (`reference_token`), and the parent's accept verdict. That makes `p` and
    `q` from F1.5 measured quantities rather than positional guesses.
    """
    path = VERIFY_DIR / f"{seed}.json"
    if not path.exists():
        return None
    rows = [r for r in json.loads(path.read_text())["row_ledger"]
            if r["kind"] == "draft"]
    if not rows:
        return None
    accepted = np.asarray([bool(r["accepted"]) for r in rows], dtype=bool)
    rnd = np.asarray([r["round"] for r in rows], dtype=np.int64)
    # Speculative acceptance is prefix-monotone: a round keeps its longest
    # correct prefix and discards every later row. So a row is LIVE only when
    # every earlier row of its round was accepted, and `accepted` factors
    # exactly as `prefix_live AND (token == reference_token)`. A substitution
    # on a dead row cannot change acceptance, because the round already threw
    # that row away. `cmd_validate` asserts the factorisation.
    first = np.concatenate([[True], rnd[1:] != rnd[:-1]])
    prefix_live = np.where(first, True,
                           np.concatenate([[True], accepted[:-1]]))
    return {
        "token": np.asarray([r["token"] for r in rows], dtype=np.int64),
        "reference": np.asarray([r["reference_token"] for r in rows], dtype=np.int64),
        "accepted": accepted,
        "prefix_live": prefix_live,
        "round": rnd,
        "draft_index": np.asarray([r["draft_index"] for r in rows], dtype=np.int64),
    }


def align_shard(seed: str, shards, ledger) -> tuple[Path, Path, int]:
    """Pick the dump shard that holds the real draft rows and find its offset.

    The worker warms legal shapes before the leg, so the head runs a few times
    on a dummy row first, and the CLI-side process contributes a warmup-only
    shard. Alignment is then CHECKED, not assumed: the dumped proposal ids must
    equal the ledger's draft tokens one for one. A mismatch raises instead of
    quietly shifting every `q`.
    """
    want = ledger["token"]
    for x_path, tok_path in sorted(shards, key=lambda s: -s[1].stat().st_size):
        tok = np.fromfile(tok_path, dtype=np.int32).astype(np.int64)
        if tok.size < want.size:
            continue
        skip = tok.size - want.size
        if not np.array_equal(tok[skip:], want):
            bad = int(np.argmax(tok[skip:] != want))
            raise SystemExit(
                f"{seed}: dump shard {tok_path.name} does not match the row "
                f"ledger at draft row {bad} "
                f"(dump {tok[skip + bad]} vs ledger {want[bad]}); "
                f"skip={skip}, dump={tok.size}, ledger={want.size}")
        return x_path, tok_path, skip
    raise SystemExit(f"{seed}: no dump shard holds {want.size} draft rows")


def chunks(batch: int, limit: int = 0, only_seeds: set[str] | None = None):
    """Yield `(seed, stratum, x_bf16, proposal, reference, accepted, live)`.

    Every yielded sample is one verified draft row: the hidden state that
    reached the readout, the token the shipped chain returned for it, the
    target's own token at that row, the parent's accept verdict, and whether
    the row's round still had an unbroken accepted prefix when it ran.
    """
    meta = seed_meta()
    total = 0
    for seed, shards in sorted(seed_shards().items()):
        if only_seeds and seed not in only_seeds:
            continue
        ledger = draft_ledger(seed)
        if ledger is None:
            print(f"  skip {seed}: no verify ledger", flush=True)
            continue
        x_path, _, skip = align_shard(seed, shards, ledger)
        stratum = meta.get(seed, {}).get("stratum", "unknown")
        x = np.memmap(x_path, dtype=np.float32, mode="r").reshape(-1, H.HIDDEN)
        n = ledger["token"].size
        for start in range(0, n, batch):
            if limit and total >= limit:
                return
            stop = min(start + batch, n)
            if limit:
                stop = min(stop, start + (limit - total))
            xb = mx.array(
                np.ascontiguousarray(x[skip + start: skip + stop])
            ).astype(mx.bfloat16)
            yield (seed, stratum, xb,
                   ledger["token"][start:stop],
                   ledger["reference"][start:stop],
                   ledger["accepted"][start:stop],
                   ledger["prefix_live"][start:stop])
            total += stop - start


def cmd_selftest(args) -> None:
    """Check the two estimators that carry the kill decision against closed
    forms and against a survival curve whose answer is known in advance.

    The screen reports rare-event rates near 1e-3 on a few thousand samples, so
    a silently wrong interval would move the T0 verdict without moving any
    measured number.
    """
    ok = True

    # Clopper-Pearson at k=0 and k=n has the closed form 1 - (alpha/2)^(1/n).
    for n in (1000, 4088, 4599):
        _, _, hi = clopper_pearson(0, n)
        want = 1.0 - 0.025 ** (1.0 / n)
        good = abs(hi - want) < 1e-12
        ok &= good
        print(f"CP k=0 n={n:5d} hi={hi:.10e} closed_form={want:.10e} "
              f"{'ok' if good else 'MISMATCH'}")
        _, lo, _ = clopper_pearson(n, n)
        want_lo = 0.025 ** (1.0 / n)
        good = abs(lo - want_lo) < 1e-12
        ok &= good
        print(f"CP k=n n={n:5d} lo={lo:.10e} closed_form={want_lo:.10e} "
              f"{'ok' if good else 'MISMATCH'}")

    # A textbook interior case: k=3, n=4088.
    p, lo, hi = clopper_pearson(3, 4088)
    print(f"CP k=3 n=4088 -> {p:.4e} [{lo:.4e}, {hi:.4e}]")
    good = lo < p < hi and hi < T0_NET_MISS
    ok &= good
    print(f"  three events in one gating stratum stay under T0: "
          f"{'ok' if good else 'MISMATCH'}")

    # The tail fit must recover a planted power law, and its interval must
    # cover the planted value.
    truth_a, truth_b, n = 0.30, 1.5, 8000
    counts = {g: int(round(n * truth_a * g ** -truth_b)) for g in RANK_GRID}
    fit = tail_fit(counts, n, 256)
    truth = truth_a * 256 ** -truth_b
    good = fit["usable"] and fit["lo"] <= truth <= fit["hi"]
    ok &= good
    print(f"tail_fit planted={truth:.4e} est={fit['p']:.4e} "
          f"[{fit['lo']:.4e}, {fit['hi']:.4e}] {'ok' if good else 'MISMATCH'}")

    # A curve with no tail must not be fitted into one.
    flat = tail_fit({g: (n if g <= 1 else 0) for g in RANK_GRID}, n, 256)
    good = not flat["usable"]
    ok &= good
    print(f"tail_fit refuses a degenerate curve: {'ok' if good else 'MISMATCH'}"
          f" ({flat.get('reason', '')})")

    # The score constant must be the corrected one (F1 section 7).
    good = MISS_TO_SCORE_PCT == 203.0
    ok &= good
    print(f"MISS_TO_SCORE_PCT={MISS_TO_SCORE_PCT} "
          f"{'ok' if good else 'MISMATCH: expected 203.0'}")

    print("SELFTEST", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


def cmd_corpus(args) -> None:
    meta = seed_meta()
    rows = []
    for seed, shards in sorted(seed_shards().items()):
        dumped = sum(
            np.memmap(t, dtype=np.int32, mode="r").shape[0] for _, t in shards)
        verify = VERIFY_DIR / f"{seed}.json"
        blob = json.loads(verify.read_text()) if verify.exists() else {}
        ledger = draft_ledger(seed)
        samples = 0 if ledger is None else int(ledger["token"].size)
        skip = None
        if ledger is not None:
            _, _, skip = align_shard(seed, shards, ledger)
        rows.append({
            "seed": seed,
            "domain": meta.get(seed, {}).get("domain", "?"),
            "stratum": meta.get(seed, {}).get("stratum", "?"),
            # Usable samples are the ledger's draft rows, not the raw dump: the
            # worker's shape warmup contributes rows that no ledger row owns.
            "samples": samples,
            "dumped_rows": int(dumped),
            "warmup_rows_skipped": skip,
            "shards": len(shards),
            "accepted_draft_rate": blob.get("accepted_draft_rate"),
            "accepted_draft_total": blob.get("accepted_draft_total"),
            "effective_mean_draft_len": blob.get("effective_mean_draft_len"),
            "round_count": blob.get("round_count"),
            "parity_all_ok": blob.get("parity_all_ok"),
            "all_tokens_matched": blob.get("all_tokens_matched"),
            "head_sha256": (blob.get("head_provenance") or {}).get("sha256"),
        })
    by_stratum: dict[str, int] = {}
    for r in rows:
        by_stratum[r["stratum"]] = by_stratum.get(r["stratum"], 0) + r["samples"]
    total = sum(r["samples"] for r in rows)

    print(f"{'seed':22s}{'stratum':14s}{'samples':>8s}{'accept':>9s}{'meanM':>8s}"
          f"{'rounds':>8s}  parity")
    for r in rows:
        acc = r["accepted_draft_rate"]
        mdl = r["effective_mean_draft_len"]
        print(f"{r['seed']:22s}{r['stratum']:14s}{r['samples']:8d}"
              f"{'' if acc is None else round(acc, 4):>9}"
              f"{'' if mdl is None else round(mdl, 3):>8}"
              f"{str(r['round_count']):>8}  {r['parity_all_ok']}")
    print(f"\ntotal samples {total}")
    for stratum in ALL_STRATA:
        n = by_stratum.get(stratum, 0)
        note = "gating" if stratum in GATING_STRATA else "reported only"
        ok = "" if stratum not in GATING_STRATA else (
            "  meets 4,000" if n >= 4000 else "  BELOW 4,000")
        print(f"  {stratum:14s}{n:7d}  ({note}){ok}")

    out = {"samples": total, "by_stratum": by_stratum, "seeds": rows,
           "parity_failures": [r["seed"] for r in rows
                               if r["parity_all_ok"] is not True]}
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
        position = np.full(H.PADDED_COUNT, -1, dtype=np.int32)
        position[self.order] = np.arange(H.PADDED_COUNT, dtype=np.int32)
        self.position_of_row = mx.array(position[: H.REAL_COUNT])
        assert int(np.min(position[: H.REAL_COUNT])) >= 0
        # Permuted position -> compact id, matching `clusterPerm` at :5535.
        self.perm_mx = mx.array(self.cluster_perm)


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


def query_second_moments(mu: mx.array, batch: int = 64) -> dict[str, np.ndarray]:
    """`E[(x - mu)(x - mu)^T]` per stratum, cached on disk.

    Row PCA is the wrong basis for this problem. The sketch error on a rank-`k`
    projection `P` is `row . (I - P)(x - mu)`, so the residual is governed by
    the QUERY distribution, not by the row covariance. A basis that spans the
    hidden states makes every row exact at once.
    """
    if QCOV_CACHE.exists():
        cached = np.load(QCOV_CACHE)
        return {k: cached[k] for k in cached.files}
    t0 = time.time()
    acc: dict[str, mx.array] = {}
    counts: dict[str, int] = {}
    for _, stratum, x, *_ in chunks(batch):
        d = x.astype(mx.float32) - mu
        block = mx.matmul(d.T, d)
        acc[stratum] = acc.get(stratum, mx.zeros((H.HIDDEN, H.HIDDEN),
                                                 mx.float32)) + block
        counts[stratum] = counts.get(stratum, 0) + d.shape[0]
        mx.eval(acc[stratum])
    out = {k: np.asarray(v, dtype=np.float64) / counts[k] for k, v in acc.items()}
    QCOV_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(QCOV_CACHE, **out)
    print(f"  query second moments in {time.time() - t0:.1f}s: "
          + ", ".join(f"{k} n={v}" for k, v in sorted(counts.items())), flush=True)
    return out


def query_bases(mu: mx.array, rank: int) -> tuple[dict[str, mx.array], dict]:
    """One held-out query basis per gating stratum, plus its energy report.

    Cross-fit, not in-fit. `beagle` is always scored by the basis fitted on
    `min_carriers` and the reverse, so no gating stratum is ever measured with
    a basis that saw it. The energy table then shows directly whether the
    hidden-state subspace transfers across domains, which is both the accuracy
    question and the compliance question.
    """
    moments = query_second_moments(mu)
    cache = QCOV_CACHE.with_name("query-basis.npz")
    if cache.exists():
        raw = {k: v for k, v in np.load(cache).items()}
    else:
        raw = {}
        for fit in GATING_STRATA:
            t0 = time.time()
            vectors = np.linalg.eigh(moments[fit])[1]
            raw[fit] = np.ascontiguousarray(
                vectors[:, ::-1][:, :QBASIS_RANK]).astype(np.float32)
            print(f"  query eigh {fit} in {time.time() - t0:.1f}s", flush=True)
        np.savez(cache, **raw)
    bases: dict[str, mx.array] = {}
    report: dict = {"fit_samples_note": "cross-fit; see e133-corpus.json for n",
                    "energy_kept": {}}
    grid = [k for k in (16, 32, 64, 128, 256, 512) if k <= QBASIS_RANK]
    for fit in GATING_STRATA:
        basis = raw[fit]
        bases[fit] = mx.array(basis)
        for test, cov in sorted(moments.items()):
            total = float(np.trace(cov))
            kept = np.cumsum(np.sum(basis * (cov @ basis), axis=0))
            report["energy_kept"][f"fit_{fit}/test_{test}"] = {
                str(k): float(kept[k - 1] / total) for k in grid}
        own = report["energy_kept"][f"fit_{fit}/test_{fit}"]
        print(f"  query basis fit on {fit}: keeps "
              + " ".join(f"k{k}={own[str(k)]:.4f}" for k in grid), flush=True)
    for fit in GATING_STRATA:
        other = [s for s in GATING_STRATA if s != fit][0]
        held = report["energy_kept"][f"fit_{fit}/test_{other}"]
        print(f"  query basis fit on {fit}, held-out on {other}: "
              + " ".join(f"k{k}={held[str(k)]:.4f}" for k in grid), flush=True)
    return {k: v[:, :rank] for k, v in bases.items()}, report


class Sketch:
    """One (family, size) cell: row codes, centroid codes and the query rule."""

    def __init__(self, family: str, size: int, rows: mx.array, centroids: mx.array,
                 mu: mx.array, basis: mx.array | None):
        self.family = family
        self.size = size
        self.key = f"{family}{size}"
        if family == "exact":
            # Not a sketch. It scores with the shipped affine-2 values, so an
            # arm that keeps every probed row must reproduce the shipped chain
            # exactly. That is the control for the base itself.
            self.proj = None
            self.row_codes = self.row_scale = None
            self.cent_codes = self.cent_scale = None
            self.row_offset = self.cent_offset = None
            self.bytes_per_row = AFFINE2_ROW_BYTES
            self.proj_bytes = 0
            return
        if family == "simhash":
            rng = np.random.default_rng(SKETCH_SEED)
            proj = np.where(
                rng.standard_normal((H.HIDDEN, size)) >= 0.0, 1.0, -1.0).astype(np.float32)
            self.proj = mx.array(proj)
            self.row_codes, self.row_scale = self._simhash(rows, mu)
            self.cent_codes, self.cent_scale = self._simhash(centroids, mu)
            # packed bits + fp32 norm + fp32 mean offset
            self.bytes_per_row = size // 8 + 8
            self.proj_bytes = H.HIDDEN * size // 8      # 1-bit +-1 R
        elif family in ("lowrank", "qlowrank"):
            assert basis is not None
            self.proj = basis[:, :size]
            self.row_codes, self.row_scale = self._lowrank(rows, mu)
            self.cent_codes, self.cent_scale = self._lowrank(centroids, mu)
            # int8 codes + fp32 scale + fp32 mean offset
            self.bytes_per_row = size + 8
            self.proj_bytes = H.HIDDEN * size * 2       # bf16 basis
        elif family == "sign":
            # `size` is the scale GROUP, not a projected width: this family
            # keeps all 5,120 dimensions at one bit and never projects.
            assert H.HIDDEN % size == 0
            self.proj = None
            self.row_codes, self.row_scale = self._sign(rows, mu)
            self.cent_codes, self.cent_scale = self._sign(centroids, mu)
            # one bit plane + fp16 group scales + fp32 mean offset
            self.bytes_per_row = H.HIDDEN // 8 + 2 * (H.HIDDEN // size) + 4
            self.proj_bytes = 0
        else:
            raise SystemExit(f"unknown sketch family {family}")
        # Both families sketch the CENTRED row, but the exact score is
        # `row . x`, and `row . x = (row - mu).(x - mu) + row.mu + c(x)`.
        # Dropping `row.mu` would rank by the wrong quantity, so it is stored
        # per row as one fp32 and added back at score time.
        self.row_offset = mx.matmul(rows.astype(mx.float32), mu)
        self.cent_offset = mx.matmul(centroids.astype(mx.float32), mu)
        mx.eval(self.row_codes, self.row_scale, self.cent_codes, self.cent_scale,
                self.row_offset, self.cent_offset)

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

    def _sign(self, table: mx.array, mu: mx.array):
        """One bit per dimension with an fp16 scale per group of `size`.

        The group scale that minimizes `||d - s sign(d)||` is `mean|d|`. It is
        divided out by the row's mean scale so the stored code stays O(1) and
        fp16 accumulation cannot dominate the one-bit error, and the row gain
        is handed back through the usual per-row scale.
        """
        parts_c, parts_s = [], []
        step = 8192
        groups = H.HIDDEN // self.size
        for a in range(0, table.shape[0], step):
            block = (table[a: a + step].astype(mx.float32) - mu).reshape(
                -1, groups, self.size)
            scale = mx.mean(mx.abs(block), axis=2)
            gain = mx.maximum(mx.mean(scale, axis=1), 1e-12)
            code = (mx.where(block >= 0.0, 1.0, -1.0)
                    * (scale / gain[:, None])[:, :, None])
            parts_c.append(code.reshape(-1, H.HIDDEN).astype(mx.float16))
            parts_s.append(gain)
            mx.eval(parts_c[-1], parts_s[-1])
        return mx.concatenate(parts_c, axis=0), mx.concatenate(parts_s, axis=0)

    def query(self, x: mx.array, mu: mx.array) -> mx.array:
        centred = x.astype(mx.float32) - mu
        if self.family == "sign":
            return centred.astype(mx.float16)
        if self.family == "simhash":
            return mx.where(mx.matmul(centred, self.proj) >= 0.0, 1.0, -1.0)
        return mx.matmul(centred, self.proj)

    def _combine(self, dot, scale, offset, x_norm) -> mx.array:
        if self.family == "simhash":
            # `dot / size = 1 - 2 theta / pi`, so `cos(theta)` is
            # `sin(pi dot / 2 size)`, and `||row - mu|| ||x - mu|| cos(theta)`
            # estimates the centred inner product.
            centred = (scale[None, :] * x_norm[:, None]
                       * mx.sin(0.5 * math.pi * dot / self.size))
        else:
            centred = scale[None, :] * dot
        return centred + offset[None, :]

    def score_rows(self, q: mx.array, x_norm: mx.array) -> mx.array:
        return self._combine(mx.matmul(q, self.row_codes.T), self.row_scale,
                             self.row_offset, x_norm)

    def score_centroids(self, q: mx.array, x_norm: mx.array) -> mx.array:
        return self._combine(mx.matmul(q, self.cent_codes.T), self.cent_scale,
                             self.cent_offset, x_norm)


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


class Stratum:
    """Everything the gate and the price need, per stratum, for one cell."""

    __slots__ = ("n", "miss_abs", "miss_inc", "worse", "better", "recall",
                 "probe_hit", "survivor_hit", "subst_is_target", "subst_total",
                 "subst_live", "subst_shipped_is_target")

    def __init__(self) -> None:
        self.n = self.miss_abs = self.miss_inc = 0
        self.worse = self.better = self.recall = 0
        self.probe_hit = self.survivor_hit = 0
        self.subst_is_target = self.subst_total = self.subst_live = 0
        self.subst_shipped_is_target = 0


class Cell:
    __slots__ = ("by_stratum", "ranks")

    def __init__(self) -> None:
        self.by_stratum: dict[str, Stratum] = {}
        # F1.4c: survival counts `#{rank >= N}` per stratum, from one stored
        # rank per sample. Any grid width can be read without a rerun.
        self.ranks: dict[str, dict[int, int]] = {}

    def slot(self, stratum: str) -> Stratum:
        if stratum not in self.by_stratum:
            self.by_stratum[stratum] = Stratum()
            self.ranks[stratum] = {g: 0 for g in RANK_GRID}
        return self.by_stratum[stratum]

    def add(self, stratum: str, miss_abs, base_abs, miss_inc, recall, probe,
            survivor, subst_total: int = 0, subst_live: int = 0,
            subst_hit: int = 0, subst_shipped_hit: int = 0,
            watch: str | None = None) -> None:
        if watch:
            self.add(watch, miss_abs, base_abs, miss_inc, recall, probe,
                     survivor, subst_total, subst_live, subst_hit,
                     subst_shipped_hit)
        s = self.slot(stratum)
        s.n += int(miss_abs.size)
        s.miss_abs += int(miss_abs.sum())
        s.miss_inc += int(miss_inc.sum())
        s.worse += int((miss_abs & ~base_abs).sum())
        s.better += int((base_abs & ~miss_abs).sum())
        s.recall += int(recall.sum())
        s.probe_hit += int(probe.sum())
        s.survivor_hit += int(survivor.sum())
        s.subst_total += subst_total
        s.subst_live += subst_live
        s.subst_is_target += subst_hit
        s.subst_shipped_is_target += subst_shipped_hit

    def add_ranks(self, stratum: str, rank: np.ndarray,
                  watch: str | None = None) -> None:
        for name in ((stratum, watch) if watch else (stratum,)):
            self.slot(name)
            for g in RANK_GRID:
                self.ranks[name][g] += int((rank >= g).sum())

    def pooled(self, field: str) -> int:
        return sum(getattr(s, field) for name, s in self.by_stratum.items()
                   if name not in WATCH_STRATA)


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
        # The permuted view must cover all 98,336 padded positions, because
        # `order` indexes the padded table. Taking from the 98,330-column
        # reachable view would clamp the six padding positions to the last
        # real column and hand the rerank a score that belongs to another row.
        exact_perm = mx.take(H.scores_all(self.exact, x), self.index.order_mx,
                             axis=1)
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
        mx.eval(argmax_row, exact_perm, coarse_perm, gt_perm, pos_r, leaf_r,
                gt_leaf, order_a2, crank_a2)
        return {
            "b": b,
            "argmax_row": argmax_row,
            "exact_perm": exact_perm,
            "coarse_perm": coarse_perm,
            "gt_perm": gt_perm,
            "gt_leaf": gt_leaf,
            "pos_r": pos_r,
            "leaf_r": leaf_r,
            "order_a2": order_a2,
            "crank_a2": crank_a2,
        }

    def probe_positions(self, order_c: mx.array, clusters: int, b: int) -> mx.array:
        positions = (order_c[:, :clusters, None] * IX.ROWS_PER_LEAF
                     + mx.arange(IX.ROWS_PER_LEAF, dtype=mx.int32)[None, None, :])
        return positions.reshape(b, clusters * IX.ROWS_PER_LEAF).astype(mx.int32)

    def output_row(self, f, positions: mx.array, order_key: mx.array) -> mx.array:
        """Stage C then D: affine-2 top-32 of `positions`, then affine-4 argmax.

        `order_key` ranks the candidates for the top-32 cut, so the shipped
        chain passes its affine-2 scores and an arm passes the same affine-2
        scores restricted to its survivors. Returns the PERMUTED position, the
        only form `clusterPerm` accepts (D5 hazard 1).
        """
        keep = min(SHORTLIST, order_key.shape[1])
        top = mx.argpartition(-order_key, kth=keep - 1, axis=1)[:, :keep]
        chosen = mx.take_along_axis(positions, top, axis=1)
        exact = mx.take_along_axis(f["exact_perm"], chosen, axis=1)
        return mx.take_along_axis(
            chosen, mx.argmax(exact, axis=1)[:, None].astype(mx.int32), axis=1)[:, 0]

    def compact(self, positions: mx.array) -> mx.array:
        """Permuted position -> compact id, folding the six padding rows back
        onto the rows they copy, exactly as `clusterPerm` does."""
        return mx.take(self.index.perm_mx, positions, axis=0)

    def shipped(self, f):
        """The live chain: affine-2 centroids, 25 % probe, affine-2 top-32.

        Returns the COMPACT id, not the permuted position, so a padding
        position and the real row it copies never read as two different
        answers (D5 hazard 1).

        The miss it returns is the SAME comparison every arm reports, namely
        `output != exact affine-4 argmax`. An earlier version returned a
        structural proxy built from `crank_a2` and the top-32 cut instead. The
        proxy counted 1 miss in 11,244 while the true rate is about a hundred
        times larger, so subtracting it from an arm's true miss inflated every
        net figure by the shipped chain's own approximation error. The
        structural flag is still returned, but only as a diagnostic.
        """
        clusters = max(1, math.ceil(SHIPPED_PROBE_FRACTION * LEAVES))
        positions = self.probe_positions(f["order_a2"], clusters, f["b"])
        keys = mx.take_along_axis(f["coarse_perm"], positions, axis=1)
        out = self.compact(self.output_row(f, positions, keys))
        cum_gt = mx.cumsum(
            mx.take_along_axis(f["gt_leaf"], f["order_a2"], axis=1), axis=1)
        structural = mx.logical_or(mx.logical_not(f["crank_a2"] < clusters),
                                   cum_gt[:, clusters - 1] >= SHORTLIST)
        miss = out != f["argmax_row"]
        mx.eval(miss, structural, out)
        return (np.asarray(miss).astype(bool), out,
                np.asarray(structural).astype(bool))


def sketch_state(screen: Screen, sketch: Sketch, f, x: mx.array) -> dict:
    """The per-batch work that both `stage_a` variants of a cell family share.

    Scoring all 98,336 rows is the dominant cost of the sweep, so it happens
    once per (sketch, batch) rather than once per arm.
    """
    if sketch.family == "exact":
        return {"order_c": f["order_a2"], "row_perm": f["coarse_perm"]}
    q = sketch.query(x, screen.mu)
    x_norm = mx.linalg.norm(x.astype(mx.float32) - screen.mu, axis=1)
    order_c = mx.argsort(-sketch.score_centroids(q, x_norm), axis=1)
    row_perm = mx.take(sketch.score_rows(q, x_norm), screen.index.order_mx, axis=1)
    mx.eval(order_c, row_perm)
    return {"order_c": order_c, "row_perm": row_perm}


def run_arm(screen: Screen, sketch: Sketch, f, state: dict, probe_fractions,
            survivor_widths, base_miss: np.ndarray, base_compact: mx.array,
            stratum: str, reference: np.ndarray, base_vocab: np.ndarray,
            live: np.ndarray, cells: dict[tuple, Cell],
            stage_a: str = "sketch", watch: str | None = None) -> None:
    """One sketch cell family. `stage_a` chooses who orders the leaves.

    `sketch` sketches both stages, which is C1 as designed. `affine2` keeps
    today's exact centroid readout and sketches only the 24,584-row stage, so a
    stage-A failure and a stage-B failure can be told apart and priced apart.
    """
    b = f["b"]
    order_c = state["order_c"] if stage_a == "sketch" else f["order_a2"]
    row_perm = state["row_perm"]
    max_clusters = max(max(1, math.ceil(p * LEAVES)) for p in probe_fractions)
    positions_full = screen.probe_positions(order_c, max_clusters, b)
    # Compare compact ids, never permuted positions: the padded table repeats
    # six rows, so two positions can name the same answer (D5 hazard 1).
    comp_full = screen.compact(positions_full)
    sk_full = mx.take_along_axis(row_perm, positions_full, axis=1)
    co_full = mx.take_along_axis(f["coarse_perm"], positions_full, axis=1)
    is_r_full = comp_full == f["argmax_row"][:, None]
    mx.eval(sk_full, co_full, is_r_full, comp_full)

    max_width = max(survivor_widths)
    for p in probe_fractions:
        clusters = max(1, math.ceil(p * LEAVES))
        width = clusters * IX.ROWS_PER_LEAF
        sk = sk_full[:, :width]
        co = co_full[:, :width]
        is_r = is_r_full[:, :width]
        comp = comp_full[:, :width]
        positions = positions_full[:, :width]
        # Read the probe hit off the probed set itself rather than off the
        # canonical leaf of the argmax row: a padding copy can sit in another
        # leaf, and then `crank_r < clusters` and `is_r` would disagree.
        probe_hit = mx.any(is_r, axis=1)

        # F1.4c. The rank of the exact argmax in the sketch ordering. A sample
        # whose leaf was never probed is beyond every width, so it is parked at
        # `width` rather than dropped, which would bias the curve low.
        sk_at_r = mx.max(mx.where(is_r, sk, mx.full(sk.shape, -3.4e38, mx.float32)),
                         axis=1)
        rank = mx.where(probe_hit,
                        mx.sum((sk > sk_at_r[:, None]).astype(mx.int32), axis=1),
                        width)
        mx.eval(rank)
        rank_np = np.asarray(rank)

        top = mx.argpartition(-sk, kth=max_width - 1, axis=1)[:, :max_width]
        top = mx.take_along_axis(
            top, mx.argsort(-mx.take_along_axis(sk, top, axis=1), axis=1), axis=1)
        exact_top1 = mx.take_along_axis(
            comp, mx.argmax(co, axis=1)[:, None].astype(mx.int32), axis=1)
        mx.eval(top, exact_top1)

        for n_keep in survivor_widths:
            sel = top[:, :n_keep]
            out = screen.compact(
                screen.output_row(f, mx.take_along_axis(positions, sel, axis=1),
                                  mx.take_along_axis(co, sel, axis=1)))
            survivor = mx.any(mx.take_along_axis(is_r, sel, axis=1), axis=1)
            recall = mx.any(
                mx.take_along_axis(comp, sel, axis=1) == exact_top1, axis=1)
            mx.eval(out, survivor, recall)

            miss_abs = np.asarray(out != f["argmax_row"]).astype(bool)
            miss_inc = np.asarray(out != base_compact).astype(bool)
            # F1.5, corrected for prefix-monotone acceptance. A substitution
            # can only cost acceptance on a LIVE row, because a round discards
            # every row after its first rejection. On a live row the shipped
            # token was accepted exactly when it matched the reference, and
            # the substitute is accepted exactly when it matches instead.
            # Both sides are counted on the SAME live substituted rows, so the
            # difference is the exact net acceptance loss.
            swapped = miss_inc & live
            hit = shipped_hit = 0
            if swapped.any():
                out_vocab = np.asarray(H.compact_to_vocab(out))
                hit = int((out_vocab[swapped] == reference[swapped]).sum())
                shipped_hit = int((base_vocab[swapped] == reference[swapped]).sum())
            cell = cells.setdefault((sketch.key, stage_a, n_keep, p), Cell())
            cell.add(stratum, miss_abs, base_miss, miss_inc,
                     np.asarray(recall).astype(bool),
                     np.asarray(probe_hit).astype(bool),
                     np.asarray(survivor).astype(bool),
                     subst_total=int(miss_inc.sum()),
                     subst_live=int(swapped.sum()), subst_hit=hit,
                     subst_shipped_hit=shipped_hit, watch=watch)
            # The rank does not depend on `n_keep`, so every cell of this
            # (family, stage_a, probe) group carries the same curve and each
            # one can be tail-fitted at its own survivor width.
            cell.add_ranks(stratum, rank_np, watch)
            del sel, out, survivor, recall
        del sk, co, is_r, comp, positions, top, exact_top1
    del sk_full, co_full, is_r_full, comp_full, positions_full


# --------------------------------------------------------------------------
# reporting


def summarize(arm: str, cell: Cell, model: dict, extra: dict,
              p_by_stratum: dict) -> dict:
    by_stratum = {}
    for name, s in cell.by_stratum.items():
        m_abs, abs_lo, abs_hi = clopper_pearson(s.miss_abs, s.n)
        m_inc, inc_lo, inc_hi = clopper_pearson(s.miss_inc, s.n)
        net = (s.worse - s.better) / s.n if s.n else float("nan")
        disc = s.worse + s.better
        _, dlo, dhi = wilson(s.worse, disc)
        q = (s.subst_is_target / s.subst_live) if s.subst_live else 0.0
        # `p` in F1.5, measured on the LIVE substituted rows themselves. The
        # pooled per-step acceptance is reported too, but pricing a
        # conditional event with a pooled rate would mix two populations.
        p_sub = ((s.subst_shipped_is_target / s.subst_live)
                 if s.subst_live else 0.0)
        p_step = p_by_stratum.get(name, float("nan"))
        loss = (s.subst_shipped_is_target - s.subst_is_target) / s.n if s.n else 0.0
        by_stratum[name] = {
            "gating": name in GATING_STRATA,
            "watch": name in WATCH_STRATA,
            "n": s.n,
            "misses_absolute": s.miss_abs,
            "m_absolute": m_abs, "m_absolute_lo": abs_lo, "m_absolute_hi": abs_hi,
            "misses_incremental": s.miss_inc,
            "m_incremental": m_inc, "m_incremental_lo": inc_lo,
            "m_incremental_hi": inc_hi,
            "net_miss": net,
            "net_miss_lo": (2 * dlo - 1) * disc / s.n if disc and s.n else 0.0,
            "net_miss_hi": (2 * dhi - 1) * disc / s.n if disc and s.n else 0.0,
            "discordant": disc,
            "recall": s.recall / s.n if s.n else float("nan"),
            "probe_hit_rate": s.probe_hit / s.n if s.n else float("nan"),
            "survivor_hit_rate": s.survivor_hit / s.n if s.n else float("nan"),
            "p_head_step_accuracy": p_step,
            "p_shipped_is_target_on_live_substituted": p_sub,
            "q_substitute_is_target": q,
            "substitutions": s.subst_total,
            "substitutions_live": s.subst_live,
            # F1.5: the realised acceptance loss, not the raw miss rate. It is
            # `(live substitutions / n) * (p_sub - q)` by construction.
            "acceptance_loss": loss,
            "acceptance_loss_pooled_p": m_inc * (p_step - q) if p_step == p_step
                                        else None,
            "survival_curve": {str(g): cell.ranks.get(name, {}).get(g, 0)
                               for g in RANK_GRID},
            "tail_fit_at_survivors": tail_fit(
                cell.ranks.get(name, {}), s.n, int(extra.get("survivors", 256))),
        }

    gating = {k: v for k, v in by_stratum.items() if k in GATING_STRATA}

    def worst(field: str) -> float:
        return max((v[field] for v in gating.values()), default=float("nan"))

    def lowest(field: str) -> float:
        return min((v[field] for v in gating.values()), default=float("nan"))

    # T0 is the ABSOLUTE NET miss: the arm's extra absolute misses less the
    # shipped chain's absolute misses that the arm repairs. Raw `m_absolute`
    # is reported beside it because the shipped chain is itself approximate.
    net_worst = worst("net_miss")
    loss = max((v["acceptance_loss"] for v in gating.values()
                if v["acceptance_loss"] is not None), default=0.0)
    watched = by_stratum.get("essays_bacon")
    return {
        "arm": arm,
        **extra,
        # A watch line repeats rows a real stratum already counted.
        "n": sum(v["n"] for k, v in by_stratum.items() if k in ALL_STRATA),
        "n_gating": sum(v["n"] for v in gating.values()),
        "net_miss_worst_gating": net_worst,
        "net_miss_worst_gating_hi": worst("net_miss_hi"),
        "m_absolute_worst_gating": worst("m_absolute"),
        "m_absolute_worst_gating_hi": worst("m_absolute_hi"),
        "m_incremental_worst_gating": worst("m_incremental"),
        "recall_worst_gating": lowest("recall"),
        "acceptance_loss_worst_gating": loss,
        # F2.3. The hardest carrier, on its own line, never gating.
        "net_miss_essays_bacon": watched["net_miss"] if watched else None,
        "m_absolute_essays_bacon": watched["m_absolute"] if watched else None,
        "recall_essays_bacon": watched["recall"] if watched else None,
        **model,
        # F1.5: price on the realised acceptance loss, kill on absolute miss.
        "predicted_pct_gating": model["pct_head_share_7"] - MISS_TO_SCORE_PCT * loss,
        "predicted_pct_raw_miss":
            model["pct_head_share_7"] - MISS_TO_SCORE_PCT * worst("m_incremental"),
        "passes_t0": net_worst <= T0_NET_MISS,
        "passes_t0b": lowest("recall") >= T0B_RECALL,
        "by_stratum": by_stratum,
    }


# --------------------------------------------------------------------------
# commands


class SketchSet:
    """One screen cell, held as one sketch or as a cross-fitted pair.

    Every fold shares the family, the size and the byte cost, so the cells they
    feed stay comparable. Only the basis differs, and `pick` guarantees that a
    gating stratum is never scored by the fold fitted on itself.
    """

    def __init__(self, folds: dict[str, Sketch], route: dict[str, str]) -> None:
        self.folds = folds
        self.route = route
        head = next(iter(folds.values()))
        self.key = head.key
        self.family = head.family
        self.size = head.size
        self.bytes_per_row = head.bytes_per_row
        self.proj_bytes = head.proj_bytes
        self.cross_fit = len(folds) > 1

    def pick(self, stratum: str) -> Sketch:
        return self.folds[self.route.get(stratum, self.route["*"])]


def build_sketches(screen: Screen,
                   families: list[tuple[str, int]]) -> tuple[list[SketchSet], dict]:
    need_row = max([s for f, s in families if f == "lowrank"], default=0)
    need_query = max([s for f, s in families if f == "qlowrank"], default=0)
    row_basis = pca_basis(screen.rows, screen.mu, need_row) if need_row else None
    query_report: dict = {}
    query_bases_by_fit: dict[str, mx.array] = {}
    if need_query:
        query_bases_by_fit, query_report = query_bases(screen.mu, need_query)
    # Score the held-out fold: `beagle` reads the `min_carriers` basis and the
    # reverse. `zero_weight` never gates, so it reads the `beagle` fold.
    route = {"beagle": "min_carriers", "min_carriers": "beagle",
             "zero_weight": "beagle", "*": "beagle"}
    out = []
    for family, size in families:
        t0 = time.time()
        if family == "qlowrank":
            folds = {fit: Sketch(family, size, screen.rows, screen.centroids,
                                 screen.mu, basis)
                     for fit, basis in query_bases_by_fit.items()}
            out.append(SketchSet(folds, route))
        else:
            sketch = Sketch(family, size, screen.rows, screen.centroids,
                            screen.mu, row_basis)
            out.append(SketchSet({"*": sketch}, {"*": "*"}))
        print(f"  sketch {family}{size}: {out[-1].bytes_per_row} B/row, "
              f"R {out[-1].proj_bytes} B, {len(out[-1].folds)} fold(s), "
              f"built in {time.time() - t0:.1f}s", flush=True)
    return out, query_report


def parse_families(spec: str) -> list[tuple[str, int]]:
    families = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        family = next(f for f in ("simhash", "qlowrank", "lowrank", "sign",
                                  "exact") if token.startswith(f))
        families.append((family, int(token[len(family):])))
    return families


def cmd_validate(args) -> None:
    index = Index(Path(args.index))
    screen = Screen(index)
    # A control that cannot fail is not a control: an 8-bit SimHash must lose
    # the argmax constantly, or the miss column below proves nothing.
    damaged = Sketch("simhash", 8, screen.rows, screen.centroids, screen.mu, None)

    n = tok_ok = shipped_miss = damaged_miss = 0
    accept_hits = verdict_ok = shipped_reproduces = live_rows = accepted_rows = 0
    structural_miss = structural_agree = 0
    ranks: list[int] = []
    for _, stratum, x, proposal, reference, accepted, live in chunks(args.batch,
                                                                     args.limit):
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
        # F1.5: `p`, the shipped chain's per-row match with the target.
        accept_hits += int((proposal == reference).sum())
        # Acceptance is prefix-monotone, so the parent's verdict must factor
        # EXACTLY as `prefix_live AND match`. If it does not, the price model's
        # notion of a live row is wrong and every cell would be mispriced.
        verdict_ok += int((((proposal == reference) & live) == accepted).sum())
        live_rows += int(live.sum())
        accepted_rows += int(accepted.sum())

        base_miss, base_compact, structural = screen.shipped(f)
        base_vocab = np.asarray(H.compact_to_vocab(base_compact))
        shipped_reproduces += int((base_vocab == proposal).sum())
        shipped_miss += int(base_miss.sum())
        structural_miss += int(structural.sum())
        structural_agree += int((structural == base_miss).sum())
        cells: dict[tuple, Cell] = {}
        run_arm(screen, damaged, f, sketch_state(screen, damaged, f, x),
                [SHIPPED_PROBE_FRACTION], [256],
                base_miss, base_compact, stratum, reference, base_vocab,
                live, cells)
        damaged_miss += cells[
            (damaged.key, "sketch", 256, SHIPPED_PROBE_FRACTION)].pooled("miss_abs")
        n += f["b"]

    _, lo, hi = clopper_pearson(shipped_miss, n)
    report = {
        "samples": n,
        "offline_argmax_matches_runtime_proposal": tok_ok / n if n else float("nan"),
        "proposal_mismatch": {"count": len(ranks), "rank_max": max(ranks, default=0)},
        "ledger_verdict_factors_as_live_and_match":
            verdict_ok / n if n else float("nan"),
        "offline_shipped_chain_reproduces_runtime":
            shipped_reproduces / n if n else float("nan"),
        "p_row_matches_reference": accept_hits / n if n else float("nan"),
        "p_row_accepted": accepted_rows / n if n else float("nan"),
        "prefix_live_rate": live_rows / n if n else float("nan"),
        "m_shipped_live_chain": {
            "misses": shipped_miss, "p": shipped_miss / n if n else float("nan"),
            "lo": lo, "hi": hi},
        # The shipped chain is an approximate readout: it probes a quarter of
        # the leaves and reranks only 32 rows, so it loses the exact affine-4
        # argmax at a rate that must be subtracted from every arm. The
        # structural flag underestimates that rate badly, which is why it is
        # reported here and no longer used as the base of any net figure.
        "m_shipped_structural_proxy": {
            "misses": structural_miss,
            "p": structural_miss / n if n else float("nan"),
            "agrees_with_true_miss": structural_agree / n if n else float("nan")},
        "m_damaged_simhash8_control": {
            "misses": damaged_miss, "p": damaged_miss / n if n else float("nan")},
    }
    report["control_can_fail"] = (
        report["m_damaged_simhash8_control"]["p"]
        > 10 * max(report["m_shipped_live_chain"]["p"], 1e-6))
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
    sketches, query_report = build_sketches(screen, families)

    cells: dict[tuple, Cell] = {}
    base_cell = Cell()
    structural_cell = Cell()
    accept: dict[str, list[int]] = {}
    n = 0
    t0 = time.time()
    for seed, stratum, x, proposal, reference, accepted, live in chunks(
            args.batch, args.limit):
        watch = seed if seed in WATCH_STRATA else None
        f = screen.front(x)
        base_miss, base_compact, structural = screen.shipped(f)
        base_vocab = np.asarray(H.compact_to_vocab(base_compact))
        falses = np.zeros_like(base_miss)
        base_cell.add(stratum, base_miss, base_miss, falses, ~falses, ~falses,
                      ~falses, watch=watch)
        structural_cell.add(stratum, structural, structural, falses, ~falses,
                            ~falses, ~falses, watch=watch)
        for key in (stratum, watch) if watch else (stratum,):
            slot = accept.setdefault(key, [0, 0, 0, 0])
            slot[0] += int(proposal.size)
            slot[1] += int((base_vocab == reference).sum())
            slot[2] += int((base_vocab == proposal).sum())
            slot[3] += int(accepted.sum())
        for entry in sketches:
            sketch = entry.pick(stratum)
            state = sketch_state(screen, sketch, f, x)
            for stage_a in stages:
                run_arm(screen, sketch, f, state, fractions, widths, base_miss,
                        base_compact, stratum, reference, base_vocab, live,
                        cells, stage_a, watch)
            del state
        n += f["b"]
        if n % (args.batch * 10) == 0:
            print(f"  {n} samples  {time.time() - t0:.0f}s", flush=True)

    p_by_stratum = {k: (v[1] / v[0] if v[0] else float("nan"))
                    for k, v in accept.items()}
    real = {k: v for k, v in accept.items() if k in ALL_STRATA}
    aligned = sum(v[0] for v in real.values())
    out = {
        "samples": n,
        "base_sha": args.base_sha,
        "wall_seconds": time.time() - t0,
        "p_head_step_accuracy": (sum(v[1] for v in real.values()) / aligned
                                 if aligned else float("nan")),
        "p_head_step_accuracy_by_stratum": p_by_stratum,
        # The emitted-token rate, which is strictly below the row match rate
        # because a round discards every row after its first rejection.
        "p_row_accepted": (sum(v[3] for v in real.values()) / aligned
                           if aligned else float("nan")),
        "p_row_accepted_by_stratum": {
            k: (v[3] / v[0] if v[0] else float("nan"))
            for k, v in accept.items()},
        "offline_shipped_chain_reproduces_runtime":
            (sum(v[2] for v in real.values()) / aligned
             if aligned else float("nan")),
        "shipped": summarize("shipped", base_cell, price(0), {}, p_by_stratum),
        "shipped_structural_proxy": summarize(
            "shipped-structural", structural_cell, price(0), {}, p_by_stratum),
        "query_basis": query_report,
        "cells": [],
    }
    shipped_bytes = shipped_stage_bytes()
    for (key, stage_a, n_keep, p), cell in cells.items():
        sketch = next(s for s in sketches if s.key == key)
        arm_bytes = arm_stage_bytes(sketch.bytes_per_row, sketch.proj_bytes,
                                    n_keep, p, stage_a)
        tag = "" if stage_a == "sketch" else "-hybridA"
        out["cells"].append(summarize(
            f"{key}{tag}-N{n_keep}-p{p:g}", cell, price(shipped_bytes - arm_bytes),
            {"family": sketch.family, "size": sketch.size, "stage_a": stage_a,
             "bytes_per_row": sketch.bytes_per_row, "proj_bytes": sketch.proj_bytes,
             "survivors": n_keep, "probe_fraction": p, "cross_fit": sketch.cross_fit,
             "arm_stage_bytes": arm_bytes, "shipped_stage_bytes": shipped_bytes},
            p_by_stratum))
    out["cells"].sort(key=lambda c: -c["predicted_pct_gating"])
    # F2.1. `hybridA` is a candidate, not a control. A stage-A kill must not
    # take it down with full C1, so each stage_a gets its own gate table and
    # its own selected cell.
    out["by_stage_a"] = {}
    for stage_a in stages:
        arms = [c for c in out["cells"] if c["stage_a"] == stage_a]
        ok = [c for c in arms if c["passes_t0"] and c["passes_t0b"]]
        best = max(ok, key=lambda c: c["predicted_pct_gating"], default=None)
        out["by_stage_a"][stage_a] = {
            "label": "full C1" if stage_a == "sketch" else "hybridA",
            "cells": len(arms),
            "cells_passing_t0": sum(1 for c in arms if c["passes_t0"]),
            "cells_passing_t0b": sum(1 for c in arms if c["passes_t0b"]),
            "cells_passing_both": len(ok),
            "best_arm": best["arm"] if best else None,
            "best_predicted_pct": best["predicted_pct_gating"] if best else 0.0,
            "best_cell": best,
        }
        print(f"\n=== {stage_a}  ({out['by_stage_a'][stage_a]['label']})  "
              f"{len(ok)}/{len(arms)} cells clear T0 and T0b")
        print(f"{'arm':30s}{'B/row':>7s}{'netWorst':>11s}{'mAbsWorst':>11s}"
              f"{'mInc':>11s}{'loss':>11s}{'recall':>9s}{'bacon_net':>11s}"
              f"{'gain%':>8s}{'pred%':>8s}{'T0':>4s}{'T0b':>5s}")
        for cell in arms[: args.top]:
            bacon = cell["net_miss_essays_bacon"]
            print(f"{cell['arm']:30s}{cell['bytes_per_row']:7d}"
                  f"{cell['net_miss_worst_gating']:11.3e}"
                  f"{cell['m_absolute_worst_gating']:11.3e}"
                  f"{cell['m_incremental_worst_gating']:11.3e}"
                  f"{cell['acceptance_loss_worst_gating']:11.3e}"
                  f"{cell['recall_worst_gating']:9.5f}"
                  f"{bacon if bacon is not None else float('nan'):11.3e}"
                  f"{cell['pct_head_share_7']:8.3f}"
                  f"{cell['predicted_pct_gating']:8.3f}"
                  f"{'ok' if cell['passes_t0'] else 'NO':>4s}"
                  f"{'ok' if cell['passes_t0b'] else 'NO':>5s}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("selftest")
    t.set_defaults(func=cmd_selftest)

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
                           "lowrank32,lowrank64,lowrank128,lowrank256,"
                           "qlowrank64,qlowrank128,qlowrank256,"
                           "sign64,sign256,sign5120")
    s.add_argument("--widths", default="64,256,1024,4096,8192,16384")
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
