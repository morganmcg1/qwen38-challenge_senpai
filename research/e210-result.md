# E210 — GPU-idle census of a whole candidate MTP leg

SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"yukon_submission_id":null,"primary_metric":{"name":"mtp_leg_gpu_idle_pct","available":true,"value":1.660},"test_metric":{"name":"all_tokens_matched","available":true,"value":1}}

- Student / branch: `qwen-alphonse` / `qwen-alphonse/e210-gpu-idle-census`
- Hypothesis and target cost: the timed candidate MTP leg holds a large GPU-idle
  pool that a cross-round overlap or pipelining family could harvest. The census
  measures that pool and places it.
- Decision: **measured — 1.660% of the MTP leg is GPU idle.** This lands in the
  pre-registered 1–3% band, so the idle map is reported and the advisor decides.
  No candidate-surface change is proposed.
- `BASE_SHA` / `UPSTREAM_SHA` / candidate commit:
  `22ca1b48c455b83b489afcd59badec2cab84ec5d` / unchanged / no candidate commit.
  The traced research build recorded `base_sha=35fdcb0adbcbe2d687278489eea17d35cdc2814e`,
  which is this branch's instrumented commit, not a candidate.
- Yukon promoted submission / source ref used as frontier: not applicable. This
  experiment is research-only and never proposes a submission.
- Candidate build fingerprint:
  `worker_sha256=2b21eff287383ca6f426b7a05da66a4e2860bccd8426fdbd5218dd458cbcbcad`,
  `cli_sha256=c1862d9659f8768653b6039861b39d3149fbffe4ac9e84c218d654778c6f6c8d`,
  `metallib_source_fingerprint=ce5e223eb6b404f16f07aca83ca44e65fc517187f3b130edaf4e7acb56789d50`.
  That worker carries the E90 interval ledger and the E210 anchors. **The
  instrumentation is stripped from this branch's terminal state**, so the
  scored-surface diff against `BASE_SHA` is empty.
- Submitted-surface / generated-twin / metallib digests: not applicable; nothing
  is submitted.
- Submitted candidate files: none.
- Supporting test, tooling, or documentation files:
  `research/e210_census.py`, `research/e210_session.sh`,
  `research/e210_summary.py`, `research/e210_wandb.py`,
  `research/e210-s1-census.json`, `research/e210-s0-off.json`,
  `research/e210-s0-on.json`, `research/e210-s0b-off.json`,
  `research/e210-s0b-on.json`, `research/e210-result.md`.
- MTP head provenance, digest, and draft policy:
  `head_provenance_sha256=dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`
  on every leg. Draft policy unchanged from the base;
  `effective_mean_draft_len=6.359`, `accepted_draft_rate=0.877`.
- Token window, fixture, reference source, and harness: 512 decode tokens after
  a 512-token seed, the public `--local-iterate` fixture, candidate-generated
  reference rows, **harness=local**.
- Exact cell: whole-leg attribution, not a kernel cell. The device axis is the
  union of `MTLCommandBuffer` `gpuStartTime`/`gpuEndTime` intervals from the
  E90 ledger, on the same mach uptime clock as the session anchors.
- Official causal path and score equation: none is claimed. This is an
  observation of where candidate MTP wall time goes. It sets an upper bound on
  what any idle-removing mechanism could return, and that bound multiplies
  candidate MTP seconds per token, which is the ranked denominator.
- Assignment-scope preflight: `senpai/validate-assignment-scope.sh` reports an
  empty scored-surface diff for the terminal branch state.
- Editable source bytes / headroom / growth / exempt-head bytes: unchanged from
  base; candidate growth is 0 bytes.
- Scored-path reachability evidence: the anchors are emitted by the live
  `Qwen36MTPBlockSession` round loop of the timed worker. The anchor span
  reproduces the parent-measured leg time to within 0.66 ms in 14 588 ms
  (0.005%), so the census covers the leg the parent charged.
- Written promotion rule and verdict: pre-registered. idle ≥ 3% promotes the
  overlap family; idle ≤ 1% closes it campaign-wide; 1–3% reports the map.
  Measured 1.660% → **report the map; advisor decides.**
- Pre-official evidence budget / timed legs used: Stage-0 validation
  (2 sessions × 2 arms) plus one traced 512-token session, as assigned.
- Frozen candidate SHA: none.
- Specific evidence that invalidated a frozen SHA: not applicable.
- Submission owner / receipt-watcher job ID: not applicable.

## Evidence

- Host, instance, chip, memory profile, toolchain, and thermal policy:
  `ip-10-231-2-22.ec2.internal`, Apple M4 Pro, 51 539 607 552 bytes,
  `swift build -c release --force-resolved-versions`, sandbox off,
  `MLXFAST_LOCAL_COOL_GATE=0`. Entry GPU temperature 37.82 C, exit 60.19 C.
  `cool_gate_passed_real_gate=false` and `gate_qualified_for_timing=false` are
  preserved. These legs are attribution observations. They are not gate-qualified
  timing and are never comparable to a gated historical run.
- `head_provenance_sha256` for every leg, baseline and candidate:
  `dadbfb806d80eca258395e5360534c5969acd5ad312b45102ad2caf65566f7e9`.
- Exact baseline and candidate commands:

  ```bash
  research/e210_session.sh e210-s0-off  128        # positive control, arm off
  research/e210_session.sh e210-s0-on   128 5000   # positive control, arm on
  research/e210_session.sh e210-s0b-off 128        # rebuilt, arm off
  research/e210_session.sh e210-s0b-on  128 5000   # rebuilt, arm on
  research/e210_session.sh e210-s1-census 512      # the census session
  python3 research/e210_census.py e210-s1-census --json research/e210-s1-census.json
  python3 research/e210_summary.py research/e210-s1-census.json
  ```

- Cheapest real falsification gate and positive-control verdict: **passed.** A
  planted `usleep` at the round top was measured in-run by its own anchor.
  On the serial leg the planted span was 7116.9 µs per round and the census
  charged 7116.9 µs of idle inside it — 100.00%. On the MTP leg the planted span
  was 6704.7 µs and the census charged 6265.6 µs — 93.4%; the 439 µs shortfall
  is the prefetched head step that genuinely executes at the round top, which is
  the expected sign. Leg idle rose from 1.353% to 3.435% (MTP) and 2.298% to
  9.513% (serial) when the stall was planted. The census can therefore see idle
  it is asked to see, and the 1.660% reading is not a floor artefact.
- Tests and risk-based checks, in execution order: build; Stage-0 arm-off and
  arm-on at 128 tokens on two independent builds; off-arm reproducibility
  (MTP 1.301% and 1.353%, serial 2.222% and 2.298%); anchor-span versus
  parent-leg cross-check; the 512-token census.
- Exact-token and row-ledger verdict: `all_tokens_matched=true`,
  `residual_divergence_count=0` on every session including the census.
- Divergent tokens or failure category: none.
- Generated-twin audit: not relevant; no Metal source changed.
- Peak RAM or head/artifact size: unchanged.
- Official status and score: not submitted.

### The census, 512 tokens, MTP leg (pid 21242, 78 rounds, 70 analysed)

| Segment | Span (ms) | Share of leg | GPU idle (ms) | Idle % of segment | Share of leg idle |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed (charged 512-token prefill) | 3919.1 | 26.87% | 20.44 | 0.52% | 8.44% |
| prologue | 1.0 | 0.01% | 1.04 | 100.00% | 0.43% |
| rounds | 9624.8 | 65.98% | 187.49 | 1.95% | 77.42% |
| turnaround (inter-round) | 39.4 | 0.27% | 0.52 | 1.33% | 0.22% |
| **leg total** | **14587.8** | **100%** | **242.17** | **1.660%** | **100%** |

Production-equivalent leg idle, with the instrument's own 2.68 ms removed, is
**1.642%**. The eight skipped warmup rounds hold the balance of leg idle that
the four analysed segments do not tile.

Per-round idle inside a round: median 2309.2 µs, p90 3166.1 µs, against a mean
round span of 137.5 ms.

### Where the round idle sits (per analysed round)

| Phase | Span/round (µs) | Idle/round (µs) | Idle % of phase |
| --- | ---: | ---: | ---: |
| `commit` | 1872.8 | 1192.5 | 63.68% |
| `d_chain` | 999.8 | 600.8 | 60.09% |
| `d_submit2` | 2047.2 | 569.1 | 27.80% |
| `eval_wall` | 63897.3 | 202.3 | 0.32% |
| `readout` | 44.2 | 44.2 | 100.00% |
| `verify_graph` | 68071.8 | 36.2 | 0.05% |
| `row_dump` (instrument) | 283.3 | 32.1 | 11.34% |
| `snapshot` | 195.2 | 0.0 | 0.00% |

The two phases that hold 93% of the round's wall time — `verify_graph` and
`eval_wall`, 132.0 ms of 137.5 ms — are 99.7% GPU busy. All the located idle is
in the short CPU-side seams: draft chain build, second submit, readout, and
commit.

### Coherence of the idle

292 idle slices of at least 100 µs hold 184.45 ms, which is 2635 µs per round
and 1.909% of the censused span. Almost all round-region idle is therefore
coherent rather than microscopic dust; a mechanism that could fill it would not
be fighting sub-10 µs gaps.

Each slice is tagged with the phase it starts in. Slices merge across phase
boundaries, so a slice starting late in `eval_wall` runs through `readout` into
`commit`.

| Starts in | Slices | Total (ms) | Per round (µs) | Max (µs) |
| --- | ---: | ---: | ---: | ---: |
| `d_chain` | 64 | 62.6 | 894.0 | 12999.3 |
| `eval_wall` | 70 | 62.4 | 891.9 | 9278.5 |
| `commit` | 85 | 39.8 | 568.0 | 1558.0 |
| `d_submit2` | 70 | 18.1 | 258.1 | 411.2 |
| `d_pre` | 1 | 1.2 | 16.6 | 1162.4 |
| `verify_graph` | 2 | 0.4 | 6.4 | 330.1 |

The largest single idle slice in each round has a median of 842.7 µs and a mean
of 1179.0 µs, and it sits in one of two places: the draft chain build seam
(`d_chain` → `d_submit2`, 30 of 70 rounds) or the round tail
(`eval_wall` end → `readout` → `commit`, 39 of 70 rounds). The two extreme
slices are isolated outliers in adjacent rounds 20 and 21 (9278.5 µs in
`eval_wall`, 12 999.3 µs in `d_chain`); removing both leaves leg idle at 1.507%,
so they do not carry the verdict either way.

### The serial K=1 leg, for contrast (pid 21216, 504 rounds analysed)

| Segment | Span (ms) | GPU idle (ms) | Idle % of segment | Share of leg idle |
| --- | ---: | ---: | ---: | ---: |
| seed | 3915.8 | 16.86 | 0.43% | 1.94% |
| rounds | 32845.5 | 634.33 | 1.93% | 72.84% |
| turnaround | 168.2 | 168.20 | **100.00%** | 19.32% |
| **leg total** | **37493.8** | **870.79** | **2.323%** | **100%** |

The serial leg's inter-round turnaround is 100% GPU idle at a median of 322.2 µs
per round. The MTP leg's turnaround is 1.33% idle. The classic cross-round
pipelining target is therefore already harvested on the drafting path by the
existing head prefetch, which is why the MTP leg's turnaround holds only 0.52 ms
of the leg's 242 ms of idle.

### Correction to FINDING 539

`research/e204_leg_budget.py` matches `wall_us` with a non-greedy prefix
(`.*?wall_us=`), so it binds to `eval_wall_us` and misattributes about 2890 ms
of seed graph-build time to a phantom "outside-anchor" pool. Recomputed on this
census leg, the reader's own numbers reproduce both readings side by side:

| Leg split | E204 reader | Corrected |
| --- | ---: | ---: |
| seed | 7.02% | 26.86% |
| anchored rounds | 72.83% | 72.83% |
| outside anchors | 20.15% | **0.31%** |

FINDING 539's hypothesis (b) assumed a large unanchored pool between rounds.
That pool does not exist: 0.31% of the leg, or 44.7 ms of 14 588 ms, lies
outside the anchored rounds and the seed.

| Metric | Baseline | Candidate | Ratio / delta |
| --- | ---: | ---: | ---: |
| serial seconds/token | n/a | 0.07323134550824761 | observation leg |
| MTP seconds/token | n/a | 0.028493175748735666 | observation leg |
| local serial-relative speedup | n/a | 2.5701 | observation leg |
| effective mean draft length | n/a | 6.359 | observation leg |
| accepted draft rate | n/a | 0.877 | observation leg |

Every identity field matched across the compared legs within a session: same
host, chip, memory, worker digest, metallib fingerprint, head digest, fixture,
and token window. The only intended difference across the Stage-0 arms is the
planted stall. The absolute seconds per token above carry the ledger's own cost
(one lock and two clock reads per command buffer), so they are attribution
context, not a timing contrast, and no ranked claim is derived from them.

## Conclusion

- What happened and why: the whole candidate MTP leg was placed on the device
  axis for the first time. The GPU is idle for 1.660% of the leg — 242 ms of
  14 588 ms. Two thirds of the leg is the anchored round region and it is 1.95%
  idle; the charged seed is 26.9% of the leg and only 0.52% idle; the inter-round
  turnaround, the classic pipelining target, is 0.27% of the leg and holds
  0.22% of its idle.
- Evidence for or against the mechanism: **against a large win, and it bounds
  the family.** Even a perfect mechanism that removed every microsecond of
  measured idle would return at most 1.66% of candidate MTP leg time, and the
  cross-round part of that is already harvested by the head prefetch. The
  largest located pool is CPU-side round-tail work — `commit` at 1192.5 µs per
  round, 0.87% of a round — followed by the draft chain build seam at 600.8 µs.
  Both are coherent (median largest slice 842.7 µs per round), so they are
  technically addressable, but their combined ceiling is about 1.3% of leg time
  before any implementation cost.
- Prompt or M5 transfer risk: **high for the exact fraction, low for the shape.**
  This is one public prompt on an M4 Pro. The M5 has a different CPU/GPU
  performance ratio, so the CPU-side seams could occupy a different share of a
  faster GPU's round. The structural conclusions — that verification dominates
  the round and is nearly fully busy, that seed prefill is a quarter of the leg,
  and that the inter-round turnaround is already covered on the drafting path —
  should transfer.
- Smallest useful next action: if the advisor still wants the overlap family
  alive, the one question worth 1.3% is whether the round-tail
  `readout` + `commit` sequence can be overlapped with the next round's draft
  chain build, since those are the two largest coherent pools and they are
  adjacent in the round. Otherwise the census says to spend effort inside
  `verify_graph` and `eval_wall`, which hold 93% of round wall time and are
  already GPU-bound, meaning kernel and scheduling work rather than overlap.
- Recommendation: **close the cross-round overlap and pipelining family as a
  top runtime axis** and keep only the round-tail overlap question as a bounded
  follow-up. The measurement lands in the 1–3% band, so the pre-registered rule
  hands the decision to the advisor rather than promoting the family.
