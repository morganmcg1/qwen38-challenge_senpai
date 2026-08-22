# E112 artifacts

Every file here is a saved run of a checked-in script. Regenerate any of them
with the commands below.

## Frozen board snapshot

`yukon` refreshes its cached board file in place, so a live file cannot support
a reproducible count. Every board number in this directory comes from one
frozen copy:

```text
path in this session   /tmp/e112/board-frozen.json   (NOT committed, 14 MB)
sha256                 a983bb7660eb9f79a4c3e88c2b8bee74e6c4031e91c281f6454f3954aba3f556
rows                   1040
latest createdAt       2026-08-21T22:12:57.825Z
usable rows after collect()  746
```

Point the scripts at a snapshot with `export YUKON_BOARD_JSON=<path>`. A
different snapshot changes the row counts and moves every table slightly.

## Files

| file | command | what it answers |
| --- | --- | --- |
| `rung0-q1-pairs.txt` | `python3 research/e112_rung0.py --mech q1` | Q1 board pairs, pooled effect, per-prompt profile |
| `rung0-q2-pairs.txt` | `python3 research/e112_rung0.py --mech q2` | Q2 board pairs, which killed Q2 on sign and profile |
| `rung1-abba.json` | `research/e112_abba.sh` then `research/e112_contrast.py` | Q1 nine-leg ABBA timing, job `2247e9b7-dfea-4be9-95dd-43d878591ea6` |
| `noise-floor.txt` | `python3 research/board_prompt_instrument.py --noise` | replicate class x time gap x solver, stratified F tests, the corrected floor |
| `provenance.txt` | `python3 research/board_prompt_instrument.py --provenance` | how a byte-identical pair is actually created |
| `validate-canon.txt` | `python3 research/board_prompt_instrument.py --validate-canon` | string-aware stripper against the naive regex |
| `survivors.txt` | `python3 research/e112_survivors.py` | single-path mechanisms that clear the corrected floor |

## W&B runs

| run | id | url |
| --- | --- | --- |
| `e112-rung1-q1-abba` | `t31les9h` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/t31les9h |
| `e112-f1-replicate-floor` | `swqvkfv3` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/swqvkfv3 |
| `e112-f1-mechanism-survivors` | `0nifdq54` | https://wandb.ai/wandb-applied-ai-team/qwen38-mlx-challenge-senpai/runs/0nifdq54 |

Reproduce them with `python3 research/e112_wandb_log.py`. Every table in all
three runs is `harness=local`. Nothing here is a ranked measurement.

## Q1 is closed and its arm switch is deleted from the scored surface

E116 rung 0a deleted `MLX_E112_SKIP_1025_WARM`, its `Self.traceRounds` witness
line and the E112 comment block from
`Sources/MLXFastModel/Qwen36MTPBlockSession.swift`, and restored the plain
`if extK.dim(2) == 1024 {` guard on the kL=1025 SDPA compile warm. Deletion is
the campaign default for a closed arm, so the scored source now carries one
obvious warm path.

`research/e112_abba.sh` and `research/e112_analyse.py` still name the flag.
They no longer run against the current source. Reapply the switch first if a
reopener below ever fires:

```bash
git show 7d3d44e5:Sources/MLXFastModel/Qwen36MTPBlockSession.swift \
  > /tmp/e112-flag-source.swift   # read the deleted arm switch
```

Only two observations reopen Q1. Neither is on the table:

1. A ranked decode window longer than 512 tokens. The seed is 512 tokens, so a
   shorter window never walks the key length to 1024 and the blocks=64 to
   blocks=128 boundary is never crossed inside the timed leg.
2. A schedule change that raises committed context enough that crossing starts
   many rounds earlier than round 65 of 77.

## Timing legs are not gate qualified

`rung1-abba.json` was measured with `MLXFAST_LOCAL_COOL_GATE=0` under the
counterbalanced ABBA exception. Every leg carries
`cool_gate_passed_real_gate=false`, `gate_qualified_for_timing=false` and
`official_or_ranked_score=false`. The numbers are directional causal evidence
inside one session. They are not a gate-qualified result and never an official
score.
