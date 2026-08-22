"""Log the F25 register-staircase table and the architecture-conditional
answer to W&B."""

import json

import wandb

TABLE = json.load(open("research/e133-artifacts/f25-staircase-table.json"))

run = wandb.init(
    entity="wandb-applied-ai-team",
    project="qwen38-mlx-challenge-senpai",
    group="e129-f25-register-staircase",
    job_type="compile-census-analysis",
    name="e129-f25-staircase-and-arch-conditional",
    config={
        "harness": "compile_only",
        "gpu_used": False,
        "timing_valid": False,
        "official_or_ranked_score": False,
        "occupancy_rule": "Rule 89 floor law",
        "source_census": TABLE["source_census"],
        "census_base_sha": TABLE["base_sha"],
        "toolchain": TABLE["toolchain"],
        "register_budget": TABLE["register_budget"],
        "register_ceiling": TABLE["register_ceiling"],
        "table_selection_architecture_conditional": False,
        "table_selection_source_line": "Qwen35.swift:1811",
        "table_selection_only_override": "MLX_E120_QMV_TABLE (Qwen35.swift:1873)",
    },
)

cols = [
    "arch", "na", "form", "air_total", "text_bytes", "delta_text_bytes",
    "registers", "delta_registers", "resident_simdgroups",
    "delta_resident_simdgroups", "registers_needed_for_next_step",
    "reaches_next_step", "spill_bytes", "clamped_at_ceiling",
]
tbl = wandb.Table(columns=cols)
for arch, block in TABLE["arches"].items():
    for r in block["rows"]:
        tbl.add_data(arch, *[r[c] for c in cols[1:]])

run.log({
    "f25/staircase_table": tbl,
    "f25/ranked_g17s_any_form_crosses": int(
        TABLE["arches"]["applegpu_g17s"]["any_form_crosses"]),
    "f25/local_g16s_any_form_crosses": int(
        TABLE["arches"]["applegpu_g16s"]["any_form_crosses"]),
    "f25/ranked_g17s_weighted_occupancy_gain_pct":
        TABLE["arches"]["applegpu_g17s"]["ranked_weighted_occupancy_gain_pct"],
    "f25/local_g16s_weighted_occupancy_gain_pct":
        TABLE["arches"]["applegpu_g16s"]["ranked_weighted_occupancy_gain_pct"],
    "f25/transfer_hazard": int(TABLE["verdict"]["transfer_hazard"]),
})

run.summary.update({
    "verdict": TABLE["verdict"],
    "conclusion": (
        "No W1/W2/A1 form crosses a residency step on the ranked g17s at "
        "NA 5, 6 or 7, so under FINDING 169 the whole family prices at "
        "exactly zero. W2 and W1+W2+A1 do cross at NA=6 on the local g16s, "
        "so a local timing arm would report a gain the ranked host cannot "
        "have."
    ),
})
print("run", run.id, run.url)
run.finish()
