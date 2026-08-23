"""Prove research/e147_rungA_report.py cannot collect the Rule 137 warmup leg,
and prove the same call DOES collect real timed legs (Rule 101 positive control).
"""
import importlib.util
import json
import pathlib
import shutil
import tempfile

spec = importlib.util.spec_from_file_location("rep", "research/e147_rungA_report.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

META_BASE = {
    "gpu_temp_entry_c": "41.0",
    "gpu_temp_exit_c": "52.0",
    "timing_valid": "true",
    "cool_gate_passed_real_gate": "false",
    "gate_qualified_for_timing": "false",
    "e147_session_commit": "deadbeef",
    "e147_base_ref": "bcc11dc6",
    "golden_sha256": "aa",
    "chip": "applegpu_g16s",
    "host": "h",
    "memory_gib": "48",
}


def leg(root, slot, prompt, extra):
    d = pathlib.Path(root) / slot / prompt
    d.mkdir(parents=True)
    meta = dict(META_BASE)
    meta.update(extra)
    (d / "meta.txt").write_text("".join(f"{k}={v}\n" for k, v in meta.items()))
    (d / "report.json").write_text(
        json.dumps(
            {
                "all_tokens_matched": True,
                "residual_divergence_count": 0,
                "decode_token_count": 512,
                "seed_token_count": 512,
                "seed_prefill_seconds": 4.0,
                "decode_seconds": 16.0,
            }
        )
    )


def timed(rep_i, pos, arm="base"):
    return {
        "e147_witness": "ok",
        "e147_arm_requested": arm,
        "e147_replicate": str(rep_i),
        "e147_position": str(pos),
        "e147_arm_worker_sha256_got": "cafe",
    }


root = tempfile.mkdtemp()
try:
    leg(root, "s1k1p1base", "beagle_a", timed(1, 1))
    # exactly what the shell warmup block writes: no e147_witness, slot not s1*
    leg(root, "warmup-s1k1", "beagle_a", {"e147_warmup": "1", "e147_arm_requested": "base"})

    slots = sorted(l["slot"] for l in rep.collect(root, "s1"))
    print("collected:", slots)
    assert slots == ["s1k1p1base"], slots
    print("positive control fired, timed leg collected: True")
    print("warmup excluded:", "warmup-s1k1" not in slots)

    leg(root, "s1k9p4base", "beagle_a", timed(9, 4))
    n = len(rep.collect(root, "s1"))
    print("guard is not a blanket reject, 2 timed legs collected:", n == 2)
    assert n == 2, n

    # isolate each guard: a witness-bearing leg under the warmup slot name is
    # still rejected by the label guard alone.
    leg(root, "warmup-s1k2", "beagle_a", timed(2, 1))
    n2 = len(rep.collect(root, "s1"))
    print("label guard alone rejects a witness-bearing warmup slot:", n2 == 2)
    assert n2 == 2, n2
    print("RESULT: warmup leg is excluded by two independent guards")
finally:
    shutil.rmtree(root)
