#!/usr/bin/env python3
"""Self-test for research/e85_abba_stats.py against a known synthetic effect."""

import json
import random
import subprocess
import sys
from pathlib import Path

TRUE_EFFECT_US = -20.0
DRIFT_US_PER_LEG = 20.0

root = Path("/tmp/e85stattest")
root.mkdir(parents=True, exist_ok=True)

random.seed(7)
header = (
    "leg\tarm\tfused_embed\tgather_qmm\tmtp_s_per_tok\tserial_s_per_tok\tratio\t"
    "mean_draft_len\taccepted_rate\tmatched\ttemp_in\ttemp_out\tseconds"
)
lines = [header]
leg = 0
for _ in range(3):
    for arm in ("base", "ab", "ab", "base"):
        leg += 1
        eff = TRUE_EFFECT_US * 1e-6 if arm == "ab" else 0.0
        mtp = 0.0500 + eff + DRIFT_US_PER_LEG * 1e-6 * leg + random.gauss(0, 5e-6)
        ser = 0.1200 + 4e-5 * leg + random.gauss(0, 1e-5)
        flag = 1 if arm == "ab" else 0
        lines.append(
            "\t".join(
                str(v)
                for v in (
                    leg, arm, flag, flag, mtp, ser, ser / mtp, 4.83, 0.955,
                    "True", 41.0 + 0.3 * leg, 42.0 + 0.3 * leg, 140,
                )
            )
        )
(root / "legs.tsv").write_text("\n".join(lines) + "\n")

out = subprocess.run(
    [sys.executable, "research/e85_abba_stats.py", str(root)],
    capture_output=True, text=True, check=True,
)
d = json.loads(out.stdout)

print(f"true effect          {TRUE_EFFECT_US:+.2f} us/token")
print(f"block estimate       {d['block_effect_us_per_token']:+.2f} us/token")
print(f"ols   estimate       {d['ols_effect_us_per_token']:+.2f} us/token")
print(f"ols   t              {d['ols_mtp_s_per_tok']['t']:+.2f}")
print(f"recovered drift/leg  {d['ols_mtp_s_per_tok']['drift_per_leg'] * 1e6:+.2f} us"
      f"  (true {DRIFT_US_PER_LEG:+.2f})")
print(f"drafts per token     {d['drafts_per_token']:.4f}")
print(f"us per draft (ols)   {d['ols_effect_us_per_draft']:+.2f}")
print(f"us per buffer (ols)  {d['ols_effect_us_per_buffer']:+.2f}")
print(f"saving per buffer    {d['saving_us_per_buffer']:+.2f}")
print(f"verdict              {d['verdict']}")
print(f"block values us      {[round(v * 1e6, 2) for v in d['block_mtp_s_per_tok']['values']]}")

ok = (
    abs(d["block_effect_us_per_token"] - TRUE_EFFECT_US) < 5.0
    and abs(d["ols_effect_us_per_token"] - TRUE_EFFECT_US) < 5.0
    and abs(d["ols_mtp_s_per_tok"]["drift_per_leg"] * 1e6 - DRIFT_US_PER_LEG) < 5.0
)
print("SELFTEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
