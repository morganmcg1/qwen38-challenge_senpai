"""E157 R0 helper: print the per-prompt receipt rows this round needs.

Read-only. It reads the payload `research/board_per_prompt.py fetch` already
wrote to /tmp/yukon-board/full.json and never calls a Yukon mutation.

    python3 research/e157_receipt_probe.py 75a21a4 0cf1637e
"""
from __future__ import annotations

import json
import sys

CACHE = "/tmp/yukon-board/full.json"
PROMPT_NAMES = {
    "919318e1": "beagle",
    "192fb621": "botany",
    "4b9e88cd": "drama",
    "a2ea8b60": "essays",
    "00142a44": "medicine",
    "c1ec5866": "plutarch",
    "ea82dcb5": "republic",
    "3b10cb4d": "travel",
}


def rows() -> list[dict]:
    with open(CACHE) as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("submissions", "rows", "data", "items"):
            if key in payload:
                return payload[key]
    return payload


def main() -> None:
    prefixes = sys.argv[1:] or ["75a21a4"]
    data = rows()
    print(f"rows={len(data)}")
    sample = data[0]
    print("row keys:", sorted(sample.keys()))
    for prefix in prefixes:
        hits = [r for r in data if str(r.get("id", "")).startswith(prefix)]
        print(f"\n=== {prefix}: {len(hits)} match ===")
        for row in hits:
            keep = {
                k: v
                for k, v in row.items()
                if not isinstance(v, (dict, list)) and k not in ("note",)
            }
            print(json.dumps(keep, indent=1, sort_keys=True)[:1400])
            metrics = row.get("officialMetrics") or {}
            print("officialMetrics keys:", sorted(metrics.keys()))
            per_prompt = metrics.get("per_prompt")
            if isinstance(per_prompt, dict):
                first = sorted(per_prompt)[0]
                print("per_prompt entry example:", first, per_prompt[first])
            elif isinstance(per_prompt, list) and per_prompt:
                print("per_prompt list example:", json.dumps(per_prompt[0])[:600])


if __name__ == "__main__":
    main()
