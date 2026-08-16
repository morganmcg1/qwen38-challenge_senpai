"""Decompose the E6/E7 draft-bits result into readout-time vs round-count effects.

Answers two questions the E9 assignment asks:
  1. Is the advisor's break-even acceptance floor derived correctly?
  2. How much of the measured 3-bit win can the readout kernel actually explain?
"""

# English public fixture, 512 decode tokens, one host, back-to-back arms.
ARMS = {
    4: dict(spt_ms=35.119320498779416, acc=0.8902691511387164, rounds=82, readouts=483),
    3: dict(spt_ms=34.45143951103091, acc=0.9094736842105263, rounds=81, readouts=475),
    2: dict(spt_ms=34.61940819397569, acc=0.8902691511387164, rounds=82, readouts=483),
}
TOKENS = 512
KERNEL_SPEEDUP = {3: 0.242, 2: 0.424}  # measured QMV sweep, flat GB/s

for a in ARMS.values():
    a["ms_per_round"] = a["spt_ms"] * TOKENS / a["rounds"]
    a["tpr"] = TOKENS / a["rounds"]
    a["D"] = a["readouts"] / a["rounds"]

print("arm   ms/round   tok/round   D(drafts/round)   1+a*D")
for b, a in sorted(ARMS.items(), reverse=True):
    print(
        f"{b}b  {a['ms_per_round']:9.4f}  {a['tpr']:9.5f}  {a['D']:15.5f}  "
        f"{1 + a['acc'] * a['D']:7.5f}"
    )

ctl = ARMS[4]
print("\n-- where the 3-bit win came from --")
t = ARMS[3]
print(f"total ms/token      : {(t['spt_ms'] / ctl['spt_ms'] - 1) * 100:+.3f}%")
print(f"  per-round time    : {(t['ms_per_round'] / ctl['ms_per_round'] - 1) * 100:+.3f}%")
print(f"  round count       : {(t['rounds'] / ctl['rounds'] - 1) * 100:+.3f}%  (82 -> 81)")

# Per-round time carries the readout saving; solve for the 4-bit readout cost.
den = ctl["D"] - t["D"] * (1 - KERNEL_SPEEDUP[3])
r4 = (ctl["ms_per_round"] - t["ms_per_round"]) / den
share = ctl["D"] * r4 / ctl["ms_per_round"]
print(f"\nimplied 4-bit readout cost   : {r4:.4f} ms")
print(f"readout share of round time  : {share * 100:.2f}%")
print(f"max win if head were FREE    : {-share * 100:.2f}% ms/token")

print("\n-- break-even acceptance (readout saving only, no acceptance change) --")
for b, advisor_floor in ((3, 0.88224), (2, 0.87620)):
    arm = ARMS[b]
    saved = (ctl["D"] - arm["D"] * (1 - KERNEL_SPEEDUP[b])) * r4
    tpr_be = (ctl["ms_per_round"] - saved) * ctl["tpr"] / ctl["ms_per_round"]
    a_be = (tpr_be - 1) / arm["D"]
    print(
        f"{b}b: mine {a_be:.5f} (drop {ctl['acc'] - a_be:.5f}) | "
        f"advisor {advisor_floor:.5f} (drop {ctl['acc'] - advisor_floor:.5f}) | "
        f"advisor is {'LENIENT' if advisor_floor < a_be else 'STRICT'} by "
        f"{abs(a_be - advisor_floor):.5f}"
    )
