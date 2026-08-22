#!/usr/bin/env python3
"""E128-F7 item 4: re-derive the head-carrying terms of the ranked cost curve
after the F13 retraction.

harness=ranked. Zero GPU.

The proposal head runs once per proposed draft, so a round of M rows carries
M - 1 head steps. In the fitted curve `round_us(M) = a + k*M` that is

    head(M) = h*(M - 1) = -h + h*M

so the head sits entirely in the per-row slope `k`, with a compensating `-h`
in the intercept. `h` and the target per-row cost are PERFECTLY collinear in
slope space, so `h` cannot be fitted from the curve; it has to be pinned from
outside. F13 pinned it at 1.82 % of the round. The corrected share is 7 % to
9 %.
"""

import json

# Advisor's frame, which reproduces both of their published numbers.
ROUND_US = 55645.4          # F83-weighted ranked round
STEPS = 4.381818181818182   # beagle draft length, the F13 divisor
BYTES_PER_STEP_MB = 323.59  # reconciled per-draft-step byte budget
RANKED_M1_RATE = 462.2      # GB/s, ranked M=1

# Headline curve, e128-artifacts/f4-candidate-curves.json -> slopeonly_b6.
K_LO, K_HI = 3446.0718068476417, 5323.531364694667
A_LO = 27725.39691958033

SHARES = {
    "F13 retracted": 0.0182,
    "byte/rate at 462.2 GB/s": 0.0551,
    "corrected low 7 %": 0.07,
    "rung-0b via F35": 0.071,
    "E79 anchor": 0.084,
    "corrected high 9 %": 0.09,
    "E82 draft_build local": 0.1006,
}


def main():
    print("harness=ranked  E128-F7 item 4 - head-carrying terms re-derived")
    print("round %.1f us   steps %.4f   bytes/step %.2f MB" %
          (ROUND_US, STEPS, BYTES_PER_STEP_MB))
    print("curve slopeonly_b6   a %.1f   k_lo %.1f   k_hi %.1f   raw ratio %.4f\n"
          % (A_LO, K_LO, K_HI, K_HI / K_LO))

    print("%-26s%8s%11s%11s%12s%11s%11s%9s"
          % ("head share of round", "%", "us/round", "h us/step", "rate GB/s",
             "k_lo-h", "k_hi-h", "ratio"))
    out = {}
    for label, s in sorted(SHARES.items(), key=lambda kv: kv[1]):
        per_round = s * ROUND_US
        h = per_round / STEPS
        rate = BYTES_PER_STEP_MB * 1e6 / (h * 1e-6) / 1e9
        lo, hi = K_LO - h, K_HI - h
        print("%-26s%8.2f%11.1f%11.1f%12.1f%11.1f%11.1f%9.4f"
              % (label, 100 * s, per_round, h, rate, lo, hi, hi / lo))
        out[label] = {
            "share": s, "us_per_round": per_round, "h_us_per_step": h,
            "implied_rate_gbs": rate, "target_k_lo": lo, "target_k_hi": hi,
            "target_ratio": hi / lo,
        }

    f13 = out["F13 retracted"]
    lo7, hi9 = out["corrected low 7 %"], out["corrected high 9 %"]
    print("\nreassignment caused by the retraction")
    print("  head per row moves %.1f -> %.1f .. %.1f us  (%.2fx to %.2fx)"
          % (f13["h_us_per_step"], lo7["h_us_per_step"], hi9["h_us_per_step"],
             lo7["h_us_per_step"] / f13["h_us_per_step"],
             hi9["h_us_per_step"] / f13["h_us_per_step"]))
    d7 = lo7["h_us_per_step"] - f13["h_us_per_step"]
    d9 = hi9["h_us_per_step"] - f13["h_us_per_step"]
    print("  %.1f to %.1f us/row moves from the target term to the head term"
          % (d7, d9))
    print("  that is %.1f %% to %.1f %% of k_lo and %.1f %% to %.1f %% of k_hi"
          % (100 * d7 / K_LO, 100 * d9 / K_LO, 100 * d7 / K_HI, 100 * d9 / K_HI))
    print("\ntarget-side slope ratio, the quantity the 1.82x question is about")
    print("  fitted end-to-end          %.4f" % (K_HI / K_LO))
    print("  net of the F13 head        %.4f" % f13["target_ratio"])
    print("  net of a 7 %% head          %.4f" % lo7["target_ratio"])
    print("  net of a 9 %% head          %.4f" % hi9["target_ratio"])
    print("  a bigger head RAISES the residual target ratio, it does not")
    print("  absorb it, because the head chain costs the same per step at")
    print("  every width and a constant subtracted from both slopes moves")
    print("  their ratio away from one.")

    print("\nwidth-independence check: head share of the MARGINAL row")
    print("  low segment  %.1f %% to %.1f %%" % (100 * lo7["h_us_per_step"] / K_LO,
                                                 100 * hi9["h_us_per_step"] / K_LO))
    print("  high segment %.1f %% to %.1f %%" % (100 * lo7["h_us_per_step"] / K_HI,
                                                 100 * hi9["h_us_per_step"] / K_HI))
    print("\nsanity: implied streaming rate must sit under the ranked M=1 rate")
    print("  ranked M=1 rate %.1f GB/s" % RANKED_M1_RATE)
    for label in ("F13 retracted", "corrected low 7 %", "corrected high 9 %"):
        r = out[label]["implied_rate_gbs"]
        print("  %-24s %8.1f GB/s   %s" % (label, r,
              "IMPOSSIBLE" if r > RANKED_M1_RATE else "plausible"))

    json.dump(out, open("e128-artifacts/f7-head-share.json", "w"), indent=2)
    print("\nwrote e128-artifacts/f7-head-share.json")


if __name__ == "__main__":
    main()
