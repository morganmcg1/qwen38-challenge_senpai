#!/usr/bin/env python3
"""Decompose the serial-leg noise that the published score frame carries.

Two frames can score one submission:

  published  raw_p = own_serial_p / own_mtp_p          (the ranked score)
  serialfree raw_p = board_mean_serial_p / own_mtp_p   (candidate leg only)

Write the serial draw of prompt ``p`` in submission ``i`` as

  log own_serial = log S_p + k_i + a_ip

where ``k_i`` is one whole-session speed factor shared by all eight prompts of
that submission and ``a_ip`` is an independent per-prompt draw.  The published
frame cancels ``k_i`` because the same session also times the candidate leg; it
keeps ``a_ip``.  The serial-free frame keeps ``k_i`` and drops ``a_ip``.

So the frames trade one noise source for the other, and which frame is quieter
depends on how the median of eight treats each one.  ``a_ip`` is independent
across prompts, but the median does not average all eight: the per-prompt raw
ratios are ordered almost deterministically by draft length, so the median
selects the same two prompts nearly every time and averages only those two
serial draws.  ``k_i`` is common to all eight prompts and the median cannot
reduce it at all.

This script measures sigma_a and sigma_k from the board, checks how stable the
median pair is, and prints the resulting per-frame serial noise at score level.
"""
import collections
import statistics as st

from board_per_prompt import PROMPT_NAMES, load, vec, serial_means

NAMES = list(PROMPT_NAMES.values())


def main():
    scored = load()
    means = serial_means(scored)

    within = []          # pooled per-prompt residual variance, df 7 each
    session = []         # per-submission mean excess
    median_pairs = collections.Counter()
    rows = 0

    for r in scored:
        v = vec(r)
        if any(n not in v for n in NAMES):
            continue
        rows += 1
        excess = {n: 100.0 * (v[n]["serial_seconds_per_token_mean"] / means[n] - 1.0)
                  for n in NAMES}
        m = st.fmean(excess.values())
        session.append(m)
        within.append((7, st.stdev(excess.values())))

        raws = sorted(NAMES, key=lambda n: v[n]["serial_seconds_per_token_mean"]
                      / v[n]["mtp_seconds_per_token_mean"])
        median_pairs[tuple(sorted(raws[3:5]))] += 1

    sigma_a = (sum(df * s * s for df, s in within)
               / sum(df for df, _ in within)) ** 0.5
    sd_mean = st.stdev(session)
    var_k = sd_mean ** 2 - sigma_a ** 2 / 8.0
    sigma_k = var_k ** 0.5 if var_k > 0 else 0.0

    print("submissions with all eight prompts: %d" % rows)
    print()
    print("sigma_a  per-prompt serial draw noise      %.3f %%  (df %d)"
          % (sigma_a, sum(df for df, _ in within)))
    print("sd of the 8-prompt mean excess            %.3f %%  (df %d)"
          % (sd_mean, rows - 1))
    print("sigma_k  between-session speed factor     %.3f %%"
          % sigma_k)
    print()

    print("which two prompts the median of eight selects:")
    for pair, n in median_pairs.most_common(6):
        print("  %-22s %5d  %5.1f %%"
              % ("+".join(pair), n, 100.0 * n / rows))
    top = median_pairs.most_common(1)[0][1]
    print("  most common pair covers %.1f %% of submissions"
          % (100.0 * top / rows))
    print()

    pub_serial = sigma_a / 2 ** 0.5
    free_serial = sigma_k
    print("serial-leg noise carried into the median-of-8 score:")
    print("  published  sigma_a / sqrt(2)  = %.3f %%" % pub_serial)
    print("  serialfree sigma_k            = %.3f %%" % free_serial)
    print()

    observed_pub = 0.196
    print("using the advisor's published replicate sd %.3f %% "
          "(39 byte-identical pairs):" % observed_pub)
    cand_var = observed_pub ** 2 - pub_serial ** 2
    if cand_var <= 0:
        print("  serial term already exceeds the observed sd; "
              "candidate-leg noise is not identifiable here")
        return
    cand = cand_var ** 0.5
    free = (cand ** 2 + free_serial ** 2) ** 0.5
    print("  candidate-leg noise sigma_b          %.3f %%" % cand)
    print("  implied serialfree replicate sd      %.3f %%" % free)
    print("  serial draw share of published var   %.1f %%"
          % (100.0 * pub_serial ** 2 / observed_pub ** 2))
    print()
    better = "serialfree" if free < observed_pub else "published"
    print("quieter frame: %s" % better)
    print("single-pair detection floor  published %.3f %%   serialfree %.3f %%"
          % (observed_pub * 2 ** 0.5, free * 2 ** 0.5))


if __name__ == "__main__":
    main()
