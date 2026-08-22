"""Where does the BOARD cost curve actually break?

`rankedcurve.py` imposes the break by splitting the eight prompts into
G1 = {plutarch, drama, travel} and G2 = {beagle, republic, essays, medicine,
botany} and fitting two lines. That split asserts the break rather than
measuring it. This scans the break position freely on the 164
reference-schedule rows, whose round counts are validated.

Reference-schedule mean widths:
  plutarch 1.154  drama 3.298  travel 3.656  beagle 5.382
  republic 5.989  essays 6.087  medicine 6.256  botany 7.148
The sampled widths leave a gap between 3.656 and 5.382, so a break anywhere in
(3.656, 5.382] is observationally identical. The scan says which side of beagle
the break falls on, which is the part that matters.
"""

import math
import statistics

import numpy as np

from e128_rounds import load_rows, per_prompt

TOKENS = 512
REF_BEAGLE = 4.381818181818182
REF_ROUNDS = {
    "plutarch": 487, "drama": 252, "travel": 212, "beagle": 110,
    "republic": 93, "essays": 92, "medicine": 90, "botany": 81,
}
DRAMA_ALT = {"drama": 168}


def ref_points(row, rounds):
    e = per_prompt(row)
    if len(e) != 8:
        return None
    if abs(e["beagle"]["effective_mean_draft_len"] - REF_BEAGLE) > 1e-9:
        return None
    out = {}
    for name, entry in e.items():
        r = rounds[name]
        out[name] = (
            entry["effective_mean_draft_len"] + 1.0,
            TOKENS * entry["mtp_seconds_per_token_mean"] / r * 1e6,
        )
    return out


def hinge_fit(xs, ys, mstar):
    a = np.column_stack(
        [np.ones(len(xs)), xs, np.maximum(0.0, np.asarray(xs) - mstar)]
    )
    beta, *_ = np.linalg.lstsq(a, ys, rcond=None)
    resid = np.asarray(ys) - a @ beta
    return beta, float(resid @ resid)


def main():
    rows = load_rows()
    for tag, rounds in (
        ("rankedcurve ROUNDS (drama 252)", REF_ROUNDS),
        ("minimal-R variant (drama 168)", {**REF_ROUNDS, **DRAMA_ALT}),
    ):
        recs = [p for p in (ref_points(r, rounds) for r in rows) if p]
        print(f"\n=== {tag}: {len(recs)} reference-schedule rows ===")
        names = list(REF_ROUNDS)
        xs = [recs[0][n][0] for n in names]
        print("  mean widths: " + "  ".join(f"{n}={x:.3f}" for n, x in zip(names, xs)))

        grid = np.arange(1.5, 8.01, 0.0625)
        rss_by_break = {float(m): [] for m in grid}
        best_break = []
        for d in recs:
            ys = [d[n][1] for n in names]
            best = None
            for m in grid:
                _, rss = hinge_fit(xs, ys, float(m))
                rss_by_break[float(m)].append(rss)
                if best is None or rss < best[1]:
                    best = (float(m), rss)
            best_break.append(best[0])
        pooled = {m: sum(v) for m, v in rss_by_break.items()}
        mbest = min(pooled, key=pooled.get)
        print(f"  pooled-RSS best break M* = {mbest:.4f}")
        print(f"  per-row best break: median {statistics.median(best_break):.3f}   "
              f"min {min(best_break):.3f}   max {max(best_break):.3f}")

        base = pooled[mbest]
        n_pts = len(recs) * 8
        print(f"  {'M*':>7s}{'pooled RSS':>14s}{'ratio':>9s}{'dAICc(8 pts/row)':>19s}")
        for m in (3.75, 4.0, 4.5, 5.0, 5.25, 5.375, 5.5, 6.0, 6.5, 7.0):
            if m not in pooled:
                m = min(pooled, key=lambda z: abs(z - m))
            d_aic = n_pts * math.log(pooled[m] / base)
            print(f"  {m:7.3f}{pooled[m]:14.4g}{pooled[m]/base:9.4f}{d_aic:19.1f}")

        # Does beagle sit on the line through the four widest prompts?
        hi = ["republic", "essays", "medicine", "botany"]
        devs = []
        for d in recs:
            hx = np.array([d[n][0] for n in hi])
            hy = np.array([d[n][1] for n in hi])
            a = np.column_stack([np.ones(4), hx])
            beta, *_ = np.linalg.lstsq(a, hy, rcond=None)
            pred = beta[0] + beta[1] * d["beagle"][0]
            devs.append(100.0 * (d["beagle"][1] - pred) / pred)
        devs.sort()
        print(
            f"  beagle vs the line through {hi}: "
            f"median {statistics.median(devs):+.2f} %  "
            f"p10 {devs[int(0.1*(len(devs)-1))]:+.2f} %  "
            f"p90 {devs[int(0.9*(len(devs)-1))]:+.2f} %"
        )
        lo = ["plutarch", "drama", "travel"]
        devs2 = []
        for d in recs:
            lx = np.array([d[n][0] for n in lo])
            ly = np.array([d[n][1] for n in lo])
            a = np.column_stack([np.ones(3), lx])
            beta, *_ = np.linalg.lstsq(a, ly, rcond=None)
            pred = beta[0] + beta[1] * d["beagle"][0]
            devs2.append(100.0 * (d["beagle"][1] - pred) / pred)
        devs2.sort()
        print(
            f"  beagle vs the line through {lo}: "
            f"median {statistics.median(devs2):+.2f} %  "
            f"p10 {devs2[int(0.1*(len(devs2)-1))]:+.2f} %  "
            f"p90 {devs2[int(0.9*(len(devs2)-1))]:+.2f} %"
        )


if __name__ == "__main__":
    main()
