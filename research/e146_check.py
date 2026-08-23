"""E146 preflight: does the library reproduce the advisor's F226 table?

Run this before any conclusion. It prints the anchor consistency check and the
twenty-row crown cluster anchored on 684821ed, which is the exact table in
advisor feedback F1 on PR #146.
"""

import e146_lib as L

path, rows = L.load()
print("board", path, "scored rows", len(rows))
by = {r.id8: r for r in rows}

print("\n--- anchor consistency, 1760479a against the F219 table ---")
anchor = by["1760479a"]
print("%-9s %9s %9s %8s %8s %9s %9s" %
      ("prompt", "d_board", "d_f219", "R_model", "R_f219", "nondraft", "drafting"))
for p in L.PROMPT_ORDER:
    _, d_f219, r_f219 = L.F219_ANCHOR[p]
    r_model = L.round_count(p, anchor.dlen(p))
    print("%-9s %9.4f %9.4f %8.2f %8.2f %9.2f %9.2f" %
          (p, anchor.dlen(p), d_f219, r_model, r_f219,
           anchor.non_drafting(p), r_model - anchor.non_drafting(p)))

print("\n--- crown draft-length cluster, anchored on 684821ed ---")
crown = by["684821ed"]
key = crown.draft_key()
members = [r for r in rows if r.draft_key() == key]
print("cluster size", len(members))
fits = []
for row in members:
    if row.id8 == crown.id8:
        continue
    fit = L.fit_k(crown, row)
    fits.append((fit["k_us_per_drafting_round"], fit, row))
fits.sort()
print("%10s %9s %7s %9s %-14s %10s %s" %
      ("k us/dr", "cand8%", "resid", "id", "solver", "score", "created"))
for k, fit, row in fits:
    print("%10.1f %+9.4f %7.4f %9s %-14s %10.6f %s" %
          (k, fit["cand_mean8_pct"], fit["residual_sd_pp"], row.id8,
           row.solver[:14], row.score, row.created[11:19]))
