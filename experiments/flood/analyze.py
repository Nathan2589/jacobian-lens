"""Aggregate results.jsonl by family and tier, and by correct vs incorrect.

    python analyze.py results.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import StatisticsError, mean, median


def p(r):  # metrics at the readout (last) position
    return r["positions"][-1]


def scored(r):  # single-letter answers never load in the band; skip their ranks
    return r.get("band_scored", True)


def inter_rank(rs):
    """Worst-loaded intermediate per item, median over items that have any."""
    worst = []
    for r in rs:
        ranks = [v for v in p(r)["band_min_inter_rank"] if v is not None]
        if ranks:
            worst.append(max(ranks))
    return median(worst)


def lead(rs):
    """Positions before the last at which the answer first reaches band rank <= 5."""
    leads = []
    for r in rs:
        if not scored(r):
            continue
        ps = r["positions"]
        hits = [i for i, q in enumerate(ps) if q["band_min_answer_rank"] is not None and q["band_min_answer_rank"] <= 5]
        if hits:
            leads.append(len(ps) - 1 - hits[0])
    return mean(leads)


COLS = [
    ("n", lambda rs: len(rs)),
    ("acc", lambda rs: mean(r["correct"] for r in rs)),
    ("later", lambda rs: mean(r["answer_later"] for r in rs)),
    ("ans_rank", lambda rs: median(
        p(r)["band_min_answer_rank"] for r in rs if scored(r) and p(r)["band_min_answer_rank"] is not None)),
    ("inter_rank", inter_rank),
    ("lead", lead),
    ("model_p", lambda rs: mean(p(r)["model_answer_p"] for r in rs if p(r)["model_answer_p"] is not None)),
    ("entropy", lambda rs: mean(p(r)["band_min_entropy"] for r in rs)),
    ("kurt", lambda rs: mean(p(r)["band_max_kurtosis"] for r in rs)),
    ("occ", lambda rs: mean(p(r)["occupancy"] for r in rs)),
    ("stab", lambda rs: mean(p(r)["top1_stability"] for r in rs)),
    ("cls_share", lambda rs: mean(p(r)["class_share"] for r in rs if p(r)["class_share"] is not None)),
    ("cls_n", lambda rs: mean(p(r)["class_distinct"] for r in rs if p(r)["class_distinct"] is not None)),
]


def fmt(v):
    if v is None:
        return "-"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def table(title, groups):
    print(f"\n{title}")
    print(f"{'group':<22}" + "".join(f"{c:>11}" for c, _ in COLS))
    for key, rs in groups:
        vals = []
        for _, fn in COLS:
            try:
                vals.append(fn(rs))
            except (StatisticsError, ValueError):
                vals.append(None)
        print(f"{key:<22}" + "".join(f"{fmt(v):>11}" for v in vals))


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_tier = defaultdict(list)
    by_ok = defaultdict(list)
    for r in rows:
        by_tier[(r["family"], r["tier"])].append(r)
        by_ok[(r["family"], "correct" if r["correct"] else "wrong")].append(r)
    table("by family x tier", [(f"{f} t{t}", rs) for (f, t), rs in sorted(by_tier.items())])
    table("by family x correctness", [(f"{f} {ok}", rs) for (f, ok), rs in sorted(by_ok.items())])
    unscored = [r for r in rows if not scored(r)]
    if unscored:
        print(f"\n{len(unscored)} items excluded from ans_rank and lead "
              f"(band_scored false, single-letter answer): " + ", ".join(r["id"] for r in unscored))
    multi = [r for r in rows if not r["answer_single_token"]]
    if multi:
        print(f"\n{len(multi)} items whose answer is not a single token (rank metrics skipped): "
              + ", ".join(r["id"] for r in multi))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")
