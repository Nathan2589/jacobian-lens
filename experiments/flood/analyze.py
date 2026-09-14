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


def rank2(r):
    """Best answer rank over the last two read positions. The answer is computed
    at the token before the last (the '=' or 'is'), and the last position is
    often a formatting token whose band holds the task, not the answer."""
    ranks = [q["band_min_answer_rank"] for q in r["positions"][-2:] if q["band_min_answer_rank"] is not None]
    return min(ranks) if ranks else None


def wrong_top1_p(rs):
    """Output-layer top-1 probability of every wrong item in the group. Records
    written before run.py recorded it contribute nothing, so the columns built
    on it come out blank rather than wrong."""
    return [v for v in (p(r).get("model_top1_p") for r in rs if not r["correct"]) if v is not None]


def confident_wrong(r):
    v = p(r).get("model_top1_p")
    return not r["correct"] and v is not None and v >= 0.5


COLS = [
    ("n", lambda rs: len(rs)),
    ("acc", lambda rs: mean(r["correct"] for r in rs)),
    ("later", lambda rs: mean(r["answer_later"] for r in rs)),
    ("ans_rank", lambda rs: median(
        p(r)["band_min_answer_rank"] for r in rs if scored(r) and p(r)["band_min_answer_rank"] is not None)),
    ("ans_rank2", lambda rs: median(rank2(r) for r in rs if scored(r) and rank2(r) is not None)),
    ("knew", lambda rs: mean(rank2(r) <= 3 for r in rs if scored(r) and rank2(r) is not None)),
    ("conf_wrong", lambda rs: mean(v >= 0.5 for v in wrong_top1_p(rs))),
    ("knew_cw", lambda rs: mean(
        rank2(r) <= 3 for r in rs if confident_wrong(r) and scored(r) and rank2(r) is not None)),
    ("inter_rank", inter_rank),
    ("lead", lead),
    ("model_p", lambda rs: mean(p(r)["model_answer_p"] for r in rs if p(r)["model_answer_p"] is not None)),
    ("entropy", lambda rs: mean(p(r)["band_min_entropy"] for r in rs)),
    # kurt dropped from the table to make room; still in results.jsonl. It never
    # separated tiers or outcomes in run 2.
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
