"""Aggregate results.jsonl by family and tier, and by correct vs incorrect.

    python analyze.py results.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import StatisticsError, mean, median

COLS = [
    ("n", lambda rs: len(rs)),
    ("acc", lambda rs: mean(r["correct"] for r in rs)),
    ("ans_rank", lambda rs: median(p(r)["band_min_answer_rank"] for r in rs if p(r)["band_min_answer_rank"] is not None)),
    ("model_p", lambda rs: mean(p(r)["model_answer_p"] for r in rs if p(r)["model_answer_p"] is not None)),
    ("entropy", lambda rs: mean(p(r)["band_min_entropy"] for r in rs)),
    ("kurt", lambda rs: mean(p(r)["band_max_kurtosis"] for r in rs)),
    ("occ", lambda rs: mean(p(r)["occupancy"] for r in rs)),
    ("stab", lambda rs: mean(p(r)["top1_stability"] for r in rs)),
    ("cls_share", lambda rs: mean(p(r)["class_share"] for r in rs if p(r)["class_share"] is not None)),
    ("cls_n", lambda rs: mean(p(r)["class_distinct"] for r in rs if p(r)["class_distinct"] is not None)),
]


def p(r):  # metrics at the readout (last) position
    return r["positions"][-1]


def fmt(v):
    if v is None:
        return "-"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def table(title, groups):
    print(f"\n{title}")
    print(f"{'group':<22}" + "".join(f"{c:>10}" for c, _ in COLS))
    for key, rs in groups:
        vals = []
        for _, fn in COLS:
            try:
                vals.append(fn(rs))
            except (StatisticsError, ValueError):
                vals.append(None)
        print(f"{key:<22}" + "".join(f"{fmt(v):>10}" for v in vals))


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_tier = defaultdict(list)
    by_ok = defaultdict(list)
    for r in rows:
        by_tier[(r["family"], r["tier"])].append(r)
        by_ok[(r["family"], "correct" if r["correct"] else "wrong")].append(r)
    table("by family x tier", [(f"{f} t{t}", rs) for (f, t), rs in sorted(by_tier.items())])
    table("by family x correctness", [(f"{f} {ok}", rs) for (f, ok), rs in sorted(by_ok.items())])
    multi = [r for r in rows if not r["answer_single_token"]]
    if multi:
        print(f"\n{len(multi)} items whose answer is not a single token (rank metrics skipped): "
              + ", ".join(r["id"] for r in multi))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")
