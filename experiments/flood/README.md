# flood: workspace behaviour past the model's ability

A suite that walks several task families up a difficulty ladder, from trivially
solvable to unsolvable without chain-of-thought, and records what the J-space looks
like at each rung. The question is whether the workspace shows a readable signature
when a problem is beyond the model, before the wrong answer is emitted. For Staxis
that is the candidate gating signal: a cheap read that says "this model is out of
its depth on this input, route or recurse".

## What the paper already established (Gurnee et al. 2026, arXiv 2607.15495)

- Occupancy: about 25 J-lens vectors are meaningfully active at once, carrying under
  10% of activation variance (§4.2, Fig 30).
- Functional capacity, list task (§4.2, Fig 31): with 80 unrelated words, only about
  six of the words read so far are present in the band at any comma, flat as the list
  grows; at a single layer it is one to two (§A.16). With related words, "nearly the
  entire 80-word family is present within the first few items, including those that
  have not even been read yet". The authors read this as the model holding the
  category, not the items. A category switch evicts the old block within a few words.
- Competition (§A.17): two held concepts co-occupy tokens at chance, essentially free.
  A concept plus an arithmetic problem are token-exclusive, and the computation bears
  the cost: answer reachability drops from 95% to 72% under dual load. "Some notion of
  task difficulty or cognitive load may influence whether different concepts' presence
  in the J-space is mutually exclusive."
- Failure cases: swap failures concentrate where the source concept was weakly loaded
  (§3.4); number words load weakly; the line-width counting task is "harder both to
  perform and to specify" and precision spans 14% to 56% (§A.10). In the answer-
  thrashing trace the lens shows the computed answer prepared at the positions just
  before the wrong output (§A.19).
- Not in the paper: any experiment that pushes a task past the model's ability and
  reports what the band looks like on the far side. Dehaene and Naccache's commentary
  asks for exactly the dual-task and threshold tests, and argues the 25 may be
  inflated because the vectors are redundant facets of one "state of mind".

## Three candidate signatures of "flood"

The term is not defined in the literature. The suite records enough to test each:

| signature | what it would look like | metrics |
|---|---|---|
| saturation | many partial candidates compete, none wins: band entropy stays high, kurtosis low, top-1 unstable across depth, occupancy high | `band_min_entropy`, `band_max_kurtosis`, `top1_stability`, `occupancy` |
| category collapse | the band fills with the answer's type class (digits, letters) rather than the answer: the list-task flood applied to a problem | `class_share`, `class_distinct` vs `band_min_answer_rank` |
| absence | the answer and its intermediates never load; the model confabulates from the final layers only | `band_min_answer_rank`, `band_min_inter_rank`, `model_answer_p` |

All three are computed per item, so which one tracks correctness is an empirical
question for `analyze.py`, not a choice made here.

## Ladder

`items.py` writes `items.jsonl`, 129 items:

| family | tiers | what rises |
|---|---|---|
| arith | 1-7 | tiers 1-4 mirror the paper's modulation tiers; 5-7 need a 2-, 3- or 4-digit product held covertly, reduced mod a small number so the answer stays one digit |
| count-letter | 1-4 | word length; count of the most frequent letter |
| nth-letter | 1-4 | word length and index depth |
| anagram | 1-5 | word length |
| multihop | 1-4 | number of bridge entities, with intermediates labelled |

Answers are single tokens where possible so the lens can name them. `run.py`
re-checks against the real tokenizer and records `answer_single_token`; rank metrics
are skipped for the rest, correctness still counts.

## Run

From the laptop, with an instance provisioned and its bootstrap finished:

```bash
cd deploy && ./flood.sh                 # copies the suite over, runs it, pulls results, analyzes
./flood.sh -- --band 20 60 --positions 4   # arguments after -- go to run.py
```

It stops the dashboard first (two copies of the model do not fit a 48GB card) and
restarts it afterwards. Results land in `experiments/flood/results.jsonl`; re-run
`python analyze.py results.jsonl` any time.

Options: `--band LO HI` (default 24 58, the paper's L38-92 on a 0-100 reindex mapped
onto 64 layers), `--positions N` to read out the last N prompt positions instead of
one, `--gen` greedy tokens for correctness, `--no-mask` to let punctuation into the
top-k. `--tiny` runs the whole pipeline on `tests/tiny.py` on CPU as a smoke test;
its numbers mean nothing.

Expect a few seconds per item on an A6000 at NF4.

## Caveats

- Same drift caveat as the dashboard: bf16-fitted lens on NF4 weights. Compare
  metrics across tiers within a family, not absolute values.
- The band is a guess mapped from a different model's CKA blocks. If the tier curves
  are flat, sweep it before concluding anything.
- Mod-reduced arithmetic has shortcuts (mod 9 is a digit sum). Tier order is still
  monotone in the size of the covert product.
- The multihop intermediates are hand-authored; check the facts before trusting a
  band-min intermediate rank.
