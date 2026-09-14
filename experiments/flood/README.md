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

## Findings so far (run 2)

- The answer is computed at the token *before* the last one, the `=` or the `is`, and
  it is read there as a number word; the last token's band holds the task rather than
  the value, which is why `ans_rank2` (best of the last two positions) and not
  `ans_rank` is the column to read.
- Arithmetic fails by absence: past tier 5 the answer is nowhere in the band, the
  covert product's first digit is only weakly loaded, occupancy rises and top-1
  stability falls.
- count-letter, and about half of multihop, fail the other way: the correct answer sits
  at band rank 0 or 1 while the model emits something else.
- Output probability alone cannot tell those two apart, which is what `conf_wrong` and
  `knew_cw` are for: a confident wrong answer with the answer in the band is a
  different fault from a confident wrong answer without it.

The two failure modes want different responses. Absence means the model never had the
value, so the fix is decomposition: split the problem and let it compute the pieces.
Answer-in-hand means the value was there and the output stage dropped it, so the fix is
cheap: re-read the band, or re-sample. A router that cannot distinguish them will spend
a decomposition budget on problems that only needed a second look.

## Three candidate signatures of "flood"

The term is not defined in the literature. The suite records enough to test each:

| signature | what it would look like | metrics |
|---|---|---|
| saturation | many partial candidates compete, none wins: band entropy stays high, kurtosis low, top-1 unstable across depth, occupancy high | `band_min_entropy`, `band_max_kurtosis`, `top1_stability`, `occupancy` |
| category collapse | the band fills with the answer's type class (digits, letters) rather than the answer: the list-task flood applied to a problem | `class_share`, `class_distinct` vs `band_min_answer_rank` |
| absence | the answer and its intermediates never load; the model confabulates from the final layers only | `band_min_answer_rank`, `band_min_inter_rank`, `model_answer_p` |

All three are computed per item, so which one tracks correctness is an empirical
question for `analyze.py`, not a choice made here. It also reports `later` (the
answer appears somewhere in the continuation but not as the next token: showed work
rather than wrong), `inter_rank` (the worst-loaded intermediate at the readout
position), and `lead` (how many positions before the last the answer first reaches
band rank 5 or better, the "does the band lead the output" test).

Four columns split the readout by where the answer was and how sure the model was:

| column | definition |
|---|---|
| `ans_rank2` | best answer rank over the last *two* positions, median. The one to read: the answer lands on the `=` or `is`, not on the final token |
| `knew` | fraction of band-scored items with `ans_rank2` <= 3, whether or not the output was right |
| `conf_wrong` | fraction of the group's *wrong* items whose output-layer top-1 probability is >= 0.5: confidently wrong. Blank if the group has no wrong items, or for records written before `model_top1_p` existed |
| `knew_cw` | of those confidently-wrong items, the fraction that had the answer at band rank <= 3. High means the answer-in-hand failure; low means absence |

`kurt` is no longer a table column, to keep the row inside a terminal; it is still in
`results.jsonl`. It moved with neither tier nor outcome in run 2.

## Ladder

`items.py` writes `items.jsonl`, 361 items:

| family | tiers | what rises |
|---|---|---|
| arith | 1-7 | tiers 1-4 mirror the paper's modulation tiers (6 items each); 5-7 need a 2-, 3- or 4-digit product held covertly, reduced mod a small number so the answer stays one digit, 30 items each. A two-shot prefix of the same shape makes the answer the very next token |
| count-letter | 1-4 | word length. 15 words a tier, two items each: the most frequent letter and the rarest letter still in the word, which has to be found rather than noticed. Two-shot prefix as above, prompt ends `is ` so the digit comes next |
| nth-letter | 1-4 | word length and index depth; the same 15 words a tier |
| anagram | 1-5 | word length |
| multihop | 1-4 | number of bridge entities, with intermediates labelled |
| trick | 1 | nothing: one tier of 28 questions whose wrong answer is more available than the right one (Moses' ark, the bat and the ball, the capital of Australia). Raw completions with no shots, so the trap is what a shot-free model reaches for. These are the cleanest test of the answer-in-hand failure: the right answer is a fact the model has |

Answers are single tokens where possible so the lens can name them. `run.py`
re-checks against the real tokenizer and records `answer_single_token`; rank metrics
are skipped for the rest, correctness still counts. Items whose every answer form is
a single letter (all of `nth-letter`, three tier-4 `multihop`) carry
`band_scored: false`: a single letter never loads in the band even when the model is
right, so `ans_rank` and `lead` skip them too.

## Run

The usual way is the dashboard's own `FLOOD SUITE` section: it runs this suite inside
the dashboard process, so it reuses the model already on the card, the page stays
usable between items, and the three visuals fill in live as each item lands. Set the
band, positions and gen there, press `start flood`, and pull the full records from
`/flood/results.jsonl` when it finishes. See `deploy/README.md`.

Headless, from the laptop, with an instance provisioned and its bootstrap finished:

```bash
cd deploy && ./flood.sh                 # copies the suite over, runs it, pulls results, analyzes
./flood.sh -- --band 20 60 --positions 4   # arguments after -- go to run.py
```

It stops the dashboard first (two copies of the model do not fit a 48GB card) and
restarts it afterwards. Results land in
`experiments/flood/results.jsonl`; re-run `python analyze.py results.jsonl` any time.

Options: `--band LO HI` (default 24 58, the paper's L38-92 on a 0-100 reindex mapped
onto 64 layers), `--positions N` to read out the last N prompt positions (default 6),
`--gen` greedy tokens for correctness (default 12), `--no-mask` to let punctuation
into the top-k. `--tiny` runs the whole pipeline on `tests/tiny.py` on CPU as a smoke test;
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
