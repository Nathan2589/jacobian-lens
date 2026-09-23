# Model branches

**Policy: one long-lived git branch per served model. `main` holds the shared code and
the reference config; a model branch holds nothing but the delta needed to serve that
model.**

Switching the dashboard to another model is `git checkout model/<slug>`, and the
provisioned instance is told which branch to clone. Nothing else changes.

## Why a branch and not a config file

Three of the model-specific values are already environment variables
(`JLENS_MODEL_ID`, `JLENS_LENS_REPO`, `JLENS_LENS_FILE`), and it is tempting to stop
there. It does not hold, because the rest of what a model changes is not a value —
it is code, sizing tables, and prose that becomes wrong:

| what changes per model | where | why an env var cannot carry it |
|---|---|---|
| VRAM floor per precision | `bootstrap.sh:65-69` | a `case` block of literals sized for a 27B model |
| offer filter (`gpu_ram`, `dph_total`, `disk_space`) | `provision.sh:123`, `DISK_GB:15`, `MIN_RAM_MB:19` | a 12B model rents a different, cheaper class of card |
| flood band defaults `24..58` | `dashboard.py:408-409` and the form at `575-576` | the fitted layer set differs in range *and* in density |
| "the final row, **L63**, is the true output" | `README.md:22,49`, `ARCHITECTURE.md:155-165`, `dashboard.py:197,607-611` | the layer index is `n_layers - 1`, and it is written into prose and UI comments |
| download sizes quoted to the operator | `README.md:27-28,53`, `provision.sh:313`, `tunnel.sh:91` | 56GB/3.3GB are Qwen's; quoting them for a 12B model misleads |
| cost table and runway | `README.md` Cost section | follows from the card the offer filter selects |
| residual-stack layout, if unusual | `jlens/hf.py:_LAYOUTS` | a new layout entry is code |

A model branch makes that one coherent, reviewable diff instead of a scatter of
overrides that are individually plausible and jointly wrong.

**The env vars stay.** They remain the runtime override for a live instance — re-running
`bootstrap.sh` with `JLENS_PRECISION=int8` is still the way to change precision without
redeploying. The branch sets their *defaults*; it does not replace them.

## The rule that stops branches rotting

This is the whole cost of the policy, and ignoring it is how branch-per-model turns
into N forks that no longer share a dashboard.

- **Shared code changes land on `main` only.** A fix to `dashboard.py`'s rendering, to
  `jlens/`, to `tunnel.sh`, to the flood suite: `main`, always. Never on a model branch,
  never cherry-picked sideways between model branches.
- **Model branches rebase onto `main`. They never merge into it.**
  `git rebase main` keeps the branch a clean one-commit-ish delta you can read in a
  single `git diff main...`. A merge commit buries it.
- **If a model branch needs a shared change to work, that change goes to `main` first,**
  then the branch rebases onto it. Two steps, in that order, every time.
- **A model branch that will not rebase cleanly is telling you something.** Usually that
  a constant should have become a variable on `main`. Fix it on `main`; do not resolve
  the conflict on the branch.

The health check, run on any model branch:

```bash
git fetch origin
git rebase origin/main
git diff --stat origin/main...HEAD      # should be deploy/ config + docs, and nothing else
```

If that diff ever lists `jlens/`, `experiments/`, or a behavioural change to
`dashboard.py`, the branch has absorbed shared work. Move it to `main`.

## What a model branch may touch

Allowed:

- `deploy/bootstrap.sh` — the three `JLENS_*` defaults and the `MIN_VRAM_GB` case block
- `deploy/dashboard.py` — the three `MODEL_ID`/`LENS_REPO`/`LENS_FILE` defaults, the
  `band_lo`/`band_hi` defaults, `EXAMPLES`, and the `L<n>` references in comments/UI text
- `deploy/provision.sh` — `DISK_GB`, `MIN_RAM_MB`, the offer `QUERY`, the size quoted at
  line 313
- `deploy/README.md`, `ARCHITECTURE.md` — the model name, the layer index, the sizes,
  the cost table
- `deploy/tunnel.sh` — the download size in the comment at line 91

Forbidden, no exceptions:

- `jlens/` — the library is model-agnostic by design. A model that needs a new residual
  layout adds a `Layout(...)` to `_LAYOUTS` **on `main`**; that entry is inert for every
  other model, so it costs nothing to carry.
- `experiments/`, `tests/` — shared. A model-specific flood item set would be a new
  items file on `main`, selected by env var, not a branch-local edit.
- `pyproject.toml`, `uv.lock` — shared.

## Prerequisite: `provision.sh` must be able to clone a branch

**Not yet true. Do this before the first model branch is provisioned.** Today
`provision.sh` pins the default branch in three places, so a fresh instance always gets
`main` no matter which branch you have checked out:

| line | today | needs to become |
|---|---|---|
| 102 | `RAW_BASE=".../main/deploy"` | `.../$BRANCH/deploy` |
| 189 | `git clone --depth 1 $REPO_URL` | `git clone --depth 1 --branch $BRANCH $REPO_URL` |
| 191 | `.../heads/main.tar.gz` | `.../heads/$BRANCH.tar.gz` |

with `BRANCH="${JLENS_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"` near the top, so the
checked-out branch is the default and `JLENS_BRANCH` overrides it. The staleness check at
line 107 then compares against the right branch instead of always against `main`.

Until that lands, the only way to reach a model branch on an instance is after the fact:

```bash
./apply.sh --pull      # then, on the instance: git checkout model/<slug> && re-run bootstrap
```

which works, but burns the download twice if the branch changes the model id.

## Adding a model

0. Confirm the prerequisite above is in place.
1. **Find a published lens, or stop.** Fitting one for a 27B model is ~24 H100-hours;
   this is outside the budget and the reason `main` serves Qwen rather than a preference.
   Search `https://huggingface.co/api/models?search=jacobian-lens`. Read the lens's
   `.meta.json`: it gives `d_model`, `n_layers`, `source_layers`, `target_layer`, the
   dtype it was fitted in, and the exact model revision it was fitted against.
2. **Check the lens's `source_layers` against the flood band defaults.** They may be
   sparse or stop well short of the last layer. Defaults that clamp to an empty band give
   the operator a 400 with no explanation of why.
3. `git checkout -b model/<hf-slug> main` — slug lowercase, matching the HF repo name.
4. Edit only the files in the allow-list above.
5. Recompute `MIN_VRAM_GB` from the actual parameter count. `bf16 ≈ 2 bytes/param`,
   `int8 ≈ 1`, `nf4 ≈ 0.56` (4 bits plus quantisation constants). Add the lens resident
   size — roughly **twice** its file size, since it is upcast on load — then keep `main`'s
   headroom for transients (~7GB for nf4/int8, ~9GB for bf16).
6. Re-derive the offer filter. A smaller model is the point at which cheaper cards become
   eligible, and the filter is what decides whether you ever see them.
7. Run the verification below **before** provisioning. Every mistake in this list costs
   real money at the offer rate, and most of them only surface after a 15-35 minute
   download.
8. Open a PR against `main` for the *shared* half (a new `Layout`, a constant that had to
   become a variable), and keep the branch itself unmerged. Model branches are never
   merged; they live as branches.

## Verification before you spend anything

```bash
# 1. the model repo resolves, is not gated, and has safetensors
curl -s "https://huggingface.co/api/models/$JLENS_MODEL_ID" \
  | jq '{gated, params: .safetensors.total}'

# 2. the lens file exists at the path you wrote down (nested paths are fine)
curl -sIL "https://huggingface.co/$JLENS_LENS_REPO/resolve/main/$JLENS_LENS_FILE" \
  | grep -i '^x-linked-size'

# 3. the lens metadata agrees with the model config
curl -sL "https://huggingface.co/$JLENS_LENS_REPO/resolve/main/$JLENS_LENS_FILE.meta.json" \
  | jq '.model, .source_layers, .target_layer'
curl -sL "https://huggingface.co/$JLENS_MODEL_ID/resolve/main/config.json" \
  | jq '.text_config | {num_hidden_layers, hidden_size, final_logit_softcapping}'
```

`d_model` and `n_layers` must match exactly. A lens fitted against a different revision
of the same model id is usable but is drift on top of quantisation drift — note it in the
branch's README edit rather than leaving it for the next person to discover.

The CPU smoke test exercises every route with no GPU and no weights:

```bash
JLENS_TINY=1 python -m uvicorn dashboard:app --host 127.0.0.1 --port 7860
```

It does not validate your model config — it swaps in a toy decoder — so it catches
syntax and routing breakage only. It is still worth the thirty seconds.

## Worked example: `model/gemma-4-12b-it`

Verified against the HF API on 2026-09-23.

| | `main` (reference) | `model/gemma-4-12b-it` |
|---|---|---|
| `JLENS_MODEL_ID` | `Qwen/Qwen3.8-27B` | `google/gemma-4-12B-it` |
| `JLENS_LENS_REPO` | `eyes-ml/Qwen3.8-27B_jacobian-lens` | `lamm-mit/gemma4-jacobian-lenses` |
| `JLENS_LENS_FILE` | `Qwen3.8-27B_jacobian_lens.pt` | `gemma4-12b-it/paper/seed0.pt` |
| architecture | `Qwen3ForCausalLM` | `Gemma4UnifiedForConditionalGeneration` |
| layers / `d_model` | 64 / 5120 | 48 / 3840 |
| true-output row | L63 | **L47** |
| lens fitted layers | 63, contiguous | **25, sparse: 0,2,…,45; target 46** |
| lens file | 3.3GB | 703MiB |
| bf16 weights | ~56GB | ~24GB (11,959,730,224 params) |
| `MIN_VRAM_GB` nf4 / int8 / bf16 | 32 / 44 / 72 | ~16 / ~21 / ~36 |
| gated on HF | no | no |

Three things this example makes concrete, all of which would have been invisible as
env-var overrides:

- **The band defaults break.** `dashboard.py` defaults to `band_lo=24, band_hi=58`, but
  this lens's fitted layers stop at 45. `58` clamps down to `45`, which happens to be
  survivable — but the band is 25 sparse layers spanning 0-45, not a dense 24-58 window,
  so the middle visual means something different and the defaults should be re-chosen.
- **`nf4` on a 12B model needs ~16GB, not 32GB.** The offer filter's `gpu_ram >= 45` and
  `dph_total < 0.70` exclude every card this model could actually run on. Re-deriving the
  filter is most of the value of the branch.
- **Softcapping is already handled.** `text_config.final_logit_softcapping` is `30.0`;
  `HFLensModel` reads it and applies the `tanh` cap in `unembed()`. No change needed —
  worth recording so nobody re-derives it.

One item is **not** verified and must be checked on first load: which `_LAYOUTS` entry
resolves for `Gemma4UnifiedForConditionalGeneration`. `Layout("model")` is tried first
and is expected to fall through to `Layout("model.language_model")`, since the
multimodal wrapper at `.model` carries `.language_model` rather than `.layers`. If
`_find_layout` raises, the fix is a new `Layout(...)` entry **on `main`**, not on the
branch. The dashboard prints the resolved layer count and `d_model` in its status line;
`48 layers · d_model 3840` confirms it.

## Retiring a branch

Delete it. There is no state on a model branch worth preserving once you stop serving
that model — the config is reconstructible from the lens metadata in ten minutes, and a
stale branch that has not been rebased in months is worse than no branch, because it
looks usable. If the model is one you expect to return to, leave a row in the table
above instead of leaving the branch.
