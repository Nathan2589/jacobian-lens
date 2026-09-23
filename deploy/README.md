# deploy: J-lens dashboard on a rented GPU

Rent a GPU on vast.ai, serve the Jacobian lens dashboard on it, reach it from your
laptop over an SSH tunnel at `http://localhost:7860`.

Four scripts, in order:

| script | runs on | what it does |
|---|---|---|
| `provision.sh` | laptop | picks an offer, creates the instance, waits for it |
| `bootstrap.sh` | instance | installs deps, downloads model and lens, starts the server |
| `tunnel.sh` | laptop | forwards `localhost:7860` to the instance |
| `destroy.sh` | laptop | kills the instance. This is the only thing that stops billing |
| `flood.sh` | laptop | runs `experiments/flood` on the instance headless and pulls the results back. Optional |
| `wait.sh` | laptop | when no offer matches: retries provision.sh on an interval, waits for bootstrap, opens the tunnel |
| `apply.sh` | laptop | copies the local `deploy/`, `experiments/`, `jlens/` to the instance and restarts the dashboard. `--pull` does a git pull there instead |

**Reaching it over the web instead of the tunnel** — on an EC2 box, behind GitHub
OAuth restricted to repo collaborators, with an experiment hub that outlives the
rented instance — is [`WEB-DEPLOY.md`](WEB-DEPLOY.md). It keeps everything below
intact: the GPU is still rented on vast.ai on demand (EC2 GPUs are 4-5x the price),
uvicorn still binds `127.0.0.1`, and `destroy.sh` is still the thing that stops the
billing. The tunnel moves off your laptop and onto the always-on box.

`dashboard.py` is the server: a prompt box that renders jlens' own slice
visualisation. Above the slice it shows the model's own output: a greedy
continuation of `gen_tokens` tokens (default 32, `0` to skip) decoded from the same
token ids the slice used, and the top-`top_n` next-token probabilities at the last
prompt position. The L63 row shows the top-1 of that distribution, masked to
word-like tokens; the panel shows it unmasked, with probabilities. It binds to
`127.0.0.1` on the instance, so it is reachable through the tunnel and invisible to
the public internet.

**No model weights are ever downloaded to your laptop.** The 56GB checkpoint and the
3.3GB lens are fetched by the instance, from the instance. Your laptop only moves
JSON and HTML over SSH.

## Why this model and this lens

Fitting a lens for a 27B model costs about 24 H100-hours. That is far outside the
budget here. A lens for `Qwen/Qwen3.8-27B` is already published at
`eyes-ml/Qwen3.8-27B_jacobian-lens`, so we apply a published lens rather than fit our
own. The app log prints the lens's own `n_prompts` when it loads.

This page describes `main`, which serves Qwen. **Serving a different model is a
different git branch, not a different set of environment variables** — the VRAM floors,
the offer filter, the flood band defaults and the `L63` convention all move with the
model. [`MODEL-BRANCHES.md`](MODEL-BRANCHES.md) is the policy: what a model branch may
change, how it stays rebased on `main`, and the checklist for adding one.

**The honest caveat.** That lens was fitted against the model in bf16. We serve the
model quantized to NF4 so it fits on a 48GB card. Reading a quantized model through a
bf16-fitted lens introduces drift, and nothing here corrects for it. Treat the lens
rows as indicative, not exact.

The built-in check is the **final row, L63**. Layer 63 is not in the lens, so jlens
skips transport for it and that row is the model's true output. If L63 looks sane and
the rows below it do not, suspect quantization drift before you suspect the model.

`JLENS_PRECISION=bf16` removes the drift entirely, but 56GB of weights only fits an
80GB card: the offer filter in `provision.sh` would have to be widened (`gpu_ram >= 78`,
and `dph_total` raised well past 0.70), and an 80GB card is outside the $15 budget.
`JLENS_PRECISION=int8` is the middle option at roughly 30GB, though its preflight wants
44GB of VRAM, so it is marginal on a 45GB card and safe on a 48GB one. On a card that is
too small for the precision you asked for, the bootstrap refuses to start before it
downloads anything.

## Usage

First time only: the instance clones this repo from GitHub, so `deploy/` must be
committed and pushed to a public fork before you provision.

```bash
git add deploy/*.sh deploy/dashboard.py deploy/README.md
git commit -m "deploy scripts" && git push
```

`provision.sh` checks for this and refuses to spend money if the push is missing.
The paths are listed explicitly because `deploy/.instance` and `deploy/.known_hosts` are
runtime state, and this fork is public.

Then:

```bash
cd deploy

./provision.sh          # search, show the top offers, confirm, create, wait
./tunnel.sh             # forward localhost:7860, leave this running
# open http://localhost:7860 in a browser
./destroy.sh            # when finished. This stops the billing
```

`provision.sh` returns once sshd is up, but the instance is still installing and
downloading. **Nothing listens on port 7860 until that finishes**, so for the first 15 to
35 minutes there is no dashboard to load: a browser pointed at `localhost:7860` gets a
connection error, not a loading page. Watch the setup instead, and open the tunnel once
the log prints `server started`.

```bash
./tunnel.sh --logs      # then Ctrl-C, and ./tunnel.sh
```

`tunnel.sh` also warns you when the remote port is not listening yet. Once the server is
up, the page does report its own load status while the model loads, which is a further 4
to 8 minutes.

Other things you will want:

```bash
./provision.sh --status     # state, uptime, accrued cost of the current instance
./provision.sh --offer N    # use a specific offer id instead of the cheapest match
./provision.sh --yes        # skip the confirmation prompt

./tunnel.sh --logs          # tail the setup and app logs on the instance
./tunnel.sh --port 7861     # use a different local port
./tunnel.sh <id>            # target an instance other than the one in .instance

./destroy.sh --list         # every instance on the account, and what it costs
./destroy.sh <id>           # destroy one by id
```

The instance id is written to `deploy/.instance` as soon as the instance exists. The
scripts read it from there, so you rarely need to pass an id by hand.

### Changing precision

`provision.sh` does not pass `JLENS_PRECISION` to the instance, so a fresh instance is
always `nf4`. To change it, SSH in and re-run the bootstrap. It is idempotent: it
skips the install and the downloads, and restarts the server.

```bash
JLENS_PRECISION=int8 bash /workspace/jacobian-lens/deploy/bootstrap.sh
```

It refuses to start if the card is too small for the precision you asked for, before
downloading anything.

## Running the flood suite from the dashboard

The `FLOOD SUITE` section at the bottom of the page runs `experiments/flood` **inside
the dashboard process**, reusing the model that is already loaded. There is no second
copy of the weights, and the page stays usable: the run takes the slice lock one item
at a time, so a `/run` slice interleaves between items instead of waiting for all 361.

Set `band_lo`/`band_hi` (clamped to the lens's fitted layers), `positions` (the last N
prompt positions to read out, 1-8) and `gen` (greedy tokens for correctness, 1-32), then
press `start flood`. Rows stream in over server-sent events and the three visuals fill in
as each item finishes:

| visual | rows | columns | colour |
|---|---|---|---|
| accuracy by tier | one line per family | tier | the family's own colour |
| answer rank in the band, last position | items, in the order they ran | the band's fitted layers | `log10(rank+1)`, dark = rank 0 |
| when the answer becomes readable | the same items | the read positions, earliest to last | same scale: this is the `lead` picture |

Hover any cell for the item id, the layer or position, the rank and the continuation.
A hatched row is one whose rank is meaningless (`band_scored: false`, or an answer the
tokenizer does not encode as a single token); the marker on the right edge of the middle
visual is green for correct, amber for answer-later, red for wrong. The left rail colours
each row by family, because `items.jsonl` interleaves `nth-letter` with `count-letter`.

The full per-layer records go to `JLENS_FLOOD_OUT` (default `/workspace/flood-results.jsonl`),
rewritten at the start of every run. Download them with the `results.jsonl` link next to
the button, or `curl localhost:7860/flood/results.jsonl`, then run
`python experiments/flood/analyze.py <file>` on the laptop.

`flood.sh` is still the headless alternative: it stops the dashboard, runs `run.py` over
SSH, pulls the results back and analyzes them. Use it for an unattended sweep; use the
page when you want to watch the tier curves form.

## Cost

Credit at time of writing: **$18.47**. Design target: stay under $15.

Rates below include 120GB of storage and are what the offer filter returned on one
search. They move daily.

| card | VRAM | $/hr with storage |
|---|---|---|
| RTX A6000 | 48GB | 0.433 |
| L40S | 45GB | 0.567 |
| RTX 4090 | 48GB | 0.578 |

Cheaper cards exist. An A40 at $0.281/hr shows up on a looser search. The filter in
`provision.sh` excludes some of them on purpose: it requires at least 16GB of usable host
RAM, Ampere or newer, an allow-listed GPU name, decent download speed, and cheap
bandwidth. A host that charges $20/TB for downloads adds over a dollar to the 59GB of
model and lens.

At $0.433/hr:

| | time | cost |
|---|---|---|
| one-off setup, per instance | 15 to 35 min | $0.11 to $0.25 |
| setup plus an hour of use | ~1.4 hr | ~$0.61 |
| the $15 target buys | ~34 hr | $15 |
| all $18.47 of credit buys | ~42 hr | $18.47 |

Setup is not the risk. Forgetting to destroy is. Storage bills for the whole life of
the instance, running or stopped, so **stopping an instance does not stop the burn.
Only `destroy.sh` does.**

Guardrails already in the scripts:

- `provision.sh` prints the rate and the runway and asks before creating anything.
- It refuses to run if `deploy/.instance` still points at a live instance.
- It refuses to run if credit is below $2. Change with `JLENS_CREDIT_FLOOR`.
- If it fails or you interrupt it after creation, it shouts the instance id and the
  destroy command on the way out. If the create call itself fails after vast.ai accepted
  it, it re-reads the account for an instance labelled `jlens` created in the last
  minute, records the id, and prints the same warning.
- `destroy.sh` re-queries after the delete and does not claim success until the
  contract is actually released.
- After a successful destroy it lists any other instance still running on the
  account. That is the check that catches a forgotten instance from a past session.

None of that survives your laptop. Closing the lid or killing the tunnel leaves the
instance billing, so running `destroy.sh` is mandatory, not advisory. The instance does
carry an idle self-destruct, but it is inert until you hand it an API key:

```bash
scp -P <ssh-port> ~/.config/vastai/vast_api_key root@<ssh-host>:/workspace/.jlens/vast_api_key
```

`tunnel.sh` prints the host and port. Armed, it destroys the instance after two hours
with no dashboard traffic. To change the window, re-run the bootstrap with
`JLENS_IDLE_KILL_S=<seconds>`.

If you are ever unsure whether something is running:

```bash
./destroy.sh --list
```

That reads the account, not the local state file, so it sees instances this repo never
knew about.

## What your laptop needs

- `jq`, `curl`, `ssh`, `git`.
- The vast.ai CLI at `~/.local/bin/vastai`. Override with `VASTAI=/path/to/vastai`.
  Only used for offer search.
- A vast.ai API key at `~/.config/vastai/vast_api_key`, or in `VAST_API_KEY`. The
  scripts read it at point of use and never print it. `provision.sh` and `destroy.sh`
  pass it to curl on stdin; `tunnel.sh` still passes it in an argument, so it is briefly
  visible in `ps` on your own machine.
- An SSH key at `~/.ssh/id_ed25519`, already registered on the vast.ai account.
  Override with `JLENS_SSH_KEY`. Do not run bare `vastai create ssh-key`: with no
  argument it generates a new keypair and overwrites your existing one.
- This fork pushed public on `main`, with `deploy/` in it.

No GPU, no CUDA, no torch, no model weights. Nothing heavy is installed locally.

`tunnel.sh` keeps its own `deploy/.known_hosts`. vast.ai recycles IP and port pairs
across machines, so a changed host key is normal churn here. If SSH refuses on a
changed key, delete that file and try again.

## Troubleshooting

**Browser says connection refused or reset.** The server has not started yet. Expected
for the first 15 to 35 minutes: see the note under Usage. `./tunnel.sh --logs` shows
which stage is running.

**Setup seems stuck.** Watch it: `./tunnel.sh --logs`. The two long stages are the model
download and the NF4 quantize, and both print timings. The full bootstrap transcript is
at `/var/log/portal/jlens-bootstrap.log` on the instance. `provision.sh --status` shows
what vast.ai thinks the instance is doing.

**Instance never reaches `running`.** If `actual_status` reaches `exited`, `unknown`
or `offline`, it will never progress. `provision.sh` detects this and stops. Destroy
it and provision again, which picks a different host.

**Gated HF repo.** Both repos are public today and no token is needed. If that
changes, the download fails with a 403 and `GatedRepoError`. Accept the terms on the
model page while logged in, create a read token, then SSH in and re-run the bootstrap
with `HF_TOKEN=hf_... bash /workspace/jacobian-lens/deploy/bootstrap.sh`. Pass the
token over SSH, never in the provision script: the onstart script is stored on
vast.ai's side.

**CUDA out of memory.** The dashboard catches this and says so rather than dying.
Lower `max_seq_len` or raise `layer_stride` in the form and run again. The expensive
combination is 512 tokens at stride 1, which is two full-vocab sorts on every one of
64 rows. If it happens at the defaults, the card is smaller than the offer claimed:
check the VRAM line in the app log.

**Tunnel port in use.** `tunnel.sh` checks the port before connecting and tells you.
Use `./tunnel.sh --port 7861` and browse `localhost:7861`. The remote port stays 7860.

**Tunnel drops.** Ctrl-C and re-run it. The tunnel is stateless and the instance keeps
running, along with its billing. Closing the tunnel does not stop the instance.

**Dashboard returns 503.** The server is up and the model is still loading. The header
shows the current stage. If it says `failed`, the page prints the full traceback.

**Page is huge or slow to render.** The visualisation inlines its own data, so page
size scales with tokens by rows by tracked tokens. Tracked tokens are capped at 256.
Raise `layer_stride` to halve the rows.
