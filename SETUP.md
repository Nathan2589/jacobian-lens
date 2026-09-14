# Setup for a teammate

Everything runs on a rented GPU. Your laptop only needs a shell, ssh, and the vast.ai
CLI. Nothing heavy is installed locally and no model weights are ever downloaded to
your machine. Linux, macOS, or WSL on Windows all work; the scripts are bash.

## 1. Laptop prerequisites

```bash
# Debian/Ubuntu/WSL
sudo apt install -y jq curl git openssh-client python3
# macOS
brew install jq
```

The vast.ai CLI, installed to `~/.local/bin/vastai` (where the scripts look):

```bash
mkdir -p ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -o ~/.local/bin/vastai
chmod +x ~/.local/bin/vastai
```

Put the shared API key where the scripts read it. Nathan has sent it to you; paste it
in, no quotes, no trailing spaces:

```bash
mkdir -p ~/.config/vastai
nano ~/.config/vastai/vast_api_key      # one line, the key
chmod 600 ~/.config/vastai/vast_api_key
~/.local/bin/vastai set api-key "$(cat ~/.config/vastai/vast_api_key)"
~/.local/bin/vastai show user | head      # sanity check: your account, current credit
```

An SSH key at `~/.ssh/id_ed25519`. If you don't have one:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
```

Then register the PUBLIC key on the vast.ai account: console.vast.ai → Account → SSH
Keys → paste the contents of `~/.ssh/id_ed25519.pub`. Do not run bare
`vastai create ssh-key` with no argument: it generates a new keypair and overwrites
yours. If your key lives elsewhere, set `JLENS_SSH_KEY=/path/to/key`.

## 2. Clone

```bash
git clone --recurse-submodules git@github.com:Nathan2589/Staxis.git
cd Staxis/jacobian-lens/deploy
```

If you cloned without the submodule, `git submodule update --init` from the Staxis
root fills `jacobian-lens/`.

## 3. Run

```bash
./provision.sh          # lists the cheapest offers, shows the rate and credit, asks y/N
./tunnel.sh --logs      # watch setup; wait for "server started" (15-35 min), then Ctrl-C
./tunnel.sh             # forwards localhost:7860; leave this terminal open
```

Open http://localhost:7860. The header shows the model loading for another 4 to 8
minutes, then `ready`. Top of the page: type a prompt, click run, and the slice view
renders with the model's own continuation above it. Bottom of the page: the FLOOD SUITE
section. Leave the defaults and click start; rows appear as items complete, and the
`results.jsonl` link downloads the full file at any point.

If `provision.sh` says `0 survived`, nothing on the market matches the filters right
now. `./wait.sh` retries every two minutes, provisions on the first match, waits for
the bootstrap, and opens the tunnel, so you can leave it running in a terminal.

When you're finished:

```bash
./destroy.sh            # the only thing that stops the billing
```

## 4. Things that will bite you

- **The account is shared and billing is per instance.** Before provisioning, run
  `./destroy.sh --list` to see whether someone else already has one up. Storage bills
  for the whole life of an instance, running or stopped, so closing your laptop does
  not stop it. Only `destroy.sh` does.
- **Credit is small.** `provision.sh` prints it and refuses below $2. An A6000 is about
  $0.50/hr all in; setup plus an hour of use is under a dollar. Check the number
  before you say `y`.
- **A bad host.** If `provision.sh` sits on `[loading] docker_build() error ...` it
  now fails fast; if it ever hangs, Ctrl-C, `./destroy.sh <id>` with the id it printed,
  and provision again. `./provision.sh --offer <id>` picks a specific offer if the
  cheapest one keeps failing.
- **Connection refused at localhost:7860** just means bootstrap hasn't finished.
  `./tunnel.sh --logs` shows which stage it's on.
- **Changed host key.** vast recycles ip:port pairs. If ssh refuses, delete
  `deploy/.known_hosts` and retry.
- **Downloads on WSL** land in the Windows Downloads folder,
  `/mnt/c/Users/<you>/Downloads/`.

## 5. Reading results

Copy the downloaded file next to the analyzer and run it:

```bash
cd Staxis/jacobian-lens/experiments/flood
cp /path/to/flood-results.jsonl results-<date>.jsonl
python3 analyze.py results-<date>.jsonl
```

Two tables: metrics by family and tier, and by family split into correct versus wrong.
What each column means, and what the visuals mean, is in `ARCHITECTURE.md` (section 4
for the dashboard, section 5 for the suite). Findings so far are in
`experiments/flood/README.md`.

## 6. Editing and applying

Edit locally, then push your working tree to the running instance and restart the
dashboard without a commit:

```bash
cd deploy && ./apply.sh
```

`./apply.sh --pull` does a git pull on the instance instead, if you've pushed. Commit
inside `jacobian-lens/` and push to the fork as a normal repo; then in the Staxis root,
`git add jacobian-lens && git commit` to move the submodule pointer.
