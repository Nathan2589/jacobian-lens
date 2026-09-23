# Putting the dashboard on the web

`deploy/README.md` describes the tunnel deployment: the dashboard binds `127.0.0.1`
on a rented vast.ai box and you reach it with `tunnel.sh`. The tunnel *is* the auth,
which is why `dashboard.py` says it has "no auth, no multi-user handling and no
persistence here by design".

This document describes the web deployment, which removes that property and has to
put something back in its place before anything is exposed.

```
                                  ALWAYS-ON                        RENTED ON DEMAND
                            ┌─────────────────────┐              ┌──────────────────┐
  browser ──── 443 ────────▶│ Caddy (TLS, ACME)   │              │  vast.ai GPU     │
                            │   ↓ 127.0.0.1:7870  │              │                  │
                            │ authproxy.py        │              │  uvicorn on      │
                            │   · GitHub OAuth    │              │  127.0.0.1:7860  │
                            │   · collaborator    │─── ssh -L ──▶│  dashboard.py    │
                            │     check           │   7860       │                  │
                            │   · run history     │              │  idle reaper     │
                            │   · /hub  (React)   │              │  destroys this   │
                            └─────────────────────┘              └──────────────────┘
                              t4g.small, ~$12/mo                  $0.433/hr, ~0 most days
```

Nothing listens publicly on 7860 on either machine. The security group opens 443 and
80 (ACME) to the world and 22 to your own address; the dashboard port is not in it.

---

## The four decisions, resolved

### 1. GPU or not — **split, and the GPU stays on vast.ai**

This was the load-bearing one, and it is settled by price rather than by taste.

The dashboard loads a 27B model in NF4 and needs ~32 GiB of VRAM (`bootstrap.sh`'s
preflight refuses to start below that), so the cheap EC2 GPU instances are out
before cost even enters:

| option | VRAM | verdict |
|---|---|---|
| `g5.xlarge` (A10G) | 24 GB | **too small.** NF4 needs ~32 GiB |
| `g5.12xlarge` (4×A10G) | 4×24 GB | 96 GB total, but `device_map={"": 0}` pins to one card. Would need code changes, and "auto" reinstates the per-call host→device copy the comment in `dashboard.py` exists to avoid |
| `g6e.xlarge` (L40S) | 48 GB | works, and is the comparison that matters |

`g6e.xlarge` lists around **$1.86/hr** on-demand in us-east-1 and somewhat more in
eu-west-1. (I could not confirm this from the box — `pricing:GetProducts` is denied
to the role available here — so treat it as approximate; the argument does not turn
on the third decimal place.) Against the **$0.433/hr** an RTX A6000 costs on vast.ai,
that is roughly 4–5×. vast.ai was chosen on price, and EC2 does not beat it.

Always-on is worse still. A `g6e.xlarge` left running is **~$1,350/month**. The
account this project runs on has **$18.47 of credit** and a $15 design target. An
always-on GPU is not a slightly expensive option, it is three orders of magnitude
outside the budget.

So: **auth and hub on a cheap always-on box, model inference on a GPU rented on
demand, exactly as today.** The tunnel does not disappear — it moves off the laptop
and onto the proxy box, where `jlens-tunnel.service` holds it open. Every existing
guarantee survives: uvicorn still binds `127.0.0.1`, `provision.sh` / `destroy.sh`
still own the instance lifecycle, and `JLENS_IDLE_KILL_S` still reaps.

This also turns out to be the *right* shape rather than merely the affordable one.
The GPU is destroyed routinely, so anything living on it is ephemeral; the proxy box
is the only thing with continuity, and it is the natural home for run history, cost
and the "should I rent one today" question. See decision 4.

### 2. Cost ceiling and idle shutdown

**The new always-on cost is small and bounded:** `t4g.small` (2 vCPU, 2 GB, arm64)
is roughly **$0.0168/hr ≈ $12/month**, plus ~$0.70/month for an 8 GB gp3 root
volume. An Elastic IP is free while attached to a running instance.

**No ALB.** An Application Load Balancer is ~$16/month before it serves a request —
more than the instance it would front. Caddy on the box terminates TLS with a free
ACME certificate and renews it. The Caddyfile is in `web/Caddyfile`.

**The trap this deployment introduces, and the guard against it.** The GPU's idle
reaper uses the mtime of the uvicorn access log as its heartbeat, and uvicorn writes
one line per request. Any health check, uptime monitor or status widget that polls
the dashboard on a timer looks exactly like a person using it. The instance would
then never go idle, never self-destruct, and burn $0.433/hr until the credit ran
out — silently reintroducing the exact failure mode the README's cost section exists
to prevent.

Three things in this deployment exist only to stop that:

- `/_jlens/health` is answered **by the proxy** and never forwarded. Point monitors
  at that path, never at `/`.
- The hub asks the proxy for state, and the proxy determines GPU reachability with a
  **bare TCP connect** — it opens a socket and closes it without sending a request,
  so uvicorn logs nothing.
- Richer status is learned **passively**: when someone has the dashboard open, their
  own `/status` polls pass through the proxy and it keeps the last body it saw. No
  request is ever manufactured.

**Still to do by hand** (the role available from this machine cannot create them —
see "What is not done" below):

```bash
# $20/month budget with an alert at 80% and at forecast-over.
aws budgets create-budget --account-id <acct> \
  --budget '{"BudgetName":"jlens","BudgetLimit":{"Amount":"20","Unit":"USD"},
             "TimeUnit":"MONTHLY","BudgetType":"COST"}' \
  --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL",
     "ComparisonOperator":"GREATER_THAN","Threshold":80},
     "Subscribers":[{"SubscriptionType":"EMAIL","Address":"you@example.com"}]}]'
```

The proxy box has no `destroy.sh` equivalent and is not meant to — it is the cheap
half. The expensive half already has one.

### 3. Which repo's collaborators — **measured, and the answer is awkward**

Checked against the live API on 2026-09-23:

| login | `Nathan2589/jacobian-lens` | `Nathan2589/Diantic` |
|---|---|---|
| `Nathan2589` | 204 ✅ | 204 ✅ |
| `juniorokafor` | **404 ❌** | 204 ✅ |
| `torvalds` | 404 ❌ | 404 ❌ |

**The sets genuinely differ, and the difference is the one teammate.**
`SETUP.md` in this repo is titled "Setup for a teammate" and was written for exactly
one person who is not a collaborator on the public fork.

So gating on `Nathan2589/jacobian-lens` as specified means **Nathan can get in and
nobody else can.** That may well be what is wanted for a first deploy, and it is
what ships as the default — widening an auth boundary is not a decision to make on
someone's behalf.

`JLENS_AUTH_REPOS` is a comma-separated **any-of** list, so opening it to the
teammate is a one-line change and a restart:

```
JLENS_AUTH_REPOS=Nathan2589/jacobian-lens,Nathan2589/Diantic
```

The alternative — adding `juniorokafor` to the public fork — also works and is
arguably tidier, but it is a change to a public repo's collaborator list, so it is
Nathan's call, not a deployment detail.

### 4. One model or several — **one, chosen at deploy time**

`MODEL-BRANCHES.md` is unchanged and still governs: a model is a git branch, because
the VRAM floors, the offer filter, the flood band defaults and the `L63` convention
all move with the model and none of them is a value an env var can carry.

An EC2 deployment picks a branch the same way provisioning does. `JLENS_BRANCH` and
`JLENS_MODEL_ID` in `/etc/default/jlens` are **labels** — the hub displays them so
you can see at a glance which model this box is fronting. They do not select
anything; the GPU instance was cloned from a branch and that is what determines the
model.

**No model switcher.** The proxy is deliberately model-agnostic: it forwards to
whatever is on the far end of the tunnel. Switching models is provision-a-new-
instance-from-another-branch and update two labels, which is the same operation it
already was.

---

## The auth design

### Authentication and authorization are separate, and only the second one matters

Signing in with GitHub proves the visitor has a GitHub account. Every attacker has a
GitHub account. The check that does the work is the collaborator lookup.

### The collaborator check uses a *server-side* token

This is the one design decision worth arguing for explicitly, because the obvious
implementation is different and worse.

The obvious implementation asks the visitor for a token with enough scope to read
collaborators, then uses *their* token to check *themselves*. That requires `repo`
or `read:org` on the visitor's grant — a broad ask for a dashboard — and it makes
the authorization answer depend on a credential the visitor controls.

Instead, the visitor's token is used for exactly one call, `GET /user`, to learn
their login. The collaborator lookup is then made with `JLENS_GITHUB_TOKEN`, a
credential belonging to the deployment.

Consequences, all good:

- **The OAuth app requests no scopes at all.** `scope=` is sent empty; an unscoped
  token still returns the login from `GET /user`. This is the minimum, and it is
  lower than the `read:org` / `repo` the task notes suggested confirming. A rejected
  stranger has granted the app nothing.
- It works for private repos. A visitor's token could never tell us whether they are
  on `Diantic`; ours can.
- The answer cannot be influenced by the person being judged.

Minimum permission on `JLENS_GITHUB_TOKEN`: a fine-grained PAT with **Repository
permissions → Administration: Read** on the repos in `JLENS_AUTH_REPOS`. A classic
token needs `repo`. It must have push access to the repo for the collaborators
endpoint to answer at all.

### The session is a signed cookie, and that is the cache

`jlens_session` carries `{login, repo, iat, exp}` with an HMAC-SHA256 signature.

- `Secure`, `HttpOnly`, `SameSite=Lax`, `Max-Age` = `JLENS_SESSION_TTL_S` (8h default).
- `Lax` and not `Strict` on purpose: the OAuth callback is a cross-site top-level GET
  and `Strict` would drop the cookie on the way back in, producing a login loop.
- No server-side store, so the proxy restarts without logging anyone out and there is
  nothing to grow unboundedly.
- **This is the per-session cache the task asked for.** A signed-in user costs zero
  GitHub API calls per request. Revoking someone takes effect within the TTL.

### Failure modes that were designed for rather than discovered

- **Denied user → terminal 403 page**, never a redirect. It names the account they
  signed in as and the repo they are not on, and links to sign-out. Bouncing a
  rejected user back to `/login` is the loop the task called out, so the rejection
  page has no redirect in it at all.
- **GitHub unreachable / token expired → 503, fails closed.** `authorize()`
  distinguishes "GitHub said no" from "GitHub did not answer". Both deny, but they
  render differently, because a rate-limited server token locking the whole team out
  is an outage to fix and not a person to turn away.
- **Callback with no state cookie → terminal 400** explaining that the cookie did not
  come back, with the likely causes. This is the other half of the loop guard.
- **`JLENS_PUBLIC_URL` not https → the proxy refuses to start.** A `Secure` cookie
  over plain http is silently discarded by the browser and the only symptom is an
  endless login loop that looks like an OAuth bug. `JLENS_INSECURE_COOKIES=1` is the
  local-testing escape hatch and is the only way to get past it.
- **API routes answer 401 JSON, not a 302.** The hub fetches these; a redirect to
  github.com surfaces in a browser as an opaque CORS error rather than "sign in
  again".

---

## The experiment hub

`/hub/` is a React app (Vite, Tailwind, shadcn-style primitives, vendored react-bits
components) served as static files by the proxy, behind the same gate as everything
else. Source in `web/hub/`, build output committed at `web/hub/dist/`.

It lives on the always-on box, not on the GPU, for the reason in decision 1: the GPU
is destroyed routinely and a history that dies with it is not a history. It shows:

- **Deployment** — model, branch, lens repo, precision, and live GPU status.
- **Experiments** — runs recorded, last 24h, success rate, median slice latency.
  All from the proxy's own log, so this renders with no vast.ai key and no GPU
  rented, which is the resting state and exactly when you are deciding whether to
  spend money renting one.
- **Instance** — accrued cost, uptime, credit, runway, and a two-step destroy
  button. The README's cost section says forgetting to destroy is the real risk,
  which makes this the highest-value control on the page; the confirm step also
  re-checks the instance id server-side, so a stale tab cannot destroy a
  *replacement* instance provisioned after the one it was showing.
- **Run history** — every `/run` and `/flood/start` that passed through the proxy,
  with who ran it and on which branch.

The run log is populated by the proxy observing traffic it is already forwarding. It
costs no extra request, which is the same constraint as everything else here.

`dashboard.py`'s own page has been restyled to the same tokens so moving between the
two does not feel like moving between two products. That change is CSS and one header
link; no JavaScript, no route and no behaviour was touched.

### Rebuilding the hub

```bash
cd deploy/web/hub
npm ci
npm run build          # writes dist/, which is committed
```

`dist/` is committed on purpose. The proxy box is a 2 GB arm64 instance with no node
toolchain, and installing one at boot purely to produce ~180 kB of gzipped static
assets is worse than carrying the assets. If you change anything under `hub/src/`,
rebuild and commit `dist/` in the same change.

> Note for this box specifically: `NODE_ENV=production` is exported in the shell
> here, which makes a plain `npm install` silently skip every devDependency —
> including vite, typescript and tailwind. Use `NODE_ENV=development npm install
> --include=dev`, or `npm ci` which ignores it.

---

## Deploying it

### 1. GitHub OAuth app

<https://github.com/settings/developers> → New OAuth App.

- Homepage URL: `https://jlens.example.com`
- **Authorization callback URL: `https://jlens.example.com/_jlens/callback`** — must
  match `JLENS_PUBLIC_URL` exactly or GitHub rejects the callback.
- Request no scopes. The app does not need any; see the auth design above.

Note the client id and generate a client secret.

### 2. The proxy box

`t4g.small`, Ubuntu 24.04 arm64, in a security group that allows **443 and 80 from
0.0.0.0/0** (80 is required for the ACME HTTP challenge) and **22 from your own
address only**. Nothing else. Pass `web/cloud-init.yaml` as user-data.

The instance boots but deliberately does **not** start serving: `authproxy.py`
refuses to run without its config, so a box that boots with an empty secret cannot
accidentally serve anything.

### 3. Fill in the secrets

```bash
sudo editor /etc/default/jlens      # hostname, OAuth pair, session secret, repos
sudo install -m 0640 -o root -g jlens /dev/stdin /etc/jlens/vast_api_key   # paste key
sudo install -m 0640 -o root -g jlens ~/.ssh/id_ed25519 /etc/jlens/id_ed25519
echo "$INSTANCE_ID" | sudo tee /etc/jlens/instance
sudo systemctl enable --now jlens-proxy jlens-tunnel caddy
```

`JLENS_SESSION_SECRET` must be at least 32 characters; the proxy refuses to start
otherwise. `openssl rand -hex 32`.

The SSH key is the one already registered on the vast.ai account — the same key
`tunnel.sh` uses from the laptop.

### 4. DNS

Point an A record at the instance's Elastic IP. Caddy gets the certificate on first
request. Until DNS resolves, ACME fails and the site does not come up; that is the
expected order.

### 5. Check it

```bash
curl -sS https://jlens.example.com/_jlens/health          # ok
curl -sSI https://jlens.example.com/ | head -1            # 302 to /_jlens/login
```

Then open it in a browser, sign in, and confirm you land on the dashboard (or on
`/hub/` if no GPU is rented — the root falls back there rather than showing a 502
nobody can act on).

---

## Tests

```bash
cd deploy/web && python3 -m pytest test_authproxy.py -q      # 42 tests
cd deploy/web/hub && npm run lint                            # tsc --noEmit
```

The Python tests are mostly tests of the authorization boundary, since that is where
being wrong is not a bug report but an open dashboard. They cover forged and expired
sessions, the fail-closed path when GitHub errors, the empty-scope assertion, the
two redirect-loop guards, path traversal in the static handler, and the
`summarise_request` / `StaticSite` / `RunStore` internals. Nothing touches the
network.

---

## What is *not* done

**Nothing has been deployed to AWS.** The role available from this machine
(`meet-role` in account 350353785522) is read-plus-a-narrow-operator-set. Verified
denials:

| action | result |
|---|---|
| `ec2:RunInstances` | `UnauthorizedOperation` |
| `route53:ListHostedZones` | `AccessDenied` |
| `budgets:ViewBudget` | `AccessDenied` |
| `pricing:GetProducts` | `AccessDenied` |

So steps 2–4 above are a human handover. Everything that could be built and tested
without AWS has been: the proxy, the hub, the units, the Caddyfile, the cloud-init,
and the tests, all exercised end to end locally against a stub dashboard.

Also outstanding:

- **The budget alarm and the DNS record** (above) need an account with those
  permissions.
- **21st.dev components were not used.** Its registry returns 403 without
  authentication and the `magic` MCP server that normally provides access failed to
  connect. The hub uses shadcn/ui-style primitives instead, which is the same
  substrate 21st.dev components are built on, so the visual result is in the same
  family. react-bits components *were* used and are vendored in
  `web/hub/src/components/reactbits/` with provenance in the README there.
- **The OAuth flow has not been exercised against real GitHub**, because that needs
  a registered app and a public callback URL. The collaborator endpoint's 204/404
  behaviour was verified directly against the live API (table in decision 3); the
  code path around it is covered by tests with a fake client.
