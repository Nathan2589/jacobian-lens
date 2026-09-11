#!/usr/bin/env bash
# Run the capacity-flood suite (experiments/flood) on the vast.ai instance from this laptop.
#   ./flood.sh                       copy the suite over, run it, pull results, analyze
#   ./flood.sh -- --band 20 60       pass extra arguments through to run.py
#   ./flood.sh --no-restart          leave the dashboard down afterwards
#   ./flood.sh <instance-id>         target an instance other than the one in .instance
# The dashboard is stopped while the suite runs (two copies of the model do not fit a
# 48GB card) and restarted afterwards. Results land in experiments/flood/results.jsonl.
set -euo pipefail

API="https://console.vast.ai/api/v0"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd "$DEPLOY_DIR/../experiments/flood" && pwd)"
STATE_FILE="$DEPLOY_DIR/.instance"
KNOWN_HOSTS="$DEPLOY_DIR/.known_hosts"
SSH_KEY="${JLENS_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_REPO=/workspace/jacobian-lens
REMOTE_OUT=/workspace/flood-results.jsonl
LOCAL_OUT="$SUITE_DIR/results.jsonl"
RESTART=1
INSTANCE_ID=""
RUN_ARGS=()

die() { echo "FATAL: $*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-restart) RESTART=0; shift ;;
    --) shift; RUN_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) die "unknown argument: $1 (arguments for run.py go after --)" ;;
    *) INSTANCE_ID="$1"; shift ;;
  esac
done

[ -f "$SSH_KEY" ] || die "ssh key not found at $SSH_KEY (set JLENS_SSH_KEY)"
[ -f "$SUITE_DIR/run.py" ] && [ -f "$SUITE_DIR/items.jsonl" ] \
  || die "suite not found at $SUITE_DIR (run 'python items.py > items.jsonl' there first)"
command -v jq >/dev/null || die "jq not installed"

# --- resolve the instance, same as tunnel.sh -------------------------------------
if [ -z "$INSTANCE_ID" ]; then
  [ -f "$STATE_FILE" ] || die "no instance id: $STATE_FILE is missing. Run provision.sh, or pass an id."
  INSTANCE_ID="$(tr -d '[:space:]' < "$STATE_FILE")"
  [ -n "$INSTANCE_ID" ] || die "$STATE_FILE is empty. Run '$DEPLOY_DIR/destroy.sh --list' to find the id."
fi
KEY="${VAST_API_KEY:-$(cat "$HOME/.config/vastai/vast_api_key")}"
KEY="${KEY//[$'\n\r']/}"
[ -n "$KEY" ] || die "no vast.ai API key"
RESP="$(curl -sS -w '\n%{http_code}' -H "Authorization: Bearer $KEY" \
        "$API/instances/$INSTANCE_ID/?owner=me")" || die "could not reach the vast.ai API"
CODE="${RESP##*$'\n'}"
BODY="${RESP%$'\n'*}"
[ "$CODE" = 200 ] || die "vast API error (HTTP $CODE): $BODY"
INST="$(jq '.instances' <<<"$BODY")"
[ "$INST" != null ] && [ -n "$INST" ] || die "instance $INSTANCE_ID does not exist (already destroyed?)"
STATUS="$(jq -r '.actual_status // "unknown"' <<<"$INST")"
[ "$STATUS" = running ] || die "instance $INSTANCE_ID is '$STATUS', not running. Check: $DEPLOY_DIR/provision.sh --status"
if jq -e '.ports["22/tcp"][0].HostPort' >/dev/null <<<"$INST"; then
  SSH_HOST="$(jq -r '.public_ipaddr' <<<"$INST")"
  SSH_PORT="$(jq -r '.ports["22/tcp"][0].HostPort' <<<"$INST")"
else
  SSH_HOST="$(jq -r '.ssh_host' <<<"$INST")"
  SSH_PORT="$(jq -r 'if (.image_runtype // "" | test("jupyter")) then (.ssh_port + 1) else .ssh_port end' <<<"$INST")"
fi
[ -n "$SSH_HOST" ] && [ "$SSH_HOST" != null ] && [[ "$SSH_PORT" =~ ^[0-9]+$ ]] \
  || die "instance $INSTANCE_ID exposes no ssh endpoint yet"

SSH_OPTS=(-i "$SSH_KEY"
  -o "UserKnownHostsFile=$KNOWN_HOSTS" -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=15 -o LogLevel=ERROR)
rssh() { ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "root@$SSH_HOST" "$@"; }

# --- bootstrap must have finished: the model cache and venv come from it ----------
rssh 'test -f /workspace/.jlens/deps.ok && test -f /venv/main/bin/activate' \
  || die "bootstrap has not finished on $INSTANCE_ID (no deps stamp). Watch it: $DEPLOY_DIR/tunnel.sh --logs"

echo "instance $INSTANCE_ID via root@$SSH_HOST:$SSH_PORT"
echo "copying $SUITE_DIR -> $REMOTE_REPO/experiments/flood"
rssh "mkdir -p $REMOTE_REPO/experiments"
scp -q "${SSH_OPTS[@]}" -P "$SSH_PORT" -r "$SUITE_DIR" "root@$SSH_HOST:$REMOTE_REPO/experiments/"

# --- stop the dashboard, run, restart -----------------------------------------------
# The suite's output is tee'd into the app log as well as its own: the idle reaper
# uses the app log's mtime as the heartbeat, and the dashboard is down meanwhile.
echo "stopping the dashboard and running the suite (model load is 4-8 min, then ~1 line/item)"
echo
RC=0
# shellcheck disable=SC2016
rssh 'set -euo pipefail
PID="$(cat /var/run/jlens.pid 2>/dev/null || true)"
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
  kill "$PID"; while kill -0 "$PID" 2>/dev/null; do sleep 1; done
  echo "dashboard stopped (pid $PID)"
fi
source /venv/main/bin/activate
export HF_HOME=/workspace/hf
cd '"$REMOTE_REPO"'
python experiments/flood/run.py --items experiments/flood/items.jsonl --out '"$REMOTE_OUT"' '"${RUN_ARGS[*]:-}"' \
  2>&1 | tee -a /var/log/portal/jlens-flood.log /var/log/portal/jlens-app.log' || RC=$?

echo
if [ "$RESTART" = 1 ]; then
  echo "restarting the dashboard in the background (bootstrap.sh is idempotent; ./tunnel.sh --logs to watch)"
  rssh "setsid nohup bash $REMOTE_REPO/deploy/bootstrap.sh >> /var/log/portal/jlens-bootstrap.log 2>&1 < /dev/null &"
else
  echo "dashboard left down. Bring it back with: ssh ... 'bash $REMOTE_REPO/deploy/bootstrap.sh'"
fi
[ "$RC" = 0 ] || die "run.py exited with $RC on the instance; see /var/log/portal/jlens-flood.log there"

scp -q "${SSH_OPTS[@]}" -P "$SSH_PORT" "root@$SSH_HOST:$REMOTE_OUT" "$LOCAL_OUT"
echo "results -> $LOCAL_OUT"
echo
python3 "$SUITE_DIR/analyze.py" "$LOCAL_OUT"
echo
echo "the instance is still running and billing. Stop it with: $DEPLOY_DIR/destroy.sh"
