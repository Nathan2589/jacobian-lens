#!/usr/bin/env bash
# Push the local working tree's code to the running vast.ai instance and restart the
# dashboard, so an edit here is live there without a commit.
#   ./apply.sh                copy deploy/, experiments/, jlens/ over scp, re-run bootstrap
#   ./apply.sh --pull         git pull on the instance instead (needs the fork pushed)
#   ./apply.sh <instance-id>  target an instance other than the one in .instance
set -euo pipefail

API="https://console.vast.ai/api/v0"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"
STATE_FILE="$DEPLOY_DIR/.instance"
KNOWN_HOSTS="$DEPLOY_DIR/.known_hosts"
SSH_KEY="${JLENS_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_REPO=/workspace/jacobian-lens
MODE=copy
INSTANCE_ID=""

die() { echo "FATAL: $*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pull) MODE=pull; shift ;;
    -h|--help) sed -n '2,6p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) die "unknown argument: $1" ;;
    *) INSTANCE_ID="$1"; shift ;;
  esac
done

[ -f "$SSH_KEY" ] || die "ssh key not found at $SSH_KEY (set JLENS_SSH_KEY)"
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

rssh "test -d $REMOTE_REPO/deploy" || die "no repo at $REMOTE_REPO on $INSTANCE_ID; bootstrap has not run yet"
echo "instance $INSTANCE_ID via root@$SSH_HOST:$SSH_PORT"

if [ "$MODE" = pull ]; then
  echo "git pull on the instance"
  rssh "cd $REMOTE_REPO && git pull --ff-only"
else
  # The instance runs the repo from an editable install, so replacing the source
  # trees is the whole update. No state lives in these directories.
  echo "copying deploy/ experiments/ jlens/ from $REPO_DIR"
  scp -q "${SSH_OPTS[@]}" -P "$SSH_PORT" -r \
    "$REPO_DIR/deploy" "$REPO_DIR/experiments" "$REPO_DIR/jlens" "root@$SSH_HOST:$REMOTE_REPO/"
  rssh "rm -f $REMOTE_REPO/deploy/.instance $REMOTE_REPO/deploy/.known_hosts"
fi

# bootstrap.sh is idempotent: it skips the install and downloads, restarts the
# server, and returns once the server answers on 7860. Model load then takes 4-8
# min more; the page reports it.
echo "restarting the dashboard (bootstrap.sh, idempotent)"
echo
rssh -t "bash $REMOTE_REPO/deploy/bootstrap.sh"
echo
echo "applied. The page shows load progress; ./tunnel.sh --logs tails it."
