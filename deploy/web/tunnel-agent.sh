#!/usr/bin/env bash
# Hold open the tunnel from the always-on proxy box to the rented GPU.
#
# This is deploy/tunnel.sh's job moved off the laptop: the dashboard still binds
# 127.0.0.1 on the GPU host, and this forwards the proxy box's own 127.0.0.1:7860
# to it. Nothing on either machine listens publicly on the dashboard port.
#
# Run under systemd (jlens-tunnel.service). It exits on any problem and lets
# systemd restart it, rather than trying to be clever about recovery.
set -uo pipefail

API="https://console.vast.ai/api/v0"
LOCAL_PORT="${JLENS_TUNNEL_LOCAL_PORT:-7860}"
REMOTE_PORT="${JLENS_TUNNEL_REMOTE_PORT:-7860}"
KEY_FILE="${JLENS_VAST_KEY_FILE:-/etc/jlens/vast_api_key}"
ID_FILE="${JLENS_VAST_INSTANCE_FILE:-/etc/jlens/instance}"
SSH_KEY="${JLENS_SSH_KEY:-/etc/jlens/id_ed25519}"
KNOWN_HOSTS="${JLENS_KNOWN_HOSTS:-/var/lib/jlens/known_hosts}"
# No instance is the resting state. Waiting quietly is correct behaviour, so this
# must not look like a crash loop in the journal.
IDLE_SLEEP="${JLENS_TUNNEL_IDLE_SLEEP:-60}"

log() { echo "[$(date -Is)] $*"; }
quiet_exit() { log "$*"; sleep "$IDLE_SLEEP"; exit 0; }

[ -r "$SSH_KEY" ]  || quiet_exit "no ssh key at $SSH_KEY"
[ -r "$KEY_FILE" ] || quiet_exit "no vast.ai API key at $KEY_FILE"
command -v jq >/dev/null || { log "FATAL: jq not installed"; exit 1; }

INSTANCE_ID="${JLENS_INSTANCE_ID:-$(tr -d '[:space:]' < "$ID_FILE" 2>/dev/null || true)}"
[ -n "$INSTANCE_ID" ] || quiet_exit "no instance id recorded; nothing to tunnel to"

KEY="$(tr -d '\n\r' < "$KEY_FILE")"
RESP="$(curl -sS -w '\n%{http_code}' -H "Authorization: Bearer $KEY" \
        "$API/instances/$INSTANCE_ID/?owner=me" 2>/dev/null)" \
  || quiet_exit "could not reach the vast.ai API"
CODE="${RESP##*$'\n'}"; BODY="${RESP%$'\n'*}"
# A rotated key returns a body with a null .instances too, so check the status
# first - otherwise "your key expired" reads as "your instance is gone".
[ "$CODE" = 200 ] || quiet_exit "vast API HTTP $CODE"

INST="$(jq '.instances' <<<"$BODY")"
[ "$INST" != null ] && [ -n "$INST" ] || quiet_exit "instance $INSTANCE_ID no longer exists"
STATUS="$(jq -r '.actual_status // "unknown"' <<<"$INST")"
[ "$STATUS" = running ] || quiet_exit "instance $INSTANCE_ID is '$STATUS'"

if jq -e '.ports["22/tcp"][0].HostPort' >/dev/null <<<"$INST"; then
  SSH_HOST="$(jq -r '.public_ipaddr' <<<"$INST")"
  SSH_PORT="$(jq -r '.ports["22/tcp"][0].HostPort' <<<"$INST")"
else
  SSH_HOST="$(jq -r '.ssh_host' <<<"$INST")"
  SSH_PORT="$(jq -r 'if (.image_runtype // "" | test("jupyter")) then (.ssh_port + 1) else .ssh_port end' <<<"$INST")"
fi
[ -n "$SSH_HOST" ] && [ "$SSH_HOST" != null ] && [[ "$SSH_PORT" =~ ^[0-9]+$ ]] \
  || quiet_exit "instance $INSTANCE_ID exposes no ssh endpoint yet"

mkdir -p "$(dirname "$KNOWN_HOSTS")"
log "tunnelling 127.0.0.1:$LOCAL_PORT -> $SSH_HOST:$SSH_PORT (instance $INSTANCE_ID)"

# -L binds 127.0.0.1 explicitly: a bare "-L 7860:..." would bind every interface
# on some builds, which would put the dashboard on the public internet with no
# auth in front of it. That is the one thing this whole deployment exists to
# prevent, so it is spelled out rather than left to a default.
exec ssh -N \
  -i "$SSH_KEY" \
  -o "UserKnownHostsFile=$KNOWN_HOSTS" \
  -o StrictHostKeyChecking=accept-new \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ConnectTimeout=15 -o LogLevel=ERROR \
  -p "$SSH_PORT" \
  -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" \
  "root@$SSH_HOST"
