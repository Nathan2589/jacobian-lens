#!/usr/bin/env bash
# Destroy the vast.ai instance. This is the ONLY thing that stops the billing.
#   ./destroy.sh          destroy the instance recorded in .instance
#   ./destroy.sh <id>     destroy a specific instance
#   ./destroy.sh --list   show every instance on the account and what it costs
set -euo pipefail

API="https://console.vast.ai/api/v0"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$DEPLOY_DIR/.instance"
KEY_FILE="$HOME/.config/vastai/vast_api_key"
GONE_TIMEOUT=90   # seconds to wait for the contract to actually release

die() { echo "FATAL: $*" >&2; exit 1; }

command -v jq >/dev/null || die "jq not installed"
if [ -n "${VAST_API_KEY:-}" ]; then
  KEY="$VAST_API_KEY"
else
  [ -r "$KEY_FILE" ] || die "no vast.ai API key at $KEY_FILE (or set VAST_API_KEY)"
  KEY="$(cat "$KEY_FILE")"
fi
KEY="${KEY//[$'\n\r']/}"
[ -n "$KEY" ] || die "vast.ai API key is empty"

# The key goes in via a header file, never on the command line: argv is readable
# in /proc for the life of the call.
auth() { printf 'Authorization: Bearer %s\n' "$KEY"; }

api() { # api METHOD PATH -> response body. Dies on anything that is not a 200.
  local r c b
  r="$(curl -sS -X "$1" -H @<(auth) -w '\n%{http_code}' "$API$2")" \
    || die "vast API $1 $2: curl failed - could not reach vast.ai"
  c="${r##*$'\n'}"; b="${r%$'\n'*}"
  [ "$c" = 200 ] || die "vast API $1 $2 returned HTTP $c: $b"
  [ -n "$b" ] || die "vast API $1 $2 returned HTTP 200 with an empty body"
  printf '%s' "$b"
}

credit() { # a courtesy line only - never fatal, the guardrail is the destroy itself
  local c
  c="$(curl -sS -H @<(auth) "$API/users/current/" | jq -r '.credit // empty' || true)"
  if [ -n "$c" ]; then printf 'credit remaining: $%.2f\n' "$c"
  else echo "credit remaining: unknown (could not read the account)"; fi
}

# Every instance on the account, not just the one this repo knows about.
all_instances() { api GET "/instances/?owner=me" | jq '.instances // []'; }

render() { # render <json-array-of-instances>
  local total
  printf '%-10s %-10s %-9s %-20s %-8s %s\n' ID STATE '$/HR' GPU UPTIME LABEL
  jq -r --argjson now "$(date +%s)" '.[] | [
      .id, (.actual_status // .cur_state // "?"), (.dph_total // 0),
      "\(.num_gpus // 1)x \(.gpu_name // "?")",
      "\((($now - (.start_date // $now)) / 360 | floor) / 10)h",
      (.label // "-")] | @tsv' <<<"$1" \
    | while IFS=$'\t' read -r a b c d e f; do
        printf '%-10s %-10s $%-8.2f %-20s %-8s %s\n' "$a" "$b" "$c" "$d" "$e" "$f"
      done
  total="$(jq '[.[] | .dph_total // 0] | add // 0' <<<"$1")"
  printf '\ntotal burn: $%.2f/hr across %s instance(s)\n' "$total" "$(jq 'length' <<<"$1")"
}

# ---------------------------------------------------------------- args
INSTANCE_ID=""
case "${1:---state}" in
  --list|-l)
    INSTANCES="$(all_instances)"
    if [ "$(jq 'length' <<<"$INSTANCES")" -eq 0 ]; then
      echo "no instances on the account - nothing is billing."
    else
      render "$INSTANCES"
      echo
      echo "destroy any of them with: $0 <id>"
    fi
    credit
    exit 0 ;;
  -h|--help) sed -n '2,5p' "${BASH_SOURCE[0]}"; exit 0 ;;
  --state)
    [ -f "$STATE_FILE" ] || die "no state file at $STATE_FILE. Pass an instance id, or run '$0 --list' to see what is running."
    INSTANCE_ID="$(tr -d '[:space:]' < "$STATE_FILE")"
    [ -n "$INSTANCE_ID" ] || die "$STATE_FILE is empty. Run '$0 --list'." ;;
  -*) die "unknown argument: $1" ;;
  *)  INSTANCE_ID="$1" ;;
esac

# ---------------------------------------------------------------- look before deleting
# A missing instance comes back as HTTP 200 with "instances": null, not a 404. api()
# has already killed every other status, so null here really does mean "not there".
INST="$(api GET "/instances/$INSTANCE_ID/?owner=me" | jq '.instances')"

if [ "$INST" = null ]; then
  if [ -f "$STATE_FILE" ] && [ "$(tr -d '[:space:]' < "$STATE_FILE")" = "$INSTANCE_ID" ]; then
    echo "instance $INSTANCE_ID no longer exists - already destroyed. Clearing $STATE_FILE."
    rm -f "$STATE_FILE"
    credit
    exit 0
  fi
  # An id typed by hand that does not exist is a mistake worth failing on: whatever
  # the user meant to kill is still running.
  die "instance $INSTANCE_ID does not exist on this account. Run '$0 --list'."
fi

jq -r '"destroying \(.id): \(.num_gpus // 1)x \(.gpu_name // "?") at $\(.dph_total // 0)/hr, state \(.actual_status // .cur_state // "?")"' <<<"$INST"
jq -r --argjson now "$(date +%s)" '
  "  uptime \((($now - (.start_date // $now)) / 360 | floor) / 10) hr, accrued ~$\((($now - (.start_date // $now)) / 3600 * (.dph_total // 0) * 100 | floor) / 100)"' <<<"$INST"

# ---------------------------------------------------------------- destroy
RESP="$(curl -sS -X DELETE -H @<(auth) -H 'Content-Type: application/json' \
        -d '{}' -w '\n%{http_code}' "$API/instances/$INSTANCE_ID/")"
CODE="${RESP##*$'\n'}"
BODY="${RESP%$'\n'*}"

if [ "$CODE" != 200 ] || [ "$(jq -r '.success // false' <<<"$BODY" 2>/dev/null)" != true ]; then
  echo "DESTROY FAILED (HTTP $CODE) - instance $INSTANCE_ID IS STILL BILLING" >&2
  echo "api response: $BODY" >&2
  echo "retry this script, or kill it by hand at https://cloud.vast.ai/instances/" >&2
  exit 1
fi

# ---------------------------------------------------------------- confirm it took
# Do not trust the 200. Re-query until the instance is gone or the contract unloads.
DEADLINE=$(( $(date +%s) + GONE_TIMEOUT ))
while :; do
  INST="$(api GET "/instances/$INSTANCE_ID/?owner=me" | jq '.instances')"
  if [ "$INST" = null ] || [ "$(jq -r '.cur_state // ""' <<<"$INST")" = unloaded ]; then
    echo "confirmed: instance $INSTANCE_ID is destroyed and no longer billing."
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "DESTROY UNCONFIRMED: the API accepted the delete but instance $INSTANCE_ID still reports" >&2
    jq -c '{cur_state, actual_status, intended_status}' <<<"$INST" >&2
    echo "after ${GONE_TIMEOUT}s. Assume it is STILL BILLING. Check https://cloud.vast.ai/instances/" >&2
    exit 1
  fi
  sleep 5
done

# Only clear the state file if it pointed at the instance we just killed - otherwise
# provision.sh loses its "one instance at a time" interlock and rents a second GPU.
if [ -f "$STATE_FILE" ] && [ "$(tr -d '[:space:]' < "$STATE_FILE")" = "$INSTANCE_ID" ]; then
  rm -f "$STATE_FILE"
fi
credit

# The instance this repo tracked is dead, but the account may not be idle. This is the
# check that catches a forgotten instance from an earlier session.
REMAINING="$(all_instances | jq --argjson gone "$INSTANCE_ID" '[.[] | select(.id != $gone)]')"
if [ "$(jq 'length' <<<"$REMAINING")" -gt 0 ]; then
  echo
  echo "############################################################"
  echo "# WARNING: other instances are STILL RUNNING on this account"
  echo "############################################################"
  render "$REMAINING"
  echo
  echo "destroy them with: $0 <id>"
  exit 0
fi
echo "no instances left on the account."
