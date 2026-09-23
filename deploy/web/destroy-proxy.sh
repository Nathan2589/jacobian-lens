#!/usr/bin/env bash
# Tear down the always-on J-lens proxy box and everything provision-proxy.sh made.
#
#   ./destroy-proxy.sh            destroy what .proxy-instance points at
#   ./destroy-proxy.sh --list     show anything still running, and stop
#   ./destroy-proxy.sh <id>       destroy a specific instance
#
# The proxy box is cheap and meant to stay up, so this is not the routine
# operation destroy.sh is for the GPU. It exists so that "I was only trying it
# out" has an exit, and so a released Elastic IP does not sit billing after the
# instance it was attached to is gone - an unattached EIP costs money, which is
# the opposite of the intuition.
set -euo pipefail

REGION="${AWS_REGION:-eu-west-1}"
NAME="${JLENS_PROXY_NAME:-jlens-proxy}"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$DEPLOY_DIR/.proxy-instance"
KEY_OUT="$DEPLOY_DIR/.proxy-key.pem"

die() { echo "FATAL: $*" >&2; exit 1; }

command -v aws >/dev/null || die "aws CLI not installed"

if [ "${1:-}" = "--list" ]; then
  echo "instances tagged $NAME in $REGION:"
  aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=pending,running,stopped,stopping" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,PublicIpAddress]' --output text
  echo "unattached elastic IPs (these bill while idle):"
  aws ec2 describe-addresses --region "$REGION" \
    --query 'Addresses[?AssociationId==null].[PublicIp,AllocationId]' --output text
  exit 0
fi

ID="${1:-}"
if [ -z "$ID" ]; then
  [ -f "$STATE_FILE" ] || die "no instance id: $STATE_FILE missing. Try --list."
  ID="$(tr -d '[:space:]' < "$STATE_FILE")"
  [ -n "$ID" ] || die "$STATE_FILE is empty. Try --list."
fi

DESC="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" \
        --query 'Reservations[].Instances[].[State.Name,PublicIpAddress]' --output text 2>/dev/null || true)"
[ -n "$DESC" ] || { echo "instance $ID does not exist (already destroyed?)"; rm -f "$STATE_FILE"; exit 0; }
echo "instance $ID: $DESC"

# Find the addresses and the security group BEFORE terminating - afterwards the
# association is gone and the EIP is orphaned, still billing, with nothing left
# to point at it.
ALLOCS="$(aws ec2 describe-addresses --region "$REGION" \
          --filters "Name=instance-id,Values=$ID" --query 'Addresses[].AllocationId' --output text)"
SGS="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" \
       --query 'Reservations[].Instances[].SecurityGroups[].GroupId' --output text)"
KEY="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" \
       --query 'Reservations[].Instances[].KeyName' --output text)"

read -r -p "Terminate $ID, release ${ALLOCS:-no} EIP(s), delete SG(s) ${SGS:-none}? [y/N] " A
[ "$A" = y ] || [ "$A" = Y ] || { echo "nothing destroyed"; exit 0; }

aws ec2 terminate-instances --region "$REGION" --instance-ids "$ID" >/dev/null
echo "terminating $ID ..."
aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$ID"
echo "terminated"
rm -f "$STATE_FILE"

for A in $ALLOCS; do
  aws ec2 release-address --region "$REGION" --allocation-id "$A" && echo "released EIP $A"
done

# A security group cannot be deleted until nothing uses it; termination has just
# completed, so this normally succeeds. If it does not, say so rather than
# failing the whole teardown - the expensive thing is already gone.
for G in $SGS; do
  if aws ec2 delete-security-group --region "$REGION" --group-id "$G" 2>/dev/null; then
    echo "deleted security group $G"
  else
    echo "could not delete security group $G (still in use?). Remove it by hand." >&2
  fi
done

if [ -n "$KEY" ] && [ "$KEY" != None ]; then
  aws ec2 delete-key-pair --region "$REGION" --key-name "$KEY" && echo "deleted key pair $KEY"
  rm -f "$KEY_OUT"
fi

echo
echo "Done. Anything else still running on this account:"
aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=pending,running,stopped" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text
echo "Remember the GPU is separate: deploy/destroy.sh --list"
