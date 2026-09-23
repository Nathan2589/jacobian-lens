#!/usr/bin/env bash
# Create the always-on EC2 box that fronts the J-lens dashboard.
#
#   ./provision-proxy.sh --hostname jlens.example.com --email you@example.com
#   ./provision-proxy.sh --preflight      check permissions, create nothing
#   ./provision-proxy.sh --status         what exists right now
#
# This is deploy/provision.sh's sibling for the cheap half. It creates a
# security group, a key pair, an Elastic IP and one t4g.small running
# cloud-init.yaml - about $12/month, versus the $0.433/hr GPU that is rented
# separately and destroyed when idle.
#
# It deliberately does NOT hold any secret. The box boots refusing to serve
# until /etc/default/jlens is filled in by a human, because a proxy that starts
# with a default session secret is worse than one that does not start.
set -euo pipefail

REGION="${AWS_REGION:-eu-west-1}"
INSTANCE_TYPE="${JLENS_PROXY_TYPE:-t4g.small}"
NAME="${JLENS_PROXY_NAME:-jlens-proxy}"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$DEPLOY_DIR/.proxy-instance"
KEY_OUT="$DEPLOY_DIR/.proxy-key.pem"

HOSTNAME_ARG=""
EMAIL_ARG=""
MODE=create

die() { echo "FATAL: $*" >&2; exit 1; }
say() { echo "  $*"; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --hostname) HOSTNAME_ARG="${2:-}"; shift 2 ;;
    --email)    EMAIL_ARG="${2:-}"; shift 2 ;;
    --preflight) MODE=preflight; shift ;;
    --status)   MODE=status; shift ;;
    --region)   REGION="${2:-}"; shift 2 ;;
    -h|--help)  sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

command -v aws >/dev/null || die "aws CLI not installed"
aws sts get-caller-identity >/dev/null 2>&1 || die "no usable AWS credentials"
CALLER="$(aws sts get-caller-identity --query Arn --output text)"

# ---------------------------------------------------------------- status
if [ "$MODE" = status ]; then
  echo "caller: $CALLER"
  if [ -f "$STATE_FILE" ]; then
    ID="$(tr -d '[:space:]' < "$STATE_FILE")"
    aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" \
      --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,PublicIpAddress]' \
      --output text 2>/dev/null || echo "instance $ID no longer exists"
  else
    echo "no instance recorded in $STATE_FILE"
  fi
  aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=pending,running,stopped" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' --output text
  exit 0
fi

# ------------------------------------------------------------- preflight
# Report every missing permission at once. Finding them one failed call at a
# time, each after some resource has already been created, is how you end up
# with a half-built stack and no script that will finish it.
echo "preflight (caller: $CALLER)"
MISSING=()
probe() {
  local label="$1"; shift
  local out
  out="$("$@" 2>&1 || true)"
  if grep -q 'DryRunOperation' <<<"$out"; then
    say "ok    $label"
  elif grep -qE 'UnauthorizedOperation|AccessDenied' <<<"$out"; then
    say "DENIED $label"
    MISSING+=("$label")
  else
    say "?     $label (unexpected: $(head -c 100 <<<"$out" | tr '\n' ' '))"
  fi
}

AMI="$(aws ssm get-parameters --region "$REGION" \
  --names /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
  --query 'Parameters[0].Value' --output text 2>/dev/null || true)"
[[ "$AMI" == ami-* ]] || die "could not resolve an Ubuntu 24.04 arm64 AMI in $REGION"
say "ami   $AMI"

probe "ec2:CreateSecurityGroup" aws ec2 create-security-group --region "$REGION" --dry-run \
  --group-name "$NAME-probe" --description probe
probe "ec2:CreateKeyPair"       aws ec2 create-key-pair --region "$REGION" --dry-run --key-name "$NAME-probe"
probe "ec2:AllocateAddress"     aws ec2 allocate-address --region "$REGION" --dry-run
probe "ec2:RunInstances"        aws ec2 run-instances --region "$REGION" --dry-run \
  --image-id "$AMI" --instance-type "$INSTANCE_TYPE"

if [ "${#MISSING[@]}" -gt 0 ]; then
  echo
  echo "Missing ${#MISSING[@]} permission(s): ${MISSING[*]}"
  echo "This role cannot provision the proxy box. Re-run with a role that can"
  echo "(the AdministratorAccess SSO role in this account does)."
  [ "$MODE" = preflight ] && exit 1
  die "refusing to start creating things it cannot finish"
fi
echo "preflight OK"
[ "$MODE" = preflight ] && exit 0

# ---------------------------------------------------------------- create
[ -n "$HOSTNAME_ARG" ] || die "--hostname is required (the name you will point at this box)"
[ -n "$EMAIL_ARG" ]    || die "--email is required (ACME registration; certs fail silently without it)"

# Refuse to double-provision. The same guard provision.sh has, for the same
# reason: a forgotten second instance bills quietly.
if [ -f "$STATE_FILE" ]; then
  OLD="$(tr -d '[:space:]' < "$STATE_FILE")"
  STATE="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$OLD" \
           --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null || true)"
  case "$STATE" in
    pending|running|stopped|stopping)
      die "$STATE_FILE still points at $OLD ($STATE). Destroy it first, or remove the file." ;;
  esac
fi

MY_IP="$(curl -fsS --max-time 10 https://checkip.amazonaws.com | tr -d '[:space:]')" \
  || die "could not determine your public IP (needed to scope the SSH rule)"
[[ "$MY_IP" =~ ^[0-9.]+$ ]] || die "unexpected public IP: $MY_IP"

VPC="$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true \
       --query 'Vpcs[0].VpcId' --output text)"
[ "$VPC" != None ] || die "no default VPC in $REGION; pass one explicitly (edit this script)"

cat <<EOF

About to create, in $REGION:
  instance      1 x $INSTANCE_TYPE  (~\$12/month, always on)
  volume        8 GB gp3            (~\$0.70/month)
  elastic IP    1                   (free while attached)
  security grp  443 + 80 from anywhere, 22 from $MY_IP/32 only
  hostname      $HOSTNAME_ARG
  key pair      written to $KEY_OUT

The dashboard port (7860) is NOT in that security group and never should be.
EOF
read -r -p "Create these? [y/N] " ANSWER
[ "$ANSWER" = y ] || [ "$ANSWER" = Y ] || { echo "nothing created"; exit 0; }

SG="$(aws ec2 create-security-group --region "$REGION" --vpc-id "$VPC" \
      --group-name "$NAME-$(date +%s)" \
      --description "J-lens auth proxy: public TLS, admin SSH only" \
      --query GroupId --output text)"
say "security group $SG"
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
  --ip-permissions \
    "IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0,Description=https}]" \
    "IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0,Description=acme-http-01}]" \
    "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$MY_IP/32,Description=admin}]" \
  >/dev/null
say "ingress 443, 80, 22/$MY_IP"

KEY_NAME="$NAME-$(date +%s)"
umask 077
aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" \
  --query KeyMaterial --output text > "$KEY_OUT"
chmod 600 "$KEY_OUT"
say "key pair $KEY_NAME -> $KEY_OUT"

USER_DATA="$(mktemp)"
trap 'rm -f "$USER_DATA"' EXIT
sed -e "s|^      JLENS_HOSTNAME=.*|      JLENS_HOSTNAME=$HOSTNAME_ARG|" \
    -e "s|^      JLENS_ACME_EMAIL=.*|      JLENS_ACME_EMAIL=$EMAIL_ARG|" \
    -e "s|^      JLENS_PUBLIC_URL=.*|      JLENS_PUBLIC_URL=https://$HOSTNAME_ARG|" \
    "$DEPLOY_DIR/cloud-init.yaml" > "$USER_DATA"

# From here on, a failure has already cost money. Shout the id on the way out
# rather than leaving an untracked instance billing.
set +e
ID="$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG" \
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=8,VolumeType=gp3,Encrypted=true}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
  --user-data "file://$USER_DATA" \
  --query 'Instances[0].InstanceId' --output text 2>&1)"
RC=$?
set -e
if [ "$RC" -ne 0 ] || [[ "$ID" != i-* ]]; then
  echo "run-instances failed: $ID" >&2
  echo "Check for an untagged instance before retrying:" >&2
  echo "  aws ec2 describe-instances --region $REGION --filters Name=tag:Name,Values=$NAME" >&2
  exit 1
fi
echo "$ID" > "$STATE_FILE"
say "instance $ID (recorded in $STATE_FILE)"

trap 'echo "INTERRUPTED. Instance $ID is running and billing. Terminate with: aws ec2 terminate-instances --region '"$REGION"' --instance-ids '"$ID" EXIT

aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
ALLOC="$(aws ec2 allocate-address --region "$REGION" --domain vpc --query AllocationId --output text)"
aws ec2 associate-address --region "$REGION" --instance-id "$ID" --allocation-id "$ALLOC" >/dev/null
EIP="$(aws ec2 describe-addresses --region "$REGION" --allocation-ids "$ALLOC" \
       --query 'Addresses[0].PublicIp' --output text)"
trap - EXIT
rm -f "$USER_DATA"

cat <<EOF

Created. The box is up and deliberately NOT serving yet.

  instance   $ID
  address    $EIP
  ssh        ssh -i $KEY_OUT ubuntu@$EIP

Remaining steps, none of which this script can do for you:

  1. DNS: point $HOSTNAME_ARG at $EIP (A record). Caddy cannot get a
     certificate until that resolves, so do this first.

  2. Register a GitHub OAuth app at https://github.com/settings/developers
       Homepage            https://$HOSTNAME_ARG
       Callback URL        https://$HOSTNAME_ARG/_jlens/callback
     Request no scopes. Note the client id, generate a client secret.

  3. On the box, fill in the blanks:
       sudo editor /etc/default/jlens
     JLENS_OAUTH_CLIENT_ID, JLENS_OAUTH_CLIENT_SECRET, JLENS_GITHUB_TOKEN
     (fine-grained PAT, Administration: Read), and
       JLENS_SESSION_SECRET=\$(openssl rand -hex 32)
     Decide JLENS_AUTH_REPOS - the default admits jacobian-lens collaborators
     only, which today is one person. See deploy/WEB-DEPLOY.md.

  4. Hand it the vast.ai credentials so it can reach a rented GPU:
       sudo install -m 0640 -o root -g jlens <(pbpaste) /etc/jlens/vast_api_key
       sudo install -m 0640 -o root -g jlens ~/.ssh/id_ed25519 /etc/jlens/id_ed25519
       echo "\$INSTANCE_ID" | sudo tee /etc/jlens/instance

  5. Start it:
       sudo systemctl enable --now jlens-proxy jlens-tunnel caddy
       curl -sS https://$HOSTNAME_ARG/_jlens/health

  6. Set a budget alarm (deploy/WEB-DEPLOY.md has the CLI). This box has no
     destroy.sh and no idle reaper - it is meant to stay up.

Tear it all down with: $DEPLOY_DIR/destroy-proxy.sh
EOF
