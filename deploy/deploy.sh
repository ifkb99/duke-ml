#!/bin/sh
# Runs the duke web app image GitHub Actions built:
#   deploy.sh <commit sha>|latest
#
# GitHub Actions calls it over SSH. If the Actions key is locked to this script
# with a forced command (deploy/README.md), sshd puts the requested command in
# SSH_ORIGINAL_COMMAND: only its last word, the tag, is used, and anything that
# isn't a commit sha or "latest" is refused.
set -eu
cd "$(dirname "$0")"

if [ -n "${SSH_ORIGINAL_COMMAND:-}" ]; then
  tag=${SSH_ORIGINAL_COMMAND##* }
else
  tag=${1:-}
fi
if [ "$tag" != latest ]; then
  case "$tag" in
    ''|*[!0-9a-f]*) echo "usage: deploy.sh <commit sha>|latest" >&2; exit 2 ;;
  esac
  [ "${#tag}" -eq 40 ] || { echo "not a commit sha: $tag" >&2; exit 2; }
fi

export DUKE_TAG="$tag"
docker compose pull --quiet
docker compose up -d --remove-orphans

# Drop older duke images (other images on the server are left alone)
docker images ghcr.io/ifkb99/duke-ml --format '{{.Repository}}:{{.Tag}}' \
  | grep -v -e ":$tag\$" -e ':latest$' | xargs -r docker rmi >/dev/null 2>&1 || true

echo "duke deployed: $tag"
