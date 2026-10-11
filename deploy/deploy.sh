#!/bin/sh
# Runs the duke web app image GitHub Actions built. This is the forced command
# of the Actions deploy key (deploy/README.md), so that key can do nothing else:
# it sends "deploy <commit sha>" (or "deploy latest") and anything else is refused.
#
# By hand: SSH_ORIGINAL_COMMAND="deploy latest" ./deploy.sh
set -eu
cd "$(dirname "$0")"

case "${SSH_ORIGINAL_COMMAND:-}" in
  "deploy latest") tag=latest ;;
  "deploy "*) tag=${SSH_ORIGINAL_COMMAND#deploy } ;;
  *) echo "usage: deploy <commit sha>|latest" >&2; exit 2 ;;
esac
if [ "$tag" != latest ]; then
  case "$tag" in
    *[!0-9a-f]*) echo "not a commit sha: $tag" >&2; exit 2 ;;
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
