#!/bin/sh
# Runs the duke web app image GitHub Actions built:
#   deploy.sh <commit sha>|latest
#
# The new version replaces whatever serves the port now. If it doesn't start, or
# doesn't answer, the previous version is put back and the script fails.
#
# GitHub Actions calls it over SSH. If the Actions key is locked to this script
# with a forced command (deploy/README.md), sshd puts the requested command in
# SSH_ORIGINAL_COMMAND: only its last word, the tag, is used, and anything that
# isn't a commit sha or "latest" is refused.
set -eu
cd "$(dirname "$0")"

REPO=ghcr.io/ifkb99/duke-ml
PORT=3001   # as in compose.yml

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

# Fetch first: if this fails, nothing has changed
export DUKE_TAG="$tag"
docker compose pull --quiet

# What serves the site now, to go back to. Ours: its image, kept as :rollback
# (":latest" has just moved to the new one). Anyone else's container on the
# port (one built by hand, say): stopped, and removed only once the new one answers.
ours=$(docker compose ps -q --status running duke)
previous=
if [ -n "$ours" ]; then
  previous=$(docker inspect --format '{{.Image}}' "$ours")
  docker tag "$previous" "$REPO:rollback"
fi
others=$(docker ps -q --no-trunc --filter "publish=$PORT" | grep -vxF -e "$ours" || true)

# The site answers within 20 s (without curl: the container is still running after 3 s)
healthy() {
  if ! command -v curl >/dev/null; then
    sleep 3
    [ "$(docker inspect --format '{{.State.Running}}' "$(docker compose ps -q duke)" 2>/dev/null)" = true ]
    return
  fi
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    curl -fsS -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null && return 0
    sleep 2
  done
  return 1
}

# shellcheck disable=SC2086  # $others is a list of container ids
start() {
  [ -z "$others" ] || docker stop $others >/dev/null
  docker compose up -d --remove-orphans
}

started=
if log=$(start 2>&1); then
  started=1
fi
if [ -n "$started" ] && healthy; then
  # shellcheck disable=SC2086
  [ -z "$others" ] || docker rm $others >/dev/null
  # Drop older duke images (other images on the server are left alone)
  docker images "$REPO" --format '{{.Repository}}:{{.Tag}}' \
    | grep -v -e ":$tag\$" -e ':latest$' | xargs -r docker rmi >/dev/null 2>&1 || true
  echo "duke deployed: $tag"
  exit 0
fi

# Failed: say why, then put back what was running
echo "$log" >&2
if [ -n "$started" ]; then
  docker compose logs --no-log-prefix --tail 15 duke >&2 || true
  why="it started but did not answer on port $PORT"
  last=$(docker compose logs --no-log-prefix --tail 1 duke 2>&1)
  [ -z "$last" ] || why="$why; its last words: $last"
else
  why=$(echo "$log" | tail -n 1)
fi
if [ -n "$previous" ]; then
  DUKE_TAG=rollback docker compose up -d --remove-orphans >/dev/null 2>&1 || true
else
  docker compose down >/dev/null 2>&1 || true
  # shellcheck disable=SC2086
  [ -z "$others" ] || docker start $others >/dev/null 2>&1 || true
fi
if [ -z "$previous$others" ]; then
  back="Nothing was running before, so the site is down."
elif healthy; then
  back="The previous version is back up."
else
  back="The previous version did not come back either: the site is down."
fi
echo "duke $tag did not come up: $why. $back" >&2
exit 1
