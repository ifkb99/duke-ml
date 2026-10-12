#!/bin/sh
# App icons for the web manifest, from the two SVGs here into ../public.
# Rendered by a headless Chrome (ImageMagick drops the gradients):
#   CHROME=/path/to/chrome ./make.sh
# Run after changing an SVG; the PNGs are committed.
set -e
cd "$(dirname "$0")"
CHROME=${CHROME:-chromium}
png() {
  "$CHROME" --headless --no-sandbox --hide-scrollbars --default-background-color=00000000 \
    --window-size="$2,$2" --screenshot="$PWD/../public/$3" "file://$PWD/$1" 2>/dev/null
}
png icon.svg 192 icon-192.png
png icon.svg 512 icon-512.png
# Maskable: the launcher crops it to its own shape, so the crown stays inside the middle 80%
png icon-maskable.svg 512 icon-maskable-512.png
# iOS rounds the corners itself and shows transparency as black
png icon-maskable.svg 180 apple-touch-icon.png
