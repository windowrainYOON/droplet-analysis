#!/usr/bin/env bash
# Build the macOS app (Apple Silicon), install it to /Applications and put the .dmg on the Desktop.
# Usage: scripts/build_mac.sh            (from the repository root)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null || PY=python3
if [ ! -d .venv-build ]; then
  "$PY" -m venv .venv-build
fi
source .venv-build/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q "briefcase>=0.3.20" pillow

briefcase create macOS app --no-input
# create skips an app that already exists; always copy the current code and requirements
briefcase update macOS app --update-requirements --update-resources --no-input
briefcase build macOS app --no-input
briefcase package macOS app --adhoc-sign --no-input

APP=$(ls -d build/droplet_map/macos/app/*.app | head -1)
DMG=$(ls -t dist/*.dmg | head -1)

# replace the installed copy
if [ -d "/Applications/Droplet Map.app" ]; then
  rm -rf "/Applications/Droplet Map.app"
fi
ditto "$APP" "/Applications/Droplet Map.app"
xattr -dr com.apple.quarantine "/Applications/Droplet Map.app" 2>/dev/null || true

# headless self-test of the installed app (synthetic data, ~1 min)
"/Applications/Droplet Map.app/Contents/MacOS/Droplet Map" --selftest || echo "WARNING: selftest failed"

# DMG name: <app>-<version>-<commit>_apple silicon.dmg (Intel Macs: _intel)
VERSION=$(python -c "import re;print(re.search(r'^__version__ = \"(.+)\"', open('src/droplet_map/__init__.py').read(), re.M).group(1))")
COMMIT=$(git rev-parse --short HEAD)
ARCH=$( [ "$(uname -m)" = "arm64" ] && echo "apple silicon" || echo "intel" )
OUT="$HOME/Desktop/Droplet Map-${VERSION}-${COMMIT}_${ARCH}.dmg"
cp -f "$DMG" "$OUT"

echo "Installed: /Applications/Droplet Map.app"
echo "DMG:       $OUT"
