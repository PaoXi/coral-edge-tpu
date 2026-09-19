#!/usr/bin/env bash
# Coral USB Accelerator setup, for systems without Google's apt repo (Arch, etc).
#
# Installs the libedgetpu runtime library and creates a Python 3.9 environment
# running Coral's own tflite-runtime 2.5.0, which is the version that matches
# libedgetpu 16.0. Newer runtimes segfault or fail at invoke(); see README.
#
# Requires: curl, bsdtar, python3 (for nothing), uv or python3.9, sudo.

set -euo pipefail

REPO="https://packages.cloud.google.com/apt"
POOL="pool/coral-edgetpu-stable"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }; }
need curl
need bsdtar
need sudo

echo "==> fetching package index"
curl -sfL "$REPO/dists/coral-edgetpu-stable/main/binary-amd64/Packages" -o "$WORK/Packages"

deb_filename() {
  awk -v pkg="$1" '
    $1=="Package:" && $2==pkg { found=1; next }
    found && $1=="Filename:" { print $2; exit }
  ' "$WORK/Packages"
}

fetch_deb() {
  local pkg="$1" f
  f="$(deb_filename "$pkg")"
  [ -n "$f" ] || { echo "package not found in repo: $pkg"; exit 1; }
  curl -sfL "$REPO/$f" -o "$WORK/$pkg.deb"
  mkdir -p "$WORK/$pkg"
  bsdtar -xf "$WORK/$pkg.deb" -C "$WORK/$pkg"
  tar -xf "$WORK/$pkg/data.tar.xz" -C "$WORK/$pkg"
}

echo "==> downloading packages"
fetch_deb libedgetpu1-std
fetch_deb python3-tflite-runtime

echo "==> installing runtime library"
sudo install -Dm755 \
  "$WORK/libedgetpu1-std/usr/lib/x86_64-linux-gnu/libedgetpu.so.1.0" \
  /usr/lib/libedgetpu.so.1.0
sudo ln -sf libedgetpu.so.1.0 /usr/lib/libedgetpu.so.1

echo "==> installing udev rule"
# Arch has no plugdev group, so use a permissive mode instead of GROUP="plugdev".
sudo tee /etc/udev/rules.d/60-libedgetpu1-std.rules >/dev/null <<'EOF'
SUBSYSTEM=="usb",ATTRS{idVendor}=="1a6e",ATTRS{idProduct}=="089a",MODE="0666"
SUBSYSTEM=="usb",ATTRS{idVendor}=="18d1",ATTRS{idProduct}=="9302",MODE="0666"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "==> creating Python 3.9 environment"
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.9 .venv
  uv pip install --python .venv/bin/python "numpy<2"
else
  python3.9 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet "numpy<2"
fi

SP=".venv/lib/python3.9/site-packages"
[ -d "$SP" ] || { echo "unexpected venv layout, no $SP"; exit 1; }

echo "==> installing Coral's tflite-runtime into the venv"
cp -r "$WORK/python3-tflite-runtime/usr/lib/python3/dist-packages/tflite_runtime" "$SP/"

echo
echo "Done. Now:"
echo "  1. plug the Accelerator in (use a USB 3 port and a USB 3 cable)"
echo "  2. ./.venv/bin/python benchmark.py"