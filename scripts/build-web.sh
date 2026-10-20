#!/usr/bin/env bash
# Build the wasm package into pkg/ (wasm-bindgen web target).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v wasm-bindgen >/dev/null || {
  echo "wasm-bindgen not found: cargo install wasm-bindgen-cli --version 0.2.129 --locked" >&2
  exit 1
}

cargo build --target wasm32-unknown-unknown --release
rm -rf pkg
wasm-bindgen --target web --out-dir pkg target/wasm32-unknown-unknown/release/ocr.wasm
