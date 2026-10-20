#!/usr/bin/env bash
# Environment setup for Jules.
#
# Installs the Rust toolchain and Node 24, then warms the cargo and npm caches.
# Idempotent: safe to re-run. This is setup, not part of any change - do not commit
# anything it produces (everything it writes is already git-ignored).

set -euo pipefail

log() { printf '\n==> %s\n' "$*"; }

# --- Rust -------------------------------------------------------------------
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  log "Installing Rust (minimal profile)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal
fi
# shellcheck disable=SC1091
. "$HOME/.cargo/env"
export PATH="$HOME/.cargo/bin:$PATH"

# --- Node 24 ----------------------------------------------------------------
need_node=1
if command -v node >/dev/null 2>&1; then
  major="$(node -p 'process.versions.node.split(".")[0]')"
  if [ "$major" -ge 24 ]; then need_node=0; fi
fi
if [ "$need_node" -eq 1 ]; then
  log "Installing Node 24"
  curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

# --- Dependencies -----------------------------------------------------------
# npm ci runs the root postinstall (install-electron); the desktop runtime is
# needed by build/typecheck and the packaged smoke test, not by `npm test`.
log "Installing npm dependencies"
npm ci

log "Fetching Rust crates"
cargo fetch --locked

# --- Report -----------------------------------------------------------------
log "Versions"
rustc --version
cargo --version
node --version
npm --version

log "Done. Next: cargo test --workspace"
