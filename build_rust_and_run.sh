#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source Cargo env if installed in user profile
if [ -f "$HOME/.cargo/env" ]; then
    . "$HOME/.cargo/env"
fi

export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

echo "=========================================================="
echo " 1. Building Compiled Rust Core (security_core_rs via PyO3)"
echo "=========================================================="
pip install --user maturin 2>/dev/null || pip install maturin
cd rust_core
maturin build --release
pip install --user target/wheels/*.whl 2>/dev/null || pip install target/wheels/*.whl
cd ..

echo "=========================================================="
echo " 2. Starting Hybrid Rust + GPU Video Analysis Server"
echo "=========================================================="
python3 main.py
