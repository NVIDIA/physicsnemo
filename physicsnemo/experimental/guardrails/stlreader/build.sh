#!/bin/bash
# Quick build script for the fast STL reader

set -e

echo "============================================================"
echo "Building Fast STL Reader (Rust)"
echo "============================================================"
echo ""

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "✗ Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    echo "✓ Rust installed"
else
    echo "✓ Rust found: $(rustc --version)"
fi

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "✗ maturin not found. Installing..."
    pip install maturin
    echo "✓ maturin installed"
else
    echo "✓ maturin found: $(maturin --version)"
fi

echo ""
echo "Building extension module (optimized)..."
maturin develop --release

echo ""
echo "============================================================"
echo "Build Complete!"
echo "============================================================"
echo ""
echo "Testing import..."
if python3 -c "import stlreader; print('✓ stlreader module imported successfully')" 2>&1; then
    echo ""
    echo "✓ Installation successful!"
    echo ""
    echo "You can now use the fast reader in PhysicsNemo:"
    echo "  from physicsnemo.experimental.guardrails.geometry import is_fast_reader_available"
    echo "  print(is_fast_reader_available())  # Should return True"
else
    echo ""
    echo "✗ Import failed. Check the error messages above."
    exit 1
fi
