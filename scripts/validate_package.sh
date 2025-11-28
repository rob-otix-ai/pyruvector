#!/bin/bash
# Package validation script for pyruvector

set -e

echo "=== Pyruvector Package Validation ==="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "✓ Python version is 3.8+"
else
    echo "✗ Python version must be 3.8 or higher"
    exit 1
fi
echo ""

# Check required files
echo "2. Checking required files..."
required_files=(
    "README.md"
    "LICENSE"
    "Cargo.toml"
    "pyproject.toml"
    "MANIFEST.in"
    "python/pyruvector/__init__.py"
    "python/pyruvector/_pyruvector.pyi"
    "python/pyruvector/py.typed"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ Missing: $file"
        exit 1
    fi
done
echo ""

# Validate Cargo.toml
echo "3. Validating Cargo.toml..."
if grep -q 'crate-type = \["cdylib"\]' Cargo.toml; then
    echo "✓ crate-type is set to cdylib"
else
    echo "✗ crate-type must be cdylib for PyO3"
    exit 1
fi

if grep -q 'pyo3 = { version = "0.20"' Cargo.toml; then
    echo "✓ PyO3 dependency found"
else
    echo "✗ PyO3 dependency not configured correctly"
    exit 1
fi
echo ""

# Validate pyproject.toml
echo "4. Validating pyproject.toml..."
if grep -q 'build-backend = "maturin"' pyproject.toml; then
    echo "✓ Maturin build backend configured"
else
    echo "✗ Maturin build backend not configured"
    exit 1
fi

if grep -q 'requires-python = ">=3.8"' pyproject.toml; then
    echo "✓ Python version requirement set"
else
    echo "✗ Python version requirement not set correctly"
    exit 1
fi

if grep -q 'bindings = "pyo3"' pyproject.toml; then
    echo "✓ PyO3 bindings configured"
else
    echo "✗ PyO3 bindings not configured"
    exit 1
fi
echo ""

# Check version consistency
echo "5. Checking version consistency..."
cargo_version=$(grep '^version = ' Cargo.toml | head -1 | cut -d'"' -f2)
pyproject_version=$(grep '^version = ' pyproject.toml | head -1 | cut -d'"' -f2)

echo "Cargo.toml version: $cargo_version"
echo "pyproject.toml version: $pyproject_version"

if [ "$cargo_version" = "$pyproject_version" ]; then
    echo "✓ Versions match"
else
    echo "✗ Version mismatch between Cargo.toml and pyproject.toml"
    exit 1
fi
echo ""

# Check for common issues
echo "6. Checking for common issues..."
if grep -q 'keywords = \[' pyproject.toml && grep -q 'classifiers = \[' pyproject.toml; then
    echo "✓ PyPI metadata (keywords, classifiers) present"
else
    echo "⚠ Warning: PyPI metadata may be incomplete"
fi

if [ -f "LICENSE" ]; then
    echo "✓ LICENSE file present"
else
    echo "✗ LICENSE file missing"
    exit 1
fi

if grep -q 'MIT' LICENSE; then
    echo "✓ MIT License detected"
fi
echo ""

# Summary
echo "=== Validation Summary ==="
echo "✓ All required files present"
echo "✓ Configuration files valid"
echo "✓ Version consistency verified"
echo "✓ Package is ready for building"
echo ""
echo "Next steps:"
echo "1. Install maturin: pip install maturin"
echo "2. Build package: maturin build --release"
echo "3. Test installation: pip install target/wheels/*.whl"
echo "4. Publish to Test PyPI: maturin publish --repository testpypi"
echo "5. Publish to PyPI: maturin publish"
