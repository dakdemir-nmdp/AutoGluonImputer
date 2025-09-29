#!/bin/bash
# Quick setup script for new contributors using uv

set -e

echo "🚀 Quick setup for AutoFillGluon contributors"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed."
    echo ""
    echo "Install uv with one of these methods:"
    echo "  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Windows:     powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
    echo "  pip:         pip install uv"
    echo ""
    echo "For more options, visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✅ uv is available"
echo ""

# Create virtual environment and install dependencies
echo "📦 Creating virtual environment and installing dependencies..."
uv venv
uv pip install -e ".[dev,viz,examples]"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Or use uv run to execute commands:"
echo "     uv run python examples/basic/simple_imputation.py"
echo "     uv run pytest tests/"
echo "     uv run jupyter lab"
echo ""
echo "  3. View available make targets:"
echo "     make help"