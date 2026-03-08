#!/usr/bin/env bash
# deepghs-mcp installer
# Creates a virtual environment and installs all dependencies.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "🤗 deepghs-mcp installer"
echo "========================"

# Check Python version
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(sys.version_info >= (3, 10))" 2>/dev/null)
        if [ "$version" = "True" ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Python 3.10+ is required but was not found."
    echo "   Install from: https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Found Python: $($PYTHON_CMD --version)"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment at .venv/ ..."
"$PYTHON_CMD" -m venv "$VENV_DIR"

# Install dependencies
echo "📥 Installing dependencies..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "✅ Installation complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Next: add this to claude_desktop_config.json"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo '{'
echo '  "mcpServers": {'
echo '    "deepghs": {'
echo "      \"command\": \"$VENV_DIR/bin/python\","
echo "      \"args\": [\"$SCRIPT_DIR/deepghs_mcp.py\"],"
echo '      "env": {'
echo '        "HF_TOKEN": "your_hf_token_here"'
echo '      }'
echo '    }'
echo '  }'
echo '}'
echo ""
echo "  Get your HF token at: https://huggingface.co/settings/tokens"
echo "  (Optional — needed only for gated/private repos)"
echo ""
