#!/usr/bin/env bash
# =============================================================================
# MAESTRO-FL — Environment Setup Script
# =============================================================================
# Usage: bash setup.sh
# Supports macOS/Homebrew and Debian/Ubuntu Linux.
# =============================================================================

set -euo pipefail

echo "============================================="
echo "  MAESTRO-FL — Environment Setup"
echo "============================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

detect_shell_rc() {
    if [[ "${SHELL:-}" == *zsh* ]]; then
        echo "$HOME/.zshrc"
    elif [[ "${SHELL:-}" == *bash* ]]; then
        echo "$HOME/.bashrc"
    else
        echo "$HOME/.profile"
    fi
}

append_once() {
    local line="$1"
    local file="$2"
    touch "$file"
    grep -qxF "$line" "$file" || echo "$line" >> "$file"
}

configure_sumo_home() {
    local sumo_bin_dir=""
    if [[ -n "${SUMO_HOME:-}" && -d "$SUMO_HOME" && -d "$SUMO_HOME/tools" ]]; then
        if command -v sumo >/dev/null 2>&1; then
            sumo_bin_dir="$(dirname "$(command -v sumo)")"
        fi
        return
    fi

    if [[ "$(uname -s)" == "Darwin" ]]; then
        if [[ -d "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo" ]]; then
            export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo"
            sumo_bin_dir="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/bin"
        elif [[ -d "/Library/Frameworks/EclipseSUMO.framework/Versions/1.27.1/EclipseSUMO/share/sumo" ]]; then
            export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/1.27.1/EclipseSUMO/share/sumo"
            sumo_bin_dir="/Library/Frameworks/EclipseSUMO.framework/Versions/1.27.1/EclipseSUMO/bin"
        fi

        if command -v brew >/dev/null 2>&1; then
            local brew_prefix
            brew_prefix="$(brew --prefix sumo 2>/dev/null || true)"
            if [[ -z "${SUMO_HOME:-}" && -n "$brew_prefix" && -d "$brew_prefix/share/sumo" ]]; then
                export SUMO_HOME="$brew_prefix/share/sumo"
            elif [[ -z "${SUMO_HOME:-}" && -d "/opt/homebrew/opt/sumo/share/sumo" ]]; then
                export SUMO_HOME="/opt/homebrew/opt/sumo/share/sumo"
            elif [[ -z "${SUMO_HOME:-}" && -d "/usr/local/opt/sumo/share/sumo" ]]; then
                export SUMO_HOME="/usr/local/opt/sumo/share/sumo"
            fi
        fi
    elif [[ -d "/usr/share/sumo" ]]; then
        export SUMO_HOME="/usr/share/sumo"
    fi

    if [[ -z "${SUMO_HOME:-}" || ! -d "$SUMO_HOME" ]]; then
        echo "  ✗ Could not detect SUMO_HOME after installation."
        echo "    Set it manually, for example: export SUMO_HOME=/usr/share/sumo"
        exit 1
    fi

    local shell_rc
    shell_rc="$(detect_shell_rc)"
    append_once "export SUMO_HOME=\"$SUMO_HOME\"" "$shell_rc"
    if [[ -n "$sumo_bin_dir" ]]; then
        append_once "export PATH=\"$sumo_bin_dir:\$PATH\"" "$shell_rc"
        export PATH="$sumo_bin_dir:$PATH"
    else
        append_once 'export PATH="$SUMO_HOME/bin:$PATH"' "$shell_rc"
        export PATH="$SUMO_HOME/bin:$PATH"
    fi
    append_once 'export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"' "$shell_rc"
    export PYTHONPATH="$SUMO_HOME/tools:${PYTHONPATH:-}"
    echo "  ✓ SUMO_HOME=$SUMO_HOME"
}

echo ""
echo "[1/5] Installing SUMO traffic simulator..."
if command -v sumo >/dev/null 2>&1; then
    echo "  ✓ SUMO already installed: $(sumo --version 2>&1 | head -1)"
elif [[ "$(uname -s)" == "Darwin" && -x "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/bin/sumo" ]]; then
    export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/share/sumo"
    export PATH="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/bin:$PATH"
    echo "  ✓ SUMO pkg install found: $(sumo --version 2>&1 | head -1)"
elif [[ "$(uname -s)" == "Darwin" && -x "/Library/Frameworks/EclipseSUMO.framework/Versions/1.27.1/EclipseSUMO/bin/sumo" ]]; then
    export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/1.27.1/EclipseSUMO/share/sumo"
    export PATH="/Library/Frameworks/EclipseSUMO.framework/Versions/1.27.1/EclipseSUMO/bin:$PATH"
    echo "  ✓ SUMO pkg install found: $(sumo --version 2>&1 | head -1)"
else
    case "$(uname -s)" in
        Darwin)
            if ! command -v brew >/dev/null 2>&1; then
                echo "  ✗ Homebrew is required on macOS."
                echo '    Install it with: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                exit 1
            fi
            echo "  Installing SUMO via Homebrew..."
            brew install sumo
            ;;
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                echo "  Installing SUMO via apt..."
                sudo add-apt-repository -y ppa:sumo/stable || true
                sudo apt-get update
                sudo apt-get install -y sumo sumo-tools sumo-doc
            else
                echo "  ✗ Unsupported Linux package manager. Install SUMO manually."
                exit 1
            fi
            ;;
        *)
            echo "  ✗ Unsupported OS: $(uname -s)"
            exit 1
            ;;
    esac
fi
configure_sumo_home

echo ""
echo "[2/5] Creating Python virtual environment..."
VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    echo "  ✓ Created $VENV_DIR"
else
    echo "  ✓ Reusing $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo ""
echo "[3/5] Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "[4/5] Creating project directories..."
mkdir -p sumo_env/logs sumo_env/network models results/plots eval/results/plots
echo "  ✓ Directories ready"

echo ""
echo "[5/5] Verifying installation..."
python - <<'PY'
import importlib
import os
import sys

sumo_home = os.environ.get("SUMO_HOME")
if sumo_home:
    sys.path.append(os.path.join(sumo_home, "tools"))

modules = [
    "gymnasium",
    "stable_baselines3",
    "torch",
    "flwr",
    "pandas",
    "matplotlib",
    "traci",
    "sumolib",
]

failed = []
for name in modules:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "ok")
        print(f"  ✓ {name} {version}")
    except Exception as exc:
        failed.append(f"{name}: {exc}")

if failed:
    print("\n  Missing or broken modules:")
    for item in failed:
        print(f"  ✗ {item}")
    raise SystemExit(1)

print("\n  All checks passed.")
PY

echo ""
echo "============================================="
echo "  Setup complete. Activate with:"
echo "    source .venv/bin/activate"
echo "============================================="
