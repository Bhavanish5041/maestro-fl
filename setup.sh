#!/usr/bin/env bash
# =============================================================================
# MAESTRO-FL — Environment Setup Script
# =============================================================================
# Run this once on a fresh machine to get everything working.
# Usage: bash setup.sh
# =============================================================================

set -e  # Exit on error

echo "============================================="
echo "  MAESTRO-FL — Environment Setup"
echo "============================================="

# ---------------------------------------------------------------------------
# 1. System dependencies (SUMO)
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Installing SUMO traffic simulator..."

if command -v sumo &> /dev/null; then
    echo "  ✓ SUMO already installed: $(sumo --version 2>&1 | head -1)"
else
    echo "  Installing SUMO via apt..."
    sudo add-apt-repository -y ppa:sumo/stable
    sudo apt-get update -qq
    sudo apt-get install -y sumo sumo-tools sumo-doc
    echo "  ✓ SUMO installed."
fi

# Set SUMO_HOME if not already set
if [ -z "$SUMO_HOME" ]; then
    export SUMO_HOME="/usr/share/sumo"
    echo "export SUMO_HOME=\"/usr/share/sumo\"" >> ~/.bashrc
    echo "  Set SUMO_HOME=$SUMO_HOME"
fi

# Verify TraCI is accessible
echo "  Checking TraCI Python bindings..."
python3 -c "import traci; print(f'  ✓ TraCI available: {traci.__file__}')" 2>/dev/null || {
    echo "  Adding SUMO tools to PYTHONPATH..."
    export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
    echo "export PYTHONPATH=\"\$SUMO_HOME/tools:\$PYTHONPATH\"" >> ~/.bashrc
    python3 -c "import traci; print(f'  ✓ TraCI available: {traci.__file__}')"
}

# ---------------------------------------------------------------------------
# 2. Python virtual environment
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Setting up Python virtual environment..."

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  ✓ Created virtual environment at $VENV_DIR"
else
    echo "  ✓ Virtual environment already exists at $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip -q

# ---------------------------------------------------------------------------
# 3. Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Installing Python dependencies..."
pip install -r requirements.txt -q
echo "  ✓ All Python packages installed."

# ---------------------------------------------------------------------------
# 4. Create necessary directories
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Creating project directories..."

mkdir -p sumo_env/logs
mkdir -p sumo_env/network
mkdir -p rl_agent/models
mkdir -p prediction
mkdir -p eval/results/plots
mkdir -p federated

echo "  ✓ Directory structure ready."

# ---------------------------------------------------------------------------
# 5. Verify installation
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Verifying installation..."

python3 -c "
import sys
errors = []

# Check core imports
try:
    import gymnasium; print(f'  ✓ gymnasium {gymnasium.__version__}')
except ImportError as e:
    errors.append(f'  ✗ gymnasium: {e}')

try:
    import stable_baselines3; print(f'  ✓ stable-baselines3 {stable_baselines3.__version__}')
except ImportError as e:
    errors.append(f'  ✗ stable-baselines3: {e}')

try:
    import torch; print(f'  ✓ torch {torch.__version__}')
except ImportError as e:
    errors.append(f'  ✗ torch: {e}')

try:
    import flwr; print(f'  ✓ flwr {flwr.__version__}')
except ImportError as e:
    errors.append(f'  ✗ flwr: {e}')

try:
    import pandas; print(f'  ✓ pandas {pandas.__version__}')
except ImportError as e:
    errors.append(f'  ✗ pandas: {e}')

try:
    import matplotlib; print(f'  ✓ matplotlib {matplotlib.__version__}')
except ImportError as e:
    errors.append(f'  ✗ matplotlib: {e}')

try:
    import traci; print(f'  ✓ traci available')
except ImportError as e:
    errors.append(f'  ✗ traci: {e}')

# Check MAESTRO-FL modules
try:
    from shared.schema import TRAFFIC_LOG_COLUMNS; print(f'  ✓ shared.schema')
except ImportError as e:
    errors.append(f'  ✗ shared.schema: {e}')

if errors:
    print('\n  WARNINGS:')
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print('\n  All checks passed!')
"

echo ""
echo "============================================="
echo "  Setup complete! Activate the venv with:"
echo "    source .venv/bin/activate"
echo ""
echo "  Then generate your SUMO network:"
echo "    cd \$SUMO_HOME/tools && python osmWebWizard.py"
echo "============================================="
