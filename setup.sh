#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

echo "==> Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing dependencies"
pip install \
    torch \
    transformers \
    accelerate \
    scikit-learn \
    matplotlib \
    anthropic \
    streamlit \
    umap-learn \
    numpy

echo ""
echo "==> Verifying GPU"
python - <<'EOF'
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, VRAM: {props.total_memory / 1e9:.1f} GB")
else:
    print("WARNING: CUDA not available — model will run on CPU (very slow)")
EOF

echo ""
echo "==> HuggingFace login (required to download Gemma — accept the licence first)"
echo "    Run: huggingface-cli login"
echo ""
echo "Setup complete. Activate the environment with:"
echo "    source $VENV_DIR/bin/activate"
