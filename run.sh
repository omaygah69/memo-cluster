#!/bin/bash
# Exit immediately if any command fails
set -e

VENV_PATH="./myvenv"

# --- Functions ---
activate_venv() {
    if [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate"
        echo "[INFO] Virtual environment activated"
    else
        echo "[ERROR] Virtual environment not found at $VENV_PATH"
        echo "Run 'python -m venv venv' first."
        exit 1
    fi
}

run_pipeline() {
    activate_venv
    echo "➡️ Running clean_memos.py ..."
    python docextractor.py
    echo "➡️ Running cluster_memos.py ..."
    python cluster.py
    echo "[INFO] Pipeline finished successfully!"
}

freeze_packages() {
    activate_venv
    echo "📦 Freezing dependencies into requirements.txt ..."
    pip freeze > requirements.txt
    echo "[INFO] requirements.txt updated"
}

install_packages() {
    activate_venv
    echo "[INSTALL] Installing dependencies from requirements.txt ..."
    pip install -r requirements.txt
    echo "[INFO] Packages installed"
}

# --- Main ---
case "$1" in
    run)
        run_pipeline
        ;;
    freeze)
        freeze_packages
        ;;
    install)
        install_packages
        ;;
    *)
        echo "Usage: $0 {run|freeze|install}"
        exit 1
        ;;
esac
