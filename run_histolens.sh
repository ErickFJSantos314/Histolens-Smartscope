#!/usr/bin/env bash
set -euo pipefail

# HistoLens one-click launcher for Linux/macOS
# - Activates conda environment
# - Validates required runtime variables
# - Starts microscope digital twin client

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available in this shell."
  echo "        Initialize conda first (e.g., source ~/miniconda3/etc/profile.d/conda.sh)."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate histolens

missing=()
[[ -z "${GOOGLE_API_KEY:-}" ]] && missing+=("GOOGLE_API_KEY")
[[ -z "${REMOTE_VISION_API_URL:-}" && -z "${COLAB_API_URL:-}" ]] && missing+=("REMOTE_VISION_API_URL")
[[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]] && missing+=("GOOGLE_APPLICATION_CREDENTIALS")

if (( ${#missing[@]} > 0 )); then
  echo "[ERROR] Missing required environment variables: ${missing[*]}"
  echo
  echo "Suggested setup:"
  echo "  export GOOGLE_API_KEY='your_google_api_key'"
  echo "  export REMOTE_VISION_API_URL='https://your-ngrok-url.ngrok-free.dev/analyze'"
  echo "  export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account.json'"
  exit 1
fi

echo "[INFO] Launching HistoLens SmartScope AI..."
python histolens.py
