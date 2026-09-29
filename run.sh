#!/usr/bin/env bash
# One command: set up (if needed), train (if needed), then open the dashboard.
#   ./run.sh            # reuse existing environment and models
#   ./run.sh --retrain  # force retraining before launching
set -euo pipefail
cd "$(dirname "$0")"

[[ -x .venv/bin/python ]] || ./setup.sh

if [[ "${1:-}" == "--retrain" || ! -f artifacts/models/meta.json ]]; then
  .venv/bin/python train.py
fi

URL=http://localhost:8501
echo "Starting dashboard at $URL  (Ctrl+C to stop)"
# Open the browser once the server responds.
( for _ in $(seq 60); do curl -s -o /dev/null "$URL" && { open "$URL" 2>/dev/null || xdg-open "$URL"; break; }; sleep 1; done ) &
exec .venv/bin/streamlit run app.py
