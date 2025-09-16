#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
celery -A aesthetics_eval.tasks worker --loglevel=INFO
