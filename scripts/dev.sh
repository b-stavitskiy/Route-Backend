#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Installing dependencies with uv..."
uv sync

echo "Running migrations..."
uv run alembic upgrade head

echo "Starting API server..."
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
