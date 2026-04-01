#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Building Docker images..."
docker compose -f infra/docker-compose.yml build

echo "Starting services..."
docker compose -f infra/docker-compose.yml up -d

echo "Services started!"
echo "API: http://localhost:8000"
echo "Docs: http://localhost:8000/docs"
