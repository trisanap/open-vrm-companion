#!/bin/bash

# Get the directory where this script is located
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 Starting Open VRM Companion..."

# Check if venv exists and activate it
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Error: Virtual environment 'venv' not found!"
    exit 1
fi

# Run the bridge
python groq_bridge.py
