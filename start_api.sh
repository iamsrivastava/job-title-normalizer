#!/bin/bash

# Startup script for the Job Title Normalization API

echo "Starting Job Title Normalization API..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import fastapi, onnxruntime, transformers, redis" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Required packages are not installed. Please run:"
    echo "   pip install -r onnx-requirements.txt"
    exit 1
fi

# Check if model directory exists
if [ ! -d "model/job-title-extractor-onnx" ]; then
    echo "ERROR: Model directory not found: model/job-title-extractor-onnx"
    echo "   Please ensure the models are in the correct location"
    exit 1
fi

# Check if Redis is running (optional)
echo "Checking Redis connection..."
python3 -c "
import redis
try:
    r = redis.from_url('redis://localhost:6379')
    r.ping()
    print('SUCCESS: Redis is running')
except:
    print('WARNING: Redis is not running - caching will be disabled')
    print('   To enable caching, start Redis: brew services start redis (macOS) or sudo systemctl start redis-server (Linux)')
"

# Start the API server
echo "Starting FastAPI server on http://localhost:8080"
echo "API documentation will be available at http://localhost:8080/docs"
echo "Health check: http://localhost:8080/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 app_onnx.py
