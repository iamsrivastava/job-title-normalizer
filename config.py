#!/usr/bin/env python3
"""
Configuration file for the Job Title Normalization API
Optimized for dual RTX 5060 Ti GPUs (16GB VRAM each)
Override default settings by modifying this file
"""

import os
from pathlib import Path

# === GPU Configuration ===
ENABLE_GPU = True
GPU_MEMORY_LIMIT = "14GB"  # Leave 2GB buffer for system operations
GPU_DEVICE_ID = 0  # Primary GPU (can be changed to 1 for second GPU)
ENABLE_MULTI_GPU = True  # Enable multi-GPU support
GPU_DEVICES = [0, 1]  # Use both GPUs

# === Model Configuration ===
MODEL_DIR = Path("./model/job-title-extractor-onnx")
MAX_INPUT_LEN = 64
MAX_OUTPUT_LEN = 16
MODEL_OPTIMIZATION_LEVEL = "high"  # high, medium, low
MODEL_PRECISION = "float16"  # float16 for better GPU performance and memory efficiency

# === Performance Tuning ===
ENABLE_TENSORRT = True  # Enable TensorRT optimization for RTX 5060 Ti
ENABLE_CUDA_GRAPH = True  # Enable CUDA graph optimization
ENABLE_MIXED_PRECISION = True  # Enable mixed precision for better performance
ENABLE_DYNAMIC_BATCHING = True
ENABLE_GPU_PIPELINING = True  # Enable GPU pipeline parallelism

# === Batch Processing ===
BATCH_SIZE = 128  # Increased for 16GB VRAM
MAX_BATCH_SIZE = 256  # Maximum batch size for dual GPU processing
BATCH_TIMEOUT = 30
BATCH_DISTRIBUTION = "round_robin"  # round_robin, load_balanced, or gpu_0_only

# === Cache Configuration ===
CACHE_TTL = 3600  # 1 hour
CACHE_ENABLED = True
CACHE_MAX_SIZE = 20000  # Increased cache size for better performance
CACHE_PREFIX = "title_norm"

# === Redis Configuration ===
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None
REDIS_USERNAME = None
REDIS_SSL = False
REDIS_SSL_CERT_REQS = "none"
REDIS_CONNECT_TIMEOUT = 5
REDIS_SOCKET_TIMEOUT = 5
REDIS_RETRY_ON_TIMEOUT = True
REDIS_HEALTH_CHECK_INTERVAL = 30

# === Server Configuration ===
HOST = "0.0.0.0"
PORT = 8080
WORKERS = 8  # Increased workers for dual GPU setup
MAX_CONCURRENT_REQUESTS = 200  # Increased for dual GPU capacity
REQUEST_TIMEOUT = 30

# === Logging Configuration ===
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# === Security Configuration ===
CORS_ORIGINS = ["*"]
RATE_LIMIT_ENABLED = False
RATE_LIMIT_REQUESTS = 200  # Increased rate limit for dual GPU
RATE_LIMIT_WINDOW = 60

# === Monitoring Configuration ===
METRICS_ENABLED = True
HEALTH_CHECK_ENABLED = True
PROFILING_ENABLED = False
GPU_MONITORING_INTERVAL = 5  # GPU monitoring every 5 seconds

# === GPU Memory Management ===
GPU_MEMORY_FRACTION = 0.85  # Use 85% of available GPU memory
GPU_MEMORY_GROWTH = True  # Allow GPU memory to grow as needed
GPU_MEMORY_POOLING = True  # Enable memory pooling for better efficiency

# === Model Distribution Strategy ===
MODEL_DISTRIBUTION = "load_balanced"  # load_balanced, round_robin, or gpu_0_only
ENABLE_MODEL_SHARDING = False  # Shard large models across GPUs if needed
MODEL_REPLICATION = True  # Keep model copies on both GPUs for redundancy

# === Environment Override ===
# You can override any setting using environment variables
# Example: export ENABLE_GPU=false to disable GPU

def get_config_value(key: str, default=None):
    """Get configuration value with environment variable override"""
    env_key = key.upper()
    return os.getenv(env_key, default)

# Override settings with environment variables if present
ENABLE_GPU = get_config_value("ENABLE_GPU", ENABLE_GPU)
ENABLE_MULTI_GPU = get_config_value("ENABLE_MULTI_GPU", ENABLE_MULTI_GPU)
GPU_MEMORY_LIMIT = get_config_value("GPU_MEMORY_LIMIT", GPU_MEMORY_LIMIT)
GPU_DEVICE_ID = int(get_config_value("GPU_DEVICE_ID", GPU_DEVICE_ID))
GPU_DEVICES = [int(x) for x in get_config_value("GPU_DEVICES", ",".join(map(str, GPU_DEVICES))).split(",")]
MODEL_OPTIMIZATION_LEVEL = get_config_value("MODEL_OPTIMIZATION_LEVEL", MODEL_OPTIMIZATION_LEVEL)
MODEL_PRECISION = get_config_value("MODEL_PRECISION", MODEL_PRECISION)
BATCH_SIZE = int(get_config_value("BATCH_SIZE", BATCH_SIZE))
MAX_BATCH_SIZE = int(get_config_value("MAX_BATCH_SIZE", MAX_BATCH_SIZE))
BATCH_DISTRIBUTION = get_config_value("BATCH_DISTRIBUTION", BATCH_DISTRIBUTION)
CACHE_TTL = int(get_config_value("CACHE_TTL", CACHE_TTL))
REDIS_HOST = get_config_value("REDIS_HOST", REDIS_HOST)
REDIS_PORT = int(get_config_value("REDIS_PORT", REDIS_PORT))
PORT = int(get_config_value("PORT", PORT))
WORKERS = int(get_config_value("WORKERS", WORKERS))
LOG_LEVEL = get_config_value("LOG_LEVEL", LOG_LEVEL)

# Print configuration summary
if __name__ == "__main__":
    print("=== Job Title Normalization API Configuration ===")
    print(f"GPU Enabled: {ENABLE_GPU}")
    print(f"Multi-GPU Enabled: {ENABLE_MULTI_GPU}")
    print(f"GPU Devices: {GPU_DEVICES}")
    print(f"Primary GPU: {GPU_DEVICE_ID}")
    print(f"GPU Memory Limit: {GPU_MEMORY_LIMIT}")
    print(f"GPU Memory Fraction: {GPU_MEMORY_FRACTION * 100}%")
    print(f"Model Optimization: {MODEL_OPTIMIZATION_LEVEL}")
    print(f"Model Precision: {MODEL_PRECISION}")
    print(f"TensorRT Enabled: {ENABLE_TENSORRT}")
    print(f"CUDA Graph Enabled: {ENABLE_CUDA_GRAPH}")
    print(f"Mixed Precision: {ENABLE_MIXED_PRECISION}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Batch Size: {MAX_BATCH_SIZE}")
    print(f"Batch Distribution: {BATCH_DISTRIBUTION}")
    print(f"Cache TTL: {CACHE_TTL}s")
    print(f"Cache Max Size: {CACHE_MAX_SIZE}")
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print(f"Server: {HOST}:{PORT}")
    print(f"Workers: {WORKERS}")
    print(f"Max Concurrent Requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Log Level: {LOG_LEVEL}")
    
    # GPU-specific recommendations
    if ENABLE_GPU and ENABLE_MULTI_GPU:
        print(f"\n=== Dual GPU Optimization Tips ===")
        print(f"• Each RTX 5060 Ti has 16GB VRAM")
        print(f"• Configured to use {GPU_MEMORY_LIMIT} per GPU")
        print(f"• Batch size increased to {BATCH_SIZE} for better GPU utilization")
        print(f"• Workers increased to {WORKERS} for dual GPU processing")
        print(f"• Consider using GPU 0 for primary and GPU 1 for backup")
        print(f"• Monitor GPU memory usage with /health endpoint")
