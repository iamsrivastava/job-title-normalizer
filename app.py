# app.py
"""
Ultra-optimized FastAPI server for job-title normalization using ONNX Runtime.
- Multi-GPU (one session pair per GPU), round-robin for singles, split for batches
- Preload & pre-warm at startup
- Optional Redis caching
"""

import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, Field
import redis
from transformers import T5Tokenizer
import uvicorn

# ===== Logging =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===== Config =====
class Config:
    MODEL_DIR = Path("./model/job-title-extractor-onnx")
    MAX_INPUT_LEN = 64
    MAX_OUTPUT_LEN = 16

    # Concurrency
    WORKERS = 8
    MAX_CONCURRENT_REQUESTS = 200

    # Caching
    CACHE_ENABLED = True
    CACHE_TTL = 3600
    CACHE_PREFIX = "title_norm"

    # Redis
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_PASSWORD = None
    REDIS_SSL = False
    REDIS_CONNECT_TIMEOUT = 5
    REDIS_SOCKET_TIMEOUT = 5
    REDIS_RETRY_ON_TIMEOUT = True
    # Optional/advanced (kept safe if undefined by using getattr)
    REDIS_USERNAME = None
    REDIS_HEALTH_CHECK_INTERVAL = 0
    REDIS_SSL_CERT_REQS = None  # e.g., "CERT_REQUIRED"

    # GPU / ORT
    ENABLE_GPU = True
    ENABLE_MULTI_GPU = True
    GPU_DEVICE_ID = 0
    GPU_DEVICES = [0, 1]             # both RTX 5060s
    GPU_MEMORY_LIMIT = "14GB"        # per device soft cap
    ENABLE_TENSORRT = False          # keep False unless your ORT build supports it
    MODEL_OPTIMIZATION_LEVEL = "high"  # "high" -> ORT_ENABLE_ALL
    ENABLE_CUDA_GRAPH = False        # not widely available on all builds

    # Batch
    ENABLE_DYNAMIC_BATCHING = True
    MAX_BATCH_SIZE = 256
    BATCH_SIZE = 128

    # CORS / Security
    CORS_ORIGINS = ["*"]

    # Logging
    LOG_LEVEL = "INFO"

# ===== Globals =====
tokenizer: Optional[T5Tokenizer] = None
redis_client: Optional[redis.Redis] = None

# One encoder/decoder session pair per GPU device
_encoder_sessions: List[ort.InferenceSession] = []
_decoder_sessions: List[ort.InferenceSession] = []
_device_ids: List[int] = []  # parallels sessions, contains device id or -1 for CPU

# RR state for single inference
_rr = 0
_rr_lock = threading.Lock()

# Warmup info
_last_warmup: Dict[str, Any] = {"ok": False, "when": None, "info": None}

# Thread pool
executor = concurrent.futures.ThreadPoolExecutor(max_workers=Config.WORKERS)

# ===== Utility =====
def _bytes_from_str(s: str) -> int:
    s = s.strip().upper()
    mult = 1
    if s.endswith("KB"):
        mult, s = 1024, s[:-2]
    elif s.endswith("MB"):
        mult, s = 1024**2, s[:-2]
    elif s.endswith("GB"):
        mult, s = 1024**3, s[:-2]
    return int(float(s) * mult)

def _providers_for_device(device_id: Optional[int]):
    avail = ort.get_available_providers()
    # CPU fallback
    if (not Config.ENABLE_GPU) or ("CUDAExecutionProvider" not in avail) or (device_id is None):
        return ["CPUExecutionProvider"], [{}]

    cuda_opts = {
        "device_id": int(device_id),
        "arena_extend_strategy": "kNextPowerOfTwo",
        "do_copy_in_default_stream": "1",
    }
    if Config.GPU_MEMORY_LIMIT:
        try:
            cuda_opts["gpu_mem_limit"] = _bytes_from_str(Config.GPU_MEMORY_LIMIT)
        except Exception:
            pass

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    provider_options = [cuda_opts, {}]

    # (Optional) TensorRT if available in your wheel
    if Config.ENABLE_TENSORRT and "TensorrtExecutionProvider" in avail:
        providers.insert(0, "TensorrtExecutionProvider")
        provider_options.insert(0, {})  # default TRT opts

    return providers, provider_options

def _session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    # Optimization level
    if Config.MODEL_OPTIMIZATION_LEVEL == "high":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    elif Config.MODEL_OPTIMIZATION_LEVEL == "medium":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    else:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    # Defensive: not all builds expose this
    if hasattr(so, "enable_cpu_mem_arena"):
        so.enable_cpu_mem_arena = True
    if hasattr(so, "enable_mem_pattern"):
        so.enable_mem_pattern = True
    if hasattr(so, "enable_mem_reuse"):
        so.enable_mem_reuse = True
    if hasattr(so, "enable_cuda_graph"):
        try:
            so.enable_cuda_graph = bool(Config.ENABLE_CUDA_GRAPH)
        except Exception:
            pass
    return so

def _pick_sessions_round_robin():
    global _rr
    with _rr_lock:
        i = _rr % max(1, len(_encoder_sessions))
        _rr += 1
    return _encoder_sessions[i], _decoder_sessions[i], _device_ids[i]

def _split_for_devices(items: List[Any], n_parts: int) -> List[List[Any]]:
    # Even-ish split preserving order; ensures each device gets work
    return [items[i::n_parts] for i in range(n_parts)]

# ===== Load / Init =====
def load_model():
    """Create tokenizer + one session pair per device."""
    global tokenizer, _encoder_sessions, _decoder_sessions, _device_ids

    logger.info("Loading tokenizer and creating ONNX sessions…")
    tokenizer = T5Tokenizer.from_pretrained(str(Config.MODEL_DIR))

    enc_path = str((Config.MODEL_DIR / "encoder_model.onnx").resolve())
    dec_path = str((Config.MODEL_DIR / "decoder_model.onnx").resolve())

    # Devices to use
    devices = Config.GPU_DEVICES if (Config.ENABLE_GPU and Config.ENABLE_MULTI_GPU) else [Config.GPU_DEVICE_ID]
    if "CUDAExecutionProvider" not in ort.get_available_providers() or not Config.ENABLE_GPU:
        devices = [None]  # CPU fallback

    _encoder_sessions.clear()
    _decoder_sessions.clear()
    _device_ids.clear()

    so = _session_options()

    for dev in devices:
        providers, provider_options = _providers_for_device(dev)
        logger.info(f"Creating sessions on {'CPU' if dev is None else f'GPU:{dev}'} with providers={providers}")
        enc_sess = ort.InferenceSession(enc_path, sess_options=so, providers=providers, provider_options=provider_options)
        dec_sess = ort.InferenceSession(dec_path, sess_options=so, providers=providers, provider_options=provider_options)
        _encoder_sessions.append(enc_sess)
        _decoder_sessions.append(dec_sess)
        _device_ids.append(-1 if dev is None else int(dev))

    logger.info(f"Loaded {len(_encoder_sessions)} session pair(s). Example providers: {_encoder_sessions[0].get_providers()}")

def init_redis():
    """Initialize Redis connection (optional)."""
    global redis_client
    try:
        cfg = dict(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=Config.REDIS_CONNECT_TIMEOUT,
            socket_timeout=Config.REDIS_SOCKET_TIMEOUT,
            retry_on_timeout=Config.REDIS_RETRY_ON_TIMEOUT,
            health_check_interval=getattr(Config, "REDIS_HEALTH_CHECK_INTERVAL", 0),
        )
        if getattr(Config, "REDIS_PASSWORD", None):
            cfg["password"] = Config.REDIS_PASSWORD
        if getattr(Config, "REDIS_USERNAME", None):
            cfg["username"] = Config.REDIS_USERNAME
        if getattr(Config, "REDIS_SSL", False):
            cfg["ssl"] = True
            if getattr(Config, "REDIS_SSL_CERT_REQS", None) is not None:
                cfg["ssl_cert_reqs"] = Config.REDIS_SSL_CERT_REQS

        redis_client = redis.Redis(**cfg)
        redis_client.ping()
        # tiny warm-read so first real call isn't cold
        redis_client.setex("warmup:key", 30, "1")
        _ = redis_client.get("warmup:key")
        logger.info(f"Redis connected at {Config.REDIS_HOST}:{Config.REDIS_PORT}")
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        redis_client = None

# ===== Cache helpers =====
def get_cache_key(title: str) -> str:
    return f"{Config.CACHE_PREFIX}:{hash(title)}"

def get_from_cache(title: str) -> Optional[str]:
    if not (Config.CACHE_ENABLED and redis_client):
        return None
    try:
        return redis_client.get(get_cache_key(title))
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
        return None

def set_cache(title: str, normalized_title: str):
    if not (Config.CACHE_ENABLED and redis_client):
        return
    try:
        redis_client.setex(get_cache_key(title), Config.CACHE_TTL, normalized_title)
    except Exception as e:
        logger.warning(f"Cache set error: {e}")

def batch_get_from_cache(titles: List[str]) -> Dict[str, str]:
    if not (Config.CACHE_ENABLED and redis_client):
        return {}
    try:
        keys = [get_cache_key(t) for t in titles]
        vals = redis_client.mget(keys)
        return {t: v for t, v in zip(titles, vals) if v is not None}
    except Exception as e:
        logger.warning(f"Batch cache get error: {e}")
        return {}

def batch_set_cache(title_results: Dict[str, str]):
    if not (Config.CACHE_ENABLED and redis_client):
        return
    try:
        pipe = redis_client.pipeline()
        for t, v in title_results.items():
            pipe.setex(get_cache_key(t), Config.CACHE_TTL, v)
        pipe.execute()
    except Exception as e:
        logger.warning(f"Batch cache set error: {e}")

# ===== Core inference =====
def _encode_inputs(text: str):
    enc = tokenizer.encode_plus(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=Config.MAX_INPUT_LEN
    )
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)

def _decode_loop(enc_sess, dec_sess, input_ids: np.ndarray, attention_mask: np.ndarray) -> str:
    h = enc_sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
    dec_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    tokens: List[int] = []
    for _ in range(Config.MAX_OUTPUT_LEN):
        logits = dec_sess.run(None, {
            "input_ids": dec_ids,
            "encoder_hidden_states": h,
            "encoder_attention_mask": attention_mask
        })[0][0, -1, :]
        nxt = int(np.argmax(logits))
        if nxt == tokenizer.eos_token_id:
            break
        tokens.append(nxt)
        dec_ids = np.concatenate([dec_ids, np.array([[nxt]], dtype=np.int64)], axis=1)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def normalize_title_onnx(title: str) -> str:
    enc_sess, dec_sess, _ = _pick_sessions_round_robin()
    txt = f"normalize job title: {title}"
    iid, am = _encode_inputs(txt)
    return _decode_loop(enc_sess, dec_sess, iid, am)

def _normalize_batch_on_session(titles: List[str], enc_sess, dec_sess) -> List[str]:
    outs = []
    for t in titles:
        txt = f"normalize job title: {t}"
        iid, am = _encode_inputs(txt)
        outs.append(_decode_loop(enc_sess, dec_sess, iid, am))
    return outs

# ===== Async wrappers =====
async def process_title_async(title: str, use_cache: bool = True) -> Dict[str, Any]:
    start = time.time()
    if use_cache:
        cached = get_from_cache(title)
        if cached:
            return {
                "title": title,
                "normalized_title": cached,
                "processing_time_ms": (time.time() - start) * 1000,
                "cached": True
            }
    loop = asyncio.get_event_loop()
    normalized = await loop.run_in_executor(executor, normalize_title_onnx, title)
    if use_cache:
        set_cache(title, normalized)
    return {
        "title": title,
        "normalized_title": normalized,
        "processing_time_ms": (time.time() - start) * 1000,
        "cached": False
    }

# ===== Warmup =====
def _make_dummy_titles(n: int = 32) -> List[str]:
    base = [
        "Senior Software Engineer", "Data Scientist", "Product Manager",
        "UX Designer", "DevOps Engineer", "QA Analyst",
        "Machine Learning Engineer", "Backend Developer",
    ]
    return [f"{base[i % len(base)]} {i}" for i in range(n)]

def _warmup_single(enc_sess, dec_sess, title: str):
    txt = f"normalize job title: {title}"
    iid, am = _encode_inputs(txt)
    # encoder forward
    h = enc_sess.run(None, {"input_ids": iid, "attention_mask": am})[0]
    # decoder a few steps
    dec_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    for _ in range(min(4, Config.MAX_OUTPUT_LEN)):
        logits = dec_sess.run(None, {
            "input_ids": dec_ids,
            "encoder_hidden_states": h,
            "encoder_attention_mask": am
        })[0][0, -1, :]
        nxt = int(np.argmax(logits))
        if nxt == tokenizer.eos_token_id:
            break
        dec_ids = np.concatenate([dec_ids, np.array([[nxt]], dtype=np.int64)], axis=1)

def _warmup_all(timeout_sec: float = 25.0) -> Dict[str, Any]:
    start = time.time()
    info: Dict[str, Any] = {"devices": [], "per_device_ms": {}, "batch_ms": None, "ok": True, "error": None}
    try:
        dummies = _make_dummy_titles(8)
        # Per-device warm
        for idx, (enc_sess, dec_sess) in enumerate(zip(_encoder_sessions, _decoder_sessions)):
            t0 = time.time()
            for t in dummies[:4]:
                _warmup_single(enc_sess, dec_sess, t)
            info["per_device_ms"][f"session_{idx}"] = (time.time() - t0) * 1000.0
            info["devices"].append(_device_ids[idx])
            if time.time() - start > timeout_sec:
                info["ok"] = False
                info["error"] = "warmup timeout (per-device)"
                return info
        # Batch warm (split across devices)
        t1 = time.time()
        _ = warmup_batch_call(dummies)
        info["batch_ms"] = (time.time() - t1) * 1000.0
        return info
    except Exception as e:
        info["ok"] = False
        info["error"] = str(e)
        return info

def warmup_batch_call(titles: List[str]) -> List[str]:
    """Internal: split across sessions and run in parallel."""
    results: Dict[str, str] = {}
    n_sessions = max(1, len(_encoder_sessions))
    splits = _split_for_devices(titles, n_sessions)
    futs = []
    for i, part in enumerate(splits):
        if not part:
            continue
        enc_sess = _encoder_sessions[i]
        dec_sess = _decoder_sessions[i]
        futs.append((part, executor.submit(_normalize_batch_on_session, part, enc_sess, dec_sess)))
    for part, f in futs:
        outs = f.result()
        for t, norm in zip(part, outs):
            results[t] = norm
    # preserve order
    return [results[t] for t in titles]

# ===== API models =====
class TitleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    use_cache: bool = Field(True)

class TitleResponse(BaseModel):
    normalized_title: str
    processing_time_ms: float
    cached: bool = False

class BatchRequest(BaseModel):
    titles: List[str] = Field(..., min_items=1, max_items=200)
    use_cache: bool = Field(True)

class BatchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_time_ms: float
    avg_time_ms: float

# ===== FastAPI =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    init_redis()

    # Pre-warm synchronously so first request is hot
    try:
        logger.info("Starting warmup…")
        warm_info = _warmup_all(timeout_sec=25.0)
        global _last_warmup
        _last_warmup = {"ok": warm_info.get("ok", False), "when": time.time(), "info": warm_info}
        logger.info(f"Warmup done: {warm_info}")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")

    logger.info("Application startup complete")
    yield
    executor.shutdown(wait=True)
    logger.info("Application shutdown complete")

app = FastAPI(
    title="Ultra-Optimized Job Title Normalization API",
    description="High-performance API for normalizing job titles using ONNX Runtime (multi-GPU, pre-warmed)",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=Config.CORS_ORIGINS)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

# ===== Endpoints =====
@app.get("/health")
async def health_check():
    info = {
        "status": "healthy",
        "models_loaded": len(_encoder_sessions) > 0 and len(_decoder_sessions) > 0,
        "cache_available": redis_client is not None,
        "optimization_level": Config.MODEL_OPTIMIZATION_LEVEL,
        "gpu_enabled": Config.ENABLE_GPU,
        "multi_gpu": Config.ENABLE_MULTI_GPU,
        "gpu_sessions": _device_ids,
        "timestamp": time.time(),
        "prewarmed": _last_warmup.get("ok", False),
        "warmup_info": _last_warmup.get("info"),
    }
    try:
        provs = _encoder_sessions[0].get_providers() if _encoder_sessions else []
        info["providers"] = provs
        info["primary_provider"] = provs[0] if provs else None
    except Exception as e:
        info["providers_error"] = str(e)
    # Optional: NVML stats
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_memory = {}
        for dev in [d for d in _device_ids if d >= 0]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            gpu_memory[f"gpu_{dev}"] = {
                "name": name,
                "used_mb": mem.used // 1024**2,
                "total_mb": mem.total // 1024**2,
                "free_mb": mem.free // 1024**2,
                "utilization_percent": util.gpu,
            }
        info["gpu_memory"] = gpu_memory
    except Exception:
        info["gpu_memory"] = "nvml not available"
    return info

@app.get("/metrics")
async def get_metrics():
    metrics = {
        "models_loaded": len(_encoder_sessions) > 0 and len(_decoder_sessions) > 0,
        "cache_available": redis_client is not None,
        "max_input_length": Config.MAX_INPUT_LEN,
        "max_output_length": Config.MAX_OUTPUT_LEN,
        "batch_size": Config.BATCH_SIZE,
        "max_batch_size": Config.MAX_BATCH_SIZE,
        "workers": Config.WORKERS,
        "gpu_enabled": Config.ENABLE_GPU,
        "gpu_memory_limit": Config.GPU_MEMORY_LIMIT,
        "multi_gpu_sessions": _device_ids,
        "cache_ttl": Config.CACHE_TTL,
        "cache_prefix": Config.CACHE_PREFIX
    }
    try:
        metrics["current_providers"] = _encoder_sessions[0].get_providers() if _encoder_sessions else []
    except Exception as e:
        metrics["provider_info"] = f"Error: {str(e)}"
    return metrics

@app.post("/warmup")
async def warmup_now():
    loop = asyncio.get_event_loop()
    info = await loop.run_in_executor(executor, _warmup_all, 25.0)
    global _last_warmup
    _last_warmup = {"ok": info.get("ok", False), "when": time.time(), "info": info}
    return info

@app.post("/normalize", response_model=TitleResponse)
async def normalize_title(request: TitleRequest):
    start = time.time()
    try:
        if request.use_cache:
            cached = get_from_cache(request.title)
            if cached:
                return TitleResponse(normalized_title=cached,
                                     processing_time_ms=(time.time() - start) * 1000,
                                     cached=True)
        loop = asyncio.get_event_loop()
        normalized = await loop.run_in_executor(executor, normalize_title_onnx, request.title)
        if request.use_cache:
            set_cache(request.title, normalized)
        return TitleResponse(normalized_title=normalized,
                             processing_time_ms=(time.time() - start) * 1000,
                             cached=False)
    except Exception as e:
        logger.error(f"Error normalizing title: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/normalize/batch", response_model=BatchResponse)
async def normalize_titles_batch(request: BatchRequest):
    start_time = time.time()
    try:
        # Cache
        cached_results: Dict[str, str] = {}
        uncached_titles: List[str] = []
        if request.use_cache:
            cached_results = batch_get_from_cache(request.titles)
            uncached_titles = [t for t in request.titles if t not in cached_results]
        else:
            uncached_titles = list(request.titles)

        # Prepare map with cached entries
        all_results: Dict[str, Dict[str, Any]] = {
            t: {"title": t, "normalized_title": cached_results.get(t), "cached": t in cached_results}
            for t in request.titles
        }

        # Split uncached across GPU sessions and run in parallel
        if uncached_titles:
            n_sessions = max(1, len(_encoder_sessions))
            splits = _split_for_devices(uncached_titles, n_sessions)

            loop = asyncio.get_event_loop()
            tasks = []
            for idx, part in enumerate(splits):
                if not part:
                    continue
                enc_sess = _encoder_sessions[idx]
                dec_sess = _decoder_sessions[idx]
                tasks.append(loop.run_in_executor(
                    executor, _normalize_batch_on_session, part, enc_sess, dec_sess
                ))

            parts_out = await asyncio.gather(*tasks)
            # Stitch back
            out_idx = 0
            for idx, part in enumerate(splits):
                if not part:
                    continue
                outs = parts_out[out_idx]
                out_idx += 1
                for t, norm in zip(part, outs):
                    all_results[t] = {"title": t, "normalized_title": norm, "cached": False}
                if request.use_cache:
                    batch_set_cache({t: n for t, n in zip(part, outs)})

        # Preserve input order
        sorted_results = [all_results[t] for t in request.titles]
        total_time = (time.time() - start_time) * 1000.0
        avg_time = total_time / max(1, len(request.titles))
        return BatchResponse(results=sorted_results, total_time_ms=total_time, avg_time_ms=avg_time)
    except Exception as e:
        logger.error(f"Error in batch normalization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Main =====
if __name__ == "__main__":
    # Tip: you can pin GPU(s) per process using CUDA_VISIBLE_DEVICES if you choose multi-process scaling
    uvicorn.run("app:app", host="0.0.0.0", port=8080, workers=1, log_level="info")
