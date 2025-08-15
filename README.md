# Job Title Normalizer – API Usage

The **Job Title Normalizer API** standardizes job titles according to [Google Job Posting structured data guidelines](https://developers.google.com/search/docs/appearance/structured-data/job-posting), powered by an **ONNX-optimized T5 model**.
Supports **multi-GPU inference**, **Redis caching**, and **batch processing** for high throughput.

---

## **Running the API**

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Docker

```bash
docker build -t job-title-normalizer .
docker run -p 8080:8080 job-title-normalizer
```

---

## **Endpoints**

### 1. **Health Check**

`GET /health`

Returns model and system status.

```bash
curl -s http://localhost:8080/health
```

Example response:

```json
{
  "status": "healthy",
  "models_loaded": true,
  "gpu_enabled": true,
  "prewarmed": true,
  "gpu_sessions": [0, 1]
}
```

---

### 2. **Normalize Single Title**

`POST /normalize`

Normalize a single job title to a Google-compliant format.

#### Request

```json
{
  "title": "Sr. SWE (Java)",
  "use_cache": true
}
```

```bash
curl -X POST http://localhost:8080/normalize \
  -H "Content-Type: application/json" \
  -d '{"title":"Sr. SWE (Java)", "use_cache":true}'
```

#### Response

```json
{
  "normalized_title": "Senior Software Engineer",
  "processing_time_ms": 12.34,
  "cached": false
}
```

---

### 3. **Normalize Batch of Titles**

`POST /normalize/batch`

Normalize multiple titles in a single request.

#### Request

```json
{
  "titles": [
    "Sr. SWE (Java)",
    "ML Eng",
    "Product Mgr"
  ],
  "use_cache": true
}
```

```bash
curl -X POST http://localhost:8080/normalize/batch \
  -H "Content-Type: application/json" \
  -d '{"titles":["Sr. SWE (Java)", "ML Eng", "Product Mgr"], "use_cache":true}'
```

#### Response

```json
{
  "results": [
    {"title": "Sr. SWE (Java)", "normalized_title": "Senior Software Engineer", "cached": false},
    {"title": "ML Eng", "normalized_title": "Machine Learning Engineer", "cached": false},
    {"title": "Product Mgr", "normalized_title": "Product Manager", "cached": false}
  ],
  "total_time_ms": 30.12,
  "avg_time_ms": 10.04
}
```

---

### 4. **Warmup**

`POST /warmup`

Pre-loads the model with sample inputs to reduce cold start latency.

```bash
curl -X POST http://localhost:8080/warmup
```

---

## **Notes**

* **`use_cache`**: When `true`, results are stored in Redis for faster repeat queries.
* The API automatically uses **round-robin GPU selection** for single requests and **split batching** for multi-GPU.
* Recommended to run with **Redis** for production use.

