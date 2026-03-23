# Module 5: Deploying a ULMFiT Text Classification Microservice

Shaohua Chen
NUID: 002300119

## 1. Service Overview

This microservice exposes the ULMFiT (AWD-LSTM) text classification model developed in Assignment 3. The model was trained on the AG News dataset and classifies news text into four categories: **World**, **Sports**, **Business**, and **Sci/Tech**.

The service is built with FastAPI, containerized with Docker, and deployed on Hugging Face Spaces.

### Service URL

**https://melodychen-ag-news-classifier.hf.space**

- **Web UI**: https://melodychen-ag-news-classifier.hf.space (interactive page with input box and classify button)
- **Swagger Docs**: https://melodychen-ag-news-classifier.hf.space/docs (auto-generated API documentation with "Try it out" feature)
- **API Endpoint**: `POST https://melodychen-ag-news-classifier.hf.space/predict`

> **Note**: The free-tier instance may spin down after ~15 minutes of inactivity. If this happens, the first request will take 1–2 minutes to restart the container. Subsequent requests will respond in under 1 second.

---

## 2. General Input and Output

### Input

The service accepts a `POST` request to `/predict` with a JSON body containing a single field:

| Field  | Type   | Required | Constraints           | Description                    |
|--------|--------|----------|-----------------------|--------------------------------|
| `text` | string | Yes      | 1–5000 characters     | The news text to classify      |

**Request format:**

```
POST /predict
Content-Type: application/json

{
  "text": "<news article text>"
}
```

### Output

The service returns a JSON response with the following fields:

| Field           | Type              | Description                                        |
|-----------------|-------------------|----------------------------------------------------|
| `prediction`    | string            | The predicted category (World, Sports, Business, or Sci/Tech) |
| `confidence`    | float             | Confidence score of the prediction (0.0 to 1.0)    |
| `label_index`   | integer           | Numeric index of the predicted category             |
| `probabilities` | object            | Probability distribution across all four categories |

**Response format:**

```json
{
  "prediction": "<category>",
  "confidence": 0.0,
  "label_index": 0,
  "probabilities": {
    "Business": 0.0,
    "Sci/Tech": 0.0,
    "Sports": 0.0,
    "World": 0.0
  }
}
```

---

## 3. Specific Examples

### Example 1: Sci/Tech

**Request:**

```bash
curl -X POST https://melodychen-ag-news-classifier.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launches new Mars exploration mission with advanced rover technology designed to search for signs of ancient life."}'
```

**Response:**

```json
{
  "prediction": "Sci/Tech",
  "confidence": 0.9895,
  "label_index": 1,
  "probabilities": {
    "Business": 0.0074,
    "Sci/Tech": 0.9895,
    "Sports": 0.0007,
    "World": 0.0024
  }
}
```

### Example 2: Sports

**Request:**

```bash
curl -X POST https://melodychen-ag-news-classifier.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Manchester United wins Champions League final in dramatic penalty shootout against Bayern Munich."}'
```

**Response:**

```json
{
  "prediction": "Sports",
  "confidence": 0.7215,
  "label_index": 2,
  "probabilities": {
    "Business": 0.0258,
    "Sci/Tech": 0.0661,
    "Sports": 0.7215,
    "World": 0.1867
  }
}
```

### Example 3: Business

**Request:**

```bash
curl -X POST https://melodychen-ag-news-classifier.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Wall Street stocks surge as Federal Reserve signals interest rate cuts amid strong economic growth data."}'
```

**Response:**

```json
{
  "prediction": "Business",
  "confidence": 0.9748,
  "label_index": 0,
  "probabilities": {
    "Business": 0.9748,
    "Sci/Tech": 0.0096,
    "Sports": 0.0019,
    "World": 0.0138
  }
}
```

### Example 4: World

**Request:**

```bash
curl -X POST https://melodychen-ag-news-classifier.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "UN Security Council holds emergency meeting on Middle East crisis as diplomatic tensions escalate."}'
```

**Response:**

```json
{
  "prediction": "World",
  "confidence": 0.9362,
  "label_index": 3,
  "probabilities": {
    "Business": 0.0194,
    "Sci/Tech": 0.0403,
    "Sports": 0.004,
    "World": 0.9362
  }
}
```

---

## 4. How to Invoke the Service

### Option A: Web Browser (Easiest)

1. Open https://melodychen-ag-news-classifier.hf.space
2. Type or paste a news article into the text box
3. Click **Classify** (or press Ctrl+Enter)
4. View the predicted category, confidence bar chart, and full JSON response

### Option B: Swagger UI

1. Open https://melodychen-ag-news-classifier.hf.space/docs
2. Click on **POST /predict**
3. Click **Try it out**
4. Replace the example text in the request body and click **Execute**

### Option C: curl

```bash
curl -X POST https://melodychen-ag-news-classifier.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news text here"}'
```

### Option D: Python

```python
import requests

response = requests.post(
    "https://melodychen-ag-news-classifier.hf.space/predict",
    json={"text": "Your news text here"}
)
print(response.json())
```

---

## 5. Deployment Process and Errors Encountered

### Architecture

The service consists of:

- **FastAPI** application serving a `POST /predict` endpoint and an interactive web UI
- **ULMFiT (AWD-LSTM)** model exported via fastai's `learn.export()` as a 148 MB pickle file
- **Docker** container based on Python 3.10 slim image
- Model file tracked with **Git LFS** in the repository

### Deployment Attempt 1: Render (Failed)

The first deployment was attempted on Render using a Docker-based Web Service on the free tier.

**Build phase**: Succeeded. The Docker image was built, dependencies (fastai, PyTorch, FastAPI) were installed, and the model file was copied into the container.

**Error 1 — Missing IPython dependency**: At startup, the application crashed with `ModuleNotFoundError: No module named 'IPython'`. This was caused by `fastprogress` (a fastai dependency) importing `IPython.display` at module load time. The fix was adding `ipython>=8.0.0` to `requirements.txt`.

**Error 2 — Out of Memory**: After fixing the IPython issue, the application started but was immediately terminated with `Out of memory (used over 512Mi)`. Render's free tier limits containers to 512 MB of RAM, which is insufficient for PyTorch (~400 MB) plus the fastai framework and the 148 MB model file, which together require approximately 1.5 GB.

This is consistent with the assignment note that students may encounter errors related to free account limitations.

### Deployment Attempt 2: Hugging Face Spaces (Succeeded)

After the Render OOM failure, the service was redeployed to Hugging Face Spaces using the Docker SDK. HF Spaces provides 16 GB of RAM on the free tier, which is more than sufficient for the model.

**Changes made for HF Spaces:**

- Changed the exposed port from 8000 to 7860 (HF Spaces default)
- Added HF Spaces YAML metadata to the top of `README.md` (`sdk: docker`)
- Pushed the repository (including the model via Git LFS) to the HF Space remote

The build completed successfully and the service has been running without issues since deployment.

### Summary of Deployment Errors

| Platform            | Phase   | Error                        | Resolution                              |
|---------------------|---------|------------------------------|-----------------------------------------|
| Render (free tier)  | Startup | `No module named 'IPython'`  | Added `ipython` to `requirements.txt`   |
| Render (free tier)  | Runtime | Out of memory (512 MB limit) | Switched to Hugging Face Spaces (16 GB) |
| HF Spaces (free)    | —       | No errors                    | Deployed successfully                   |

---

## 6. Source Code

The complete source code is available at: https://github.com/MelodyChen713/EAI6010
