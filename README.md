---
title: AG News Classifier
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# EAI6010 Module 5: ULMFiT Text Classification Microservice

## Overview

This project wraps the **ULMFiT (AWD-LSTM) text classification model** developed in Assignment 3 into a REST API microservice and deploys it to the cloud. The model is trained on the AG News dataset and classifies news text into four categories:

| Label | Category |
|-------|----------|
| 0     | World    |
| 1     | Sports   |
| 2     | Business |
| 3     | Sci/Tech |

**Live Service URL:** https://melodychen-ag-news-classifier.hf.space

**Swagger Docs:** https://melodychen-ag-news-classifier.hf.space/docs

---

## Project Structure

```
EAI6010/
├── app.py                     # FastAPI application entry point
├── model.py                   # Model loading and inference logic
├── schemas.py                 # Pydantic request/response schemas
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container definition
├── .dockerignore              # Docker ignore file
├── .gitignore                 # Git ignore file
├── .gitattributes             # Git LFS tracking config
├── render.yaml                # Render deployment config
├── models/
│   └── ag_news_classifier.pkl # Exported fastai model (Git LFS)
├── docs/
│   └── assignment3.md         # Assignment 3 write-up (reference)
└── notebooks/
    ├── EAI6010_Assignment3.ipynb  # Original Assignment 3 notebook
    └── train_export_model.ipynb   # Model training & export notebook
```

---

## Tech Stack

| Component      | Technology              |
|----------------|-------------------------|
| ML Framework   | fastai 2.x + PyTorch    |
| Model          | ULMFiT (AWD-LSTM)       |
| Dataset        | AG News (4-class)       |
| Web Framework  | FastAPI + Uvicorn        |
| Containerization | Docker                |
| Deployment     | Hugging Face Spaces      |
| API Docs       | Swagger UI (auto-generated) |

---

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the service
uvicorn app:app --host 0.0.0.0 --port 8000

# Test request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launches new Mars exploration mission with advanced rover technology"}'
```

---

## API Usage

### GET /

Returns service information and available labels.

### POST /predict

**Request:**

```json
{
  "text": "NASA launches new Mars exploration mission with advanced rover technology"
}
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

---

## Deployment Notes

| Platform               | Result  | Reason                                              |
|------------------------|---------|-----------------------------------------------------|
| Render (free tier)     | Failed  | Out of memory (512MB limit; PyTorch+model needs ~1.5GB) |
| Hugging Face Spaces    | Success | 16GB RAM, Docker-based deployment                   |
