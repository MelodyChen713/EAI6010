import os
from pathlib import Path
from fastai.text.all import load_learner

MODEL_PATH = os.environ.get("MODEL_PATH", "models/ag_news_classifier.pkl")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "MelodyChen713/ag-news-classifier")
HF_FILENAME = os.environ.get("HF_FILENAME", "ag_news_classifier.pkl")


def _ensure_model() -> Path:
    path = Path(MODEL_PATH)
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download
    print(f"Downloading model from HuggingFace Hub: {HF_REPO_ID}/{HF_FILENAME} ...")
    downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    return Path(downloaded)


learn = load_learner(_ensure_model())
LABELS = list(learn.dls.vocab[1])


def predict(text: str) -> dict:
    pred_class, pred_idx, probs = learn.predict(text)
    return {
        "prediction": str(pred_class),
        "confidence": round(float(probs.max()), 4),
        "label_index": int(pred_idx),
        "probabilities": {
            label: round(float(probs[i]), 4) for i, label in enumerate(LABELS)
        },
    }
