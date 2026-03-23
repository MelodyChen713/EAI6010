from fastapi import FastAPI, HTTPException
from schemas import PredictRequest, PredictResponse
import model as clf

app = FastAPI(
    title="AG News Text Classification Service",
    description=(
        "A microservice for news text classification based on ULMFiT (AWD-LSTM).\n\n"
        "Classifies input news text into one of the following four categories:\n"
        "- **World**\n"
        "- **Sports**\n"
        "- **Business**\n"
        "- **Sci/Tech**"
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "service": "AG News Text Classification API",
        "model": "ULMFiT (AWD-LSTM)",
        "labels": clf.LABELS,
        "usage": "POST /predict with JSON body {\"text\": \"your news text here\"}",
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = clf.predict(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
