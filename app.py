from fastapi import FastAPI, HTTPException
from schemas import PredictRequest, PredictResponse
import model as clf

app = FastAPI(
    title="AG News 文本分类服务",
    description=(
        "基于 ULMFiT (AWD-LSTM) 的新闻文本分类微服务。\n\n"
        "将输入的新闻文本自动分类为以下四个类别之一：\n"
        "- **World**（世界新闻）\n"
        "- **Sports**（体育）\n"
        "- **Business**（商业）\n"
        "- **Sci/Tech**（科技）"
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
