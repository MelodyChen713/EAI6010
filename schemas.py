from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="待分类的新闻文本",
        json_schema_extra={
            "examples": [
                "NASA launches new Mars exploration mission with advanced rover technology"
            ]
        },
    )


class PredictResponse(BaseModel):
    prediction: str = Field(description="预测的新闻类别")
    confidence: float = Field(description="预测置信度（最高概率）")
    label_index: int = Field(description="预测类别的索引")
    probabilities: dict[str, float] = Field(
        description="各类别的概率分布"
    )
