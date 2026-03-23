from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="News text to classify",
        json_schema_extra={
            "examples": [
                "NASA launches new Mars exploration mission with advanced rover technology"
            ]
        },
    )


class PredictResponse(BaseModel):
    prediction: str = Field(description="Predicted news category")
    confidence: float = Field(description="Prediction confidence (highest probability)")
    label_index: int = Field(description="Index of the predicted category")
    probabilities: dict[str, float] = Field(
        description="Probability distribution across all categories"
    )
