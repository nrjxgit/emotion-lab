from pydantic import BaseModel

CLASSES = [
    "happiness",
    "sadness",
    "anger",
    "surprise",
    "fear",
    "disgust",
    "neutral",
    "unknown",
]


class AU(BaseModel):
    code: str
    intensity: float
    confidence: float


class InferResponse(BaseModel):
    ethics_notice: str
