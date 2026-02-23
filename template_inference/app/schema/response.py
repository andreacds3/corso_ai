from typing import Optional
from pydantic import BaseModel
from enum import Enum


class Sentiment(Enum):
    positive = 'positive'
    negative='negative'


class SentimentAnalysisResponse(BaseModel):
    """
    Represent a summary of a document.
    text: The text of the summary
    """
    # id:str
    sentiment: str
    score: float
