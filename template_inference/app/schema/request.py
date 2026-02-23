from typing import Optional,Self
from pydantic import BaseModel,Field,model_validator


class SentimentAnalysisRequest(BaseModel):
    """
    Represent a document to be translated.
    the attributes are:
    text: the text to be translated
    language: the optional language of the text
    output_language: the language the document should be translated
    """
    # id:str
    text: str


    class Config:
        json_schema_extra = {
            "example": {
                "text": "The movie was amazing",
            }
        }

