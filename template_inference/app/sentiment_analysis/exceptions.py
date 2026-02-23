from typing import Optional, List


class SentimentAnalysisError(Exception):
    pass


class SentimentAnalysisErrorWithIds(SentimentAnalysisError):
    def __init__(self, error_ids: Optional[List]):
        super(SentimentAnalysisError).__init__()
        self.error_ids = error_ids


class ApiException(Exception):
    pass
