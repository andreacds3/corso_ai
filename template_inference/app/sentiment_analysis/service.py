from functools import lru_cache
from typing import Optional
from async_lru import alru_cache

from config import BatchConfiguration
from logging_utils import APPLICATION_LOGGER
from schema.base import Device
from sentiment_analysis.batching import Batcher
from sentiment_analysis.exceptions import ApiException
from sentiment_analysis.nlp import NLPSentimentPipeline


class SentimentAnalysisService:
    """
    Class representing the service responsible for handling the translation requests
    """

    def __init__(self, device: Device, batch_configuration: Optional[BatchConfiguration]):
        """

        :param device: The device where load the model

        """
        self.device = device
        self.batch_configuration = batch_configuration
        self.nlp_pipeline = NLPSentimentPipeline(device=device)
        self.batched_pipeline: Optional[Batcher] = None

    def initialize(self):
        self.nlp_pipeline.initialize()
        if self.batch_configuration:
            self.batched_pipeline = Batcher(nlp_pipeline=self.nlp_pipeline,
                                            batch_size=self.batch_configuration.batch_size,
                                            batch_timeout=self.batch_configuration.batch_timeout)
            self.batched_pipeline.start_runner()

    @alru_cache(maxsize=32)
    async def extract_sentiment(self, text: str):
        """
        Summarize a text with a number of tokens between min_length and max_length

        :param text: The text to be analyzed for sentiment
        :return: the summary of the given text
        """
        if self.batched_pipeline:
            try:
                result = await self.batched_pipeline.extract_sentiment(text.strip())
                return result
            except Exception as ex:
                APPLICATION_LOGGER.exception(ex)
                raise ApiException()
        else:
            try:
                result = await self.nlp_pipeline.extract_sentiment([text])
                return result[0]
            except Exception as ex:
                APPLICATION_LOGGER.exception(ex)
                raise ApiException()
