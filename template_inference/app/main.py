import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

from config import Configuration
from schema.request import SentimentAnalysisRequest
from schema.response import SentimentAnalysisResponse
from sentiment_analysis.exceptions import SentimentAnalysisError
from sentiment_analysis.service import SentimentAnalysisService
from logging_utils import AUDIT_LOGGER, UVICORN_LOGGER, APPLICATION_LOGGER
from middleware import LoggingMiddleWare



API_DESCRIPTION = """
App to extract sentiment from a text
"""



def create_app() -> FastAPI:
    """
    Factory method to create the FastAPI app, its routes
    :return: the FastApi app configured
    """

    # Set up authentication through API KEY
    api_key_header = APIKeyHeader(name="X-API-Key")

    def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
        if api_key_header == os.getenv("API-KEY", "devel"):
            return api_key_header
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key",
        )

    state_dict = dict()

    @asynccontextmanager
    async def app_init(app: FastAPI):
        """
        Initialize the app by creating and initializing the summarizer service
        :param app: the fastapi app
        :return: None
        """
        # Load the ML models
        configuration = Configuration.from_configuration_file(
            os.getenv('CONFIGURATION_FILE', '/config/configuration.yml'))
        service = SentimentAnalysisService(
            device=configuration.device,
            batch_configuration=configuration.batch_configuration
        )
        service.initialize()
        APPLICATION_LOGGER.debug(configuration)
        state_dict['configuration'] = configuration
        state_dict['service'] = service
        yield

    # create the app
    app = FastAPI(
        title="Sentiment Analysis",
        description=API_DESCRIPTION,
        version="0.1.0",
        docs_url='/',
        lifespan=app_init
    )

    app.add_middleware(LoggingMiddleWare)


    @app.post('/rest/async/extract_sentiment', response_model=SentimentAnalysisResponse)
    async def summarize(document: SentimentAnalysisRequest, api_key: str = Security(get_api_key)):
        """
        Extract sentiment from  a given document

        :param document: A document object with a *text*

        :return: response: An object with the sentiment e the score

        """
        service: SentimentAnalysisService = state_dict['service']

        sentiment, score = await service.extract_sentiment(document.text)
        return SentimentAnalysisResponse(sentiment=sentiment, score=score)


    return app


app = create_app()
