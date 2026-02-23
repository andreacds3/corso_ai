import asyncio
import hashlib
import logging
import time
from typing import List

from sentiment_analysis.exceptions import SentimentAnalysisErrorWithIds,SentimentAnalysisError
from sentiment_analysis.nlp import NLPSentimentPipeline

from logging_utils import AUDIT_LOGGER, UVICORN_LOGGER, APPLICATION_LOGGER


class Batcher:
    """
    Class representing a Batcher. A Batcher is a component that queues the incoming summarization requests and aggregates
    them in batches in order to call the model exploiting parallel capabilities of cpus and gpus
    """

    def __init__(self, nlp_pipeline: NLPSentimentPipeline, batch_size: int, batch_timeout: float):
        """

        :param nlp_pipeline: The summarization pipeline (tokenize + generate + decode)
        :param batch_size: The size of the batch
        :param batch_timeout: the timeout of the batching process: the number of seconds to wait before 'closing' the batch and proceeding to calling the pipeline
        """
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout
        self.nlp_pipeline = nlp_pipeline
        #The hashmap representing the queue with texts to be processed
        self._to_process = dict()
        # The hashmap representing the queue with text translated
        self._processed = dict()
        #The list of ids of text translation errors
        self._errors = list()

        self._batcher_task = None


    def start_runner(self):
        """Start the runner loop"""
        _ = asyncio.get_event_loop()
        self._batcher_task = asyncio.create_task(self._runner())

    def stop_runner(self):
        self._batcher_task.cancel()

    async def _runner(self):
        """
        Asyncronous batching task.
        It waits for batch_timout seconds or batch_size elements before calling the inner process_batch_function
        :return: None
        """

        async def process_batch():
            """process a batch of summarization requests"""
            batch_to_process = [(key, tuple_text) for key, tuple_text in self._to_process.items()][:self._batch_size]
            keys = {key for key,_ in batch_to_process}
            try:
                [self._to_process.pop(key) for key in keys]
                lists = tuple(list(x) for x in zip(*[tuple_text[1] for tuple_text in batch_to_process]))
                results = await self.nlp_pipeline.extract_sentiment(*lists)
                self._processed.update({batch[0]: result for batch, result in zip(batch_to_process, results)})
            except Exception as ex:
                APPLICATION_LOGGER.exception(ex)
                raise SentimentAnalysisErrorWithIds(error_ids=[batch[0] for batch in list(batch_to_process)])

        while True:
            time_out = time.perf_counter() + self._batch_timeout
            while time.perf_counter() < time_out:

                if len(self._to_process) >= self._batch_size:
                    try:
                        await process_batch()
                    except SentimentAnalysisErrorWithIds as e:
                        APPLICATION_LOGGER.exception(e)
                        self._errors.extend(e.error_ids)
                    except Exception as e:
                        APPLICATION_LOGGER.exception(e)
                await asyncio.sleep(0)
            else:
                if len(self._to_process) > 0:
                    try:
                        await process_batch()
                    except SentimentAnalysisErrorWithIds as e:
                        APPLICATION_LOGGER.exception(e)
                        self._errors.extend(e.error_ids)
                    except Exception as e:
                        APPLICATION_LOGGER.exception(e)

            await asyncio.sleep(0)

    async def extract_sentiment(self, text: str) -> str:
        """Send signal to batcher task to process translation on given texts"""
        id = hashlib.md5(text.encode())
        self._to_process[id] = (text,)
        while True:

            if id in self._errors:
                self._errors.remove(id)
                raise SentimentAnalysisError()
            elif id in self._processed:
                sentiment_tuple = self._processed.pop(id)
                return sentiment_tuple
            else:
                await asyncio.sleep(0)


