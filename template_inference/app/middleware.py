import time
import re
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from logging_utils import AUDIT_LOGGER


class LoggingMiddleWare(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        response = await call_next(request)
        process_time = time.time() - start_time
        try:
            endpoint = re.findall("https:\/\/[\w\d.]+:\d+((\/\w*)*)",  str(request.url))[0][0]
            AUDIT_LOGGER.info(f'HOST: {request.client.host}, '
                              f'METHOD: {request.method}, '
                              f'ENDPOINT: {endpoint}, '
                              f'ELAPSED_TIME: {int(process_time * 1000)}ms, '
                              f'STATUS_CODE: {response.status_code}, '
                              f'BYTES_REQUEST: {request.headers["content-length"] if "content-length" in request.headers else 0},'
                              f'BYTES_RESPONSE: {response.headers["content-length"]}')
        except:
            AUDIT_LOGGER.info(f'HOST: {request.client.host}, '
                              f'METHOD: {request.method}, '
                              f'ENDPOINT: {str(request.url)}, '
                              f'ELAPSED_TIME: {int(process_time * 1000)}ms, '
                              f'STATUS_CODE: {response.status_code}, '
                              f'BYTES_REQUEST: {request.headers["content-length"] if "content-length" in request.headers else 0},'
                              f'BYTES_RESPONSE: {response.headers["content-length"]}')
        return response
