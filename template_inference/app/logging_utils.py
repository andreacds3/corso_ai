import logging
import os
from enum import Enum
import logging.config as config
class LoggerType(str, Enum):
    AUDIT = 'audit_logger'
    APPLICATION = 'application_logger'
    UVICORN = 'uvicorn_logger'

class LoggerFactory:

    _is_initialized: bool = False
    @classmethod
    def get_logger(cls, logger_type: LoggerType):
        if not cls._is_initialized:
            LoggerFactory.initialize_loggers()
            cls._is_initialized = True
        return logging.getLogger(logger_type.value)

    @staticmethod
    def initialize_loggers():
        logging.config.fileConfig(
            os.getenv("LOGGING_CONFIG_FILE", "/config/log.ini"),
            disable_existing_loggers=False
        )


AUDIT_LOGGER = LoggerFactory.get_logger(LoggerType.AUDIT)
UVICORN_LOGGER = LoggerFactory.get_logger(LoggerType.UVICORN)
APPLICATION_LOGGER = LoggerFactory.get_logger(LoggerType.APPLICATION)
