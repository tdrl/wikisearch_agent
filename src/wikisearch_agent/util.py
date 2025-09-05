"""Shared utility functions."""

import keyring  # Accessing API keys securely from Apple Keychain or Windows Credential Store.
import structlog
import logging
import logging.config
from pathlib import Path
import os
import sys

__all__ = [
    'setup_logging',
    'fetch_api_keys',
]


def get_default_working_dir() -> Path:
    """Get the default working directory based on the current script name."""
    return (Path('/tmp/') /
            os.getenv('USER', 'unknown_user') /
            Path(sys.argv[0]).stem)


def setup_logging(loglevel: str = 'INFO',
                  logdir: Path = get_default_working_dir() / 'logs') -> structlog.BoundLogger:
    """Set up logging configuration.

    Sets a local logger and configures the logging format.

    Args:
        loglevel (str): The logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        logdir (Path): The directory where log files will be stored.

    Returns:
        structlog.BoundLogger: A logger instance configured with the specified settings.
    """
    # Suppress asyncio 'KqueueSelector' messages.
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    # Ensure logdir exists
    logdir.mkdir(parents=True, exist_ok=True)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt='iso'),
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'console_colored': {
                '()': structlog.stdlib.ProcessorFormatter,
                'processor': structlog.dev.ConsoleRenderer(colors=True),
            },
            'json': {
                '()': structlog.stdlib.ProcessorFormatter,
                'processor': structlog.processors.JSONRenderer(),
            },
        },
        'handlers': {
            'console': {
                'level': loglevel,
                'class': 'logging.StreamHandler',
                'formatter': 'console_colored',
            },
            'file_debug': {
                'level': 'DEBUG',
                'class': 'logging.handlers.WatchedFileHandler',
                'filename': str(logdir / 'debug.json.log'),
                'formatter': 'json',
            },
            'file_error': {
                'level': 'ERROR',
                'class': 'logging.handlers.WatchedFileHandler',
                'filename': str(logdir / 'error.json.log'),
                'formatter': 'json',
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file_debug', 'file_error'],
                'level': 'DEBUG',
                'propagate': True,
            },
        },
    })
    return structlog.get_logger(__name__)


def fetch_api_keys() -> dict[str, str]:
    """Fetch the API key for Hugging Face from the system's keyring.

    Currently, it retrieves the Hugging Face access token.

    Returns:
        dict[str, str]: A dictionary mapping service names to API keys.
            If no keys are found, returns an empty dictionary. Currently supported keys:
                - 'huggingface': The Hugging Face API access token.
    """
    api_key = keyring.get_password('net.illation.heather/huggingface/exploration',
                                   'studentbane')
    return {'huggingface': api_key} if api_key is not None else {}
