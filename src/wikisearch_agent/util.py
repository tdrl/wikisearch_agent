"""Shared utility functions."""

import keyring  # Accessing API keys securely from Apple Keychain or Windows Credential Store.
import structlog
import logging
import logging.config
from pathlib import Path
import os
import sys
from dataclasses import dataclass
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
import yaml


__all__ = [
    'setup_logging',
    'fetch_api_keys',
    'ApplicationSecrets',
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
    logging.getLogger('keyring').setLevel(logging.WARNING)
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

@dataclass
class ApplicationSecrets:
    """Container for secrets maintained in a central secrets store like AWS Secrets Manager or Apple Keychain."""
    langsmith_api: Optional[str] = None
    openai_api: Optional[str] = None
    openai_project_id: Optional[str] = None
    wikimedia_access_token: Optional[str] = None


def fetch_api_keys() -> ApplicationSecrets:
    """Fetch a set of API keys from the system's keyring.

    This pulls a set of keys that the agent will need from the local keyring, returning the ones
    it finds in an ApplicationSecrets object; non-existent or permission denied keys are returned as None.

    Returns:
        ApplicationSecrets
    """
    result = ApplicationSecrets(
        langsmith_api=keyring.get_password(service_name='net.illation.heather/langsmith', username='terran.lane@gmail.com'),
        openai_api=keyring.get_password(service_name='net.illation.heather/openai/prototyping', username='heather'),
        openai_project_id=keyring.get_password(service_name='net.illation.heather/openai/project_ids/wikisearch', username='heather'),
        wikimedia_access_token=keyring.get_password('net.illation.heather/wikimedia/access_token/AgenticWikisearch', username='Lady Dataslayer')
    )
    return result

def prompt_template_from_file(file: Path) -> ChatPromptTemplate:
    """Load a set of messages from a YAML file and convert them to a PromptTemplate.

    This loads a YAML file containing a sequence of role-typed prompt messages
    and emits them as a Langchain PromptTemplate. The YAML file should be structured
    like:
        - - role1
          - msg1
        - - role2
          - msg2
        ...

    Args:
        file (Path): Path to a YAML file containing sequence of messages.

    Returns:
        ChatPromptTemplate: Messages reformatted into a prompt template.
    """
    with file.open('rt') as d_in:
        raw = yaml.safe_load(d_in)
    return ChatPromptTemplate([(role, msg) for role, msg in raw])
