import pytest

from wikisearch_agent.util import (
       setup_logging,
)

class TestUtil:

    def test_logging_creates_files(self, tmp_path):
        """Test that logging creates files in the specified directory."""
        logdir = tmp_path / 'logs'
        logger = setup_logging(loglevel='DEBUG', logdir=logdir)
        logger.debug('This is a debug message')
        logger.info('This is an info message')
        assert logdir.exists(), f'Log directory {str(logdir)} should exist.'
        for logfile in ('debug.json.log', 'error.json.log'):
            assert (logdir / logfile).exists(), f'Log file {logfile} should exist in {str(logdir)}.'

    def test_logging_writes_to_files(self, tmp_path):
        logdir = tmp_path / 'logs'
        logger = setup_logging(loglevel='ERROR', logdir=logdir)
        logger.debug('Debug message to file')
        logger.info('Info message to file')
        logger.error('Error message to file')
        # Check that only the error message is in the error log file.
        err_log_contents = (logdir / 'error.json.log').read_text()
        assert 'Error message to file' in err_log_contents, 'Error message should be present in the error log file.'
        assert 'Debug message to file' not in err_log_contents, 'Debug message should not be present in the error log file.'
        assert 'Info message to file' not in err_log_contents, 'Info message should not be present in the error log file.'
        debug_log_contents = (logdir / 'debug.json.log').read_text()
        assert 'Error message to file' in debug_log_contents, 'Error message should be present in the debug log file.'
        assert 'Info message to file' in debug_log_contents, 'Info message should be present in the debug log file.'
        assert 'Debug message to file' in debug_log_contents, 'Debug message should be present in the debug log file.'

    def test_logging_writes_to_stderr(self, capsys, tmp_path):
        """Test that logging writes to stderr."""
        logdir = tmp_path / 'logs'
        logger = setup_logging(loglevel='DEBUG', logdir=logdir)
        logger.debug('Debug message to stderr')
        logger.info('Info message to stderr')
        captured = capsys.readouterr()
        assert 'Debug message to stderr' in captured.err
        assert 'Info message to stderr' in captured.err
