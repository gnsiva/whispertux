"""Unit tests for WhisperManager transcription timeout behaviour."""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.whisper_manager import WhisperManager


def _make_manager(timeout_value=60):
    """Return a WhisperManager with a mocked config and stubbed paths."""
    config = MagicMock()
    config.get_setting.side_effect = lambda key, default=None: (
        timeout_value if key == 'transcription_timeout' else default
    )
    config.get_whisper_binary_path.return_value = Path('/fake/whisper-cli')
    config.get_temp_directory.return_value = Path('/tmp')
    config.get_whisper_model_path.return_value = Path('/fake/model.bin')

    manager = WhisperManager(config_manager=config)
    manager.whisper_binary = Path('/fake/whisper-cli')
    manager.model_path = Path('/fake/model.bin')
    manager.temp_dir = Path('/tmp')
    manager.ready = True
    return manager


class TestTranscriptionTimeout(unittest.TestCase):

    def test_timeout_returns_sentinel(self):
        """TimeoutExpired from subprocess must return the __TIMEOUT__ sentinel."""
        manager = _make_manager()
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(cmd='whisper', timeout=60)):
            result = manager._run_whisper('/fake/audio.wav')
        self.assertEqual(result, '__TIMEOUT__')

    def test_config_timeout_used(self):
        """The timeout passed to subprocess.run must come from config."""
        manager = _make_manager(timeout_value=120)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = 'hello world'
        with patch('subprocess.run', return_value=mock_result) as mock_run, \
             patch('os.path.exists', return_value=False):
            manager._run_whisper('/fake/audio.wav')
        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs['timeout'], 120)

    def test_default_timeout_60s(self):
        """Config default for transcription_timeout must be 60."""
        from src.config_manager import ConfigManager
        cm = ConfigManager()
        self.assertEqual(cm.default_config['transcription_timeout'], 60)

    def test_sentinel_propagates_through_transcribe_audio(self):
        """__TIMEOUT__ returned by _run_whisper must be passed through transcribe_audio."""
        import numpy as np
        manager = _make_manager()
        with patch.object(manager, '_run_whisper', return_value='__TIMEOUT__'), \
             patch.object(manager, '_save_audio_as_wav'), \
             patch('tempfile.NamedTemporaryFile') as mock_tmp, \
             patch('os.unlink'):
            mock_tmp.return_value.__enter__.return_value.name = '/tmp/fake.wav'
            audio = np.zeros(16000, dtype=np.float32)
            result = manager.transcribe_audio(audio, sample_rate=16000)
        self.assertEqual(result, '__TIMEOUT__')


if __name__ == '__main__':
    unittest.main()
