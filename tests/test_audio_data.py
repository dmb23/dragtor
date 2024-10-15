from os.path import isfile

import pytest
import subprocess
from pathlib import Path
from dragtor.utils.audio_utils import preprocess_audio_file
from dragtor.audio_loader import AudioLoader
from unittest import mock

# Sample paths to test
project_root = Path(__file__).parent.parent
local_audio_file = project_root / "tests" / "assets" / "audio_files" / "sample2.flac"
transcript_file = project_root / "tests" / "assets" / "audio_transcript" / "sample2.txt"

# Mock the response for requests.get to avoid real HTTP calls
@pytest.fixture
def mock_request_get(monkeypatch):
    def mock_get(*args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.content = f"fake_audio_content"

            def raise_for_status(self):
                pass
        return MockResponse()

    monkeypatch.setattr("requests.get", mock_get)

# Mock the subprocess call to ffmpeg to avoid real processing
@pytest.fixture
def mock_subprocess_run(monkeypatch):
    def mock_run(*args, **kwargs):
        return mock.Mock()

    monkeypatch.setattr("subprocess.run", mock_run)

# Test for preprocess_audio_file function with a URL
def test_preprocess_audio_file(mock_request_get, mock_subprocess_run):
    result = preprocess_audio_file(str(local_audio_file))

    assert isinstance(result, Path)
    assert result.suffix == ".wav"


@pytest.fixture
def mock_subprocess_popen(monkeypatch):
    class MockPopen:
        def __init__(self, *args, **kwargs):
            self.stdout = b""  # Simulated stdout result
            self.stderr = b""  # Simulated no error

        def communicate(self):
            transcript_sample = """
            [00:03:30.840 --> 00:03:36.720]   A lot of people have injuries and niggles and things that they're confused about and I think we got
            [00:03:36.720 --> 00:03:42.520]   a lot of great questions that I think will lead into hopefully kind of more global helpful
            [00:03:42.520 --> 00:03:47.120]   takeaways for how to prevent these injuries, how to deal with things like lumbricle injuries
            [00:03:47.120 --> 00:03:53.280]   or cinnavitis or capsulitis or hypermobile fingers and how you train if you have super
            [00:03:53.280 --> 00:03:58.560]   mobile fingers and then like injury prevention for fingers and how to promote finger health
            [00:03:58.560 --> 00:04:04.000]   for all of us for aging climbers, etc so yeah, I'm really looking forward to it but as a
            [00:04:04.000 --> 00:04:10.480]   way to kick things off, tell me a little bit about your background.
            """
            self.stdout = transcript_sample.encode("utf-8")
            return self.stdout, self.stderr  # Return stdout and stderr as tuple

        def wait(self):
            return None

        # Add context manager methods
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    # Mock the Popen object within subprocess
    monkeypatch.setattr(subprocess, "Popen", MockPopen)

# Test the transcribe method in PodcastLoader with mocked Popen
def test_transcribe_method(mock_request_get, mock_subprocess_run, mock_subprocess_popen):
    loader = AudioLoader()

    # Call the method using a valid local audio file path
    result = loader.transcribe(str(local_audio_file))

    # Assertions
    assert isinstance(result, str)
    assert len(result) > 0
    assert result[:31] == "[00:03:30.840 --> 00:03:36.720]"
    assert isfile(transcript_file)