from pathlib import Path
import shutil
import subprocess

import pytest

from dragtor import config
from dragtor.data.audio import AudioLoader


def test_ffmpeg_availability():
    """Test if your local machine already installed with ffmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert result.returncode == 0
    except FileNotFoundError:
        pytest.fail("ffmpeg is not installed or not available in the system's PATH")


def test_whisper_availability():
    """Test if transcription function is executable.
    This will probably fail on windows systems..."""
    exec_file = Path(config.conf.executables.whisper_project) / "main"

    # Check if the file exists
    if not exec_file.is_file():
        pytest.fail(
            f"Transcribe executable not found at {exec_file}."
            " Ensure the whisper.cpp project path is configured correctly."
        )

    # Check if the file is executable
    if not shutil.which(str(exec_file)):
        pytest.fail(
            f"Transcribe executable at {exec_file} is not recognized as executable."
            " Verify permissions and path configuration."
        )

    try:
        result = subprocess.run(
            [str(exec_file), "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert (
            result.returncode == 0
        ), f"Transcribe executable at {exec_file} failed to run. Error: {result.stderr}"
    except FileNotFoundError as e:
        pytest.fail(f"Failed to run whisper.cpp main executable at {exec_file}: {e}")


def test_models_availability():
    """Test if the required models are available in the project.
    NOTE: this is not really helpful, since testing configuration intercepts original config.
    """
    required_models = [config.conf.model.file_path, config.conf.audio.model]

    for model in required_models:
        model_file = Path(config.conf.project_path) / "models" / model
        assert model_file.is_file(), f"Model {model} is missing"


@pytest.fixture
def audio_loader(tmp_path):
    orig_base = config.conf.base_path
    config.conf.base_path = str(tmp_path.resolve())
    audio_loader = AudioLoader()
    config.conf.base_path = orig_base
    yield audio_loader


def test_get_audio_cache(audio_loader):
    """Test for transcript retrievals."""

    # Manually create two transcription files in the temporary directory
    sample_text = "And before he had time to think..."
    sample_transcripts = [
        Path(audio_loader._cache_dir) / "audio_1.txt",
        Path(audio_loader._cache_dir) / "audio_2.txt",
    ]
    for file in sample_transcripts:
        file.write_text(sample_text)

    audio_full_texts = audio_loader.get_cache()

    assert len(audio_full_texts) == len(sample_transcripts)
    for text in audio_full_texts:
        assert text.startswith("And before he had time")

