import pytest
import subprocess

from importlib.resources import as_file, files
from pathlib import Path
from dragtor.audio_loader import AudioLoader
from dragtor import config

def test_ffmpeg_availability():
    """Test if your local machine already installed with ffmpeg."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0
    except FileNotFoundError:
        pytest.fail("ffmpeg is not installed or not available in the system's PATH")

@pytest.fixture
def setup_audio_loader():
    audio_loader = AudioLoader(outdir=Path(config.conf.base_path) / config.conf.data.audio_cache)
    yield audio_loader

@pytest.mark.parametrize(
    "test_url, expected_output",
    [
        ("https://cdn-media.huggingface.co/speech_samples/sample2.flac", True),
        ("sample.wav", True),
        ("https://cdn-media.huggingface.co/speech_samples/samplex.flac", False),
        ("sample99.wav", False),
    ]
)
def test_transcribe_to_file(setup_audio_loader, test_url, expected_output):
    audio_loader = setup_audio_loader

    # Check if the test URL is a local file in assets, and use importlib.resources to locate it
    if test_url in ["sample.wav", "sample99.mp3"]:
        with as_file(files("tests.assets.audio_sample").joinpath(test_url)) as local_audio_sample:
            test_url = str(local_audio_sample)

    # Run main function of AudioLoader for each URL
    audio_loader.load_audio_to_cache(test_url)

    # Check if the transcription is created & match the file name logic
    file_name = "_".join(test_url.split("/")[-2:])
    output_file = Path(audio_loader.outdir) / f"{file_name.rsplit('.', 1)[0]}.txt"

    if expected_output:
        # Running using correct URL
        assert output_file.exists()
        assert output_file.read_text().startswith("And before he had time") == True
    else:
        # Running using incorrect URL
        assert not output_file.exists()

def test_get_audio_cache(setup_audio_loader):
    """Test for transcript retrievals. Make sure to run this after previous test!!!"""
    audio_loader = setup_audio_loader

    audio_full_texts = audio_loader.get_audio_cache()

    assert len(audio_full_texts) == 2
    assert audio_full_texts[0].startswith("And before he had time") == True
    assert audio_full_texts[1].startswith("And before he had time") == True

    for file in audio_loader.outdir.glob("*.txt"):
        if file.name in ["audio_sample_sample.txt", "speech_samples_sample2.txt"]:
            file.unlink()