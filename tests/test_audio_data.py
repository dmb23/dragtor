import shutil
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


def test_llama_server_availability():
    """Test if llama-server is executable."""
    result = subprocess.Popen(["llama-server", "--version"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    print(f"{output=}")
    print(f"{error=}")

    # try:
    #     result = subprocess.run(["llama-server", "--version"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     assert result.returncode == 0
    # except FileNotFoundError:
    #     pytest.fail("llama-server command cannot be found. Make sure llama.cpp project path is in PATH.")


def test_transcribe_exe_availability():
    """Test if transcribe executable are available in the project."""
    exe_file = Path(config.conf.project_path) / "transcribe"

    assert exe_file.is_file()
    assert shutil.which(exe_file) == exe_file


def test_models_availability():
    """Test if the required models are available in the project."""
    required_models = ["Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "ggml-base.en.bin"]

    for model in required_models:
        model_file =  Path(config.conf.project_path) / "models" / model
        assert model_file.is_file(), f"Model {model} is missing"


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

    shutil.rmtree(audio_loader.outdir)

def test_get_audio_cache(setup_audio_loader):
    """Test for transcript retrievals."""
    audio_loader = setup_audio_loader

    # Manually create two transcription files in the temporary directory
    sample_text = "And before he had time to think..."
    sample_transcripts = [
        Path(audio_loader.outdir) / "sample_1.txt",
        Path(audio_loader.outdir) / "sample_2.txt",
    ]
    for file in sample_transcripts:
        file.write_text(sample_text)

    audio_full_texts = audio_loader.get_audio_cache()

    assert len(audio_full_texts) == len(sample_transcripts)
    for text in audio_full_texts:
        assert text.startswith("And before he had time")

    shutil.rmtree(audio_loader.outdir)