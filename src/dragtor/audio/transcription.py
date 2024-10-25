import subprocess
import requests
import re
import tempfile

from typing import Iterable
from pathlib import Path
from loguru import logger
from requests.exceptions import RequestException
from dragtor import config


def _download_audio_file(audio_path: str):
    """Download audio file from URL and return a tempfile."""
    inputs = requests.get(audio_path).content
    temp_file = tempfile.NamedTemporaryFile(delete=True, dir=config.conf.base_path)
    temp_file.write(inputs)
    temp_file.flush()
    return temp_file


def _convert_to_wav(input_file: Path, output_dir: Path, output_filename: str) -> Path:
    """Converts audio file into .wav format with 16kHz sampling."""
    wav_file_name = output_dir / output_filename
    if wav_file_name.suffix.lower() != ".wav":
        wav_file = wav_file_name.with_suffix(".wav")
        command = ["ffmpeg", "-y", "-i", str(input_file), "-ar", "16000", str(wav_file)]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error converting to WAV: {e}")

        return wav_file
    return input_file


def _transcribe_audio(wav_file, model_path, language) -> str:
    """Transcribe .wav audio file using transcription model."""
    command = f"transcribe -m '{model_path}' -f '{str(wav_file)}' -l '{language}'"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Transcription process failed for {wav_file} with error {error.decode('utf-8')}. Make sure transcription model is available under models/")

    decoded_str = output.decode('utf-8').strip()
    processed_str = decoded_str.replace('[BLANK_AUDIO]', '').strip()
    return processed_str


def _save_transcript(transcript, output_file: Path):
    """Write the transcript to a text file in the specified directory."""
    parsed_audio = _parse_transcript_to_paragraph(transcript)
    output_file.write_text(" ".join(parsed_audio))
    logger.debug(f"Transcript saved at {output_file}")


def _parse_transcript_to_paragraph(transcript: str) -> list[str]:
    """Remove timestamp from raw transcription output and collect it as a paragraph."""
    # Regex to match [hh:mm:ss.xxx --> hh:mm:ss.xxx] and the following text
    pattern = r"\[(\d+):(\d+):([\d.]+)\.\d+ --> (\d+):(\d+):([\d.]+)\.\d+\]\s+(.+)"

    transcript_segments = []
    for match in re.finditer(pattern, transcript):
        text = match.group(7).strip()
        clean_text = re.sub(r"[^\w.,'?\-\s]", "", text)
        transcript_segments.append(clean_text)

    return transcript_segments


def _clean_filename(name: str) -> str:
    """Cleans file name to remove illegal characters."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)
