import subprocess
import requests
import shutil
import re

from loguru import logger
from pathlib import Path
from dragtor import config

def preprocess_audio_file(audio_path: str) -> Path:
    output_dir = Path(config.conf.audio.files)
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Processing: {audio_path}")
    if audio_path.startswith(("http://", "https://")):
        inputs = requests.get(audio_path).content
        file_path = output_dir / "_".join(Path(audio_path).parts[-2:])
        file_path.write_bytes(inputs)
        logger.debug(f"File downloaded and saved at {file_path}")
        file_path = convert_to_wav(file_path)
    else:
        file_path = convert_to_wav(Path(audio_path))

    return file_path

def convert_to_wav(audio_path: Path) -> Path:
    if audio_path.suffix.lower() != ".wav":
        wav_file = audio_path.with_suffix(".wav")
        command = ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", str(wav_file)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_file
    return audio_path

def parse_transcript(transcript: str):
    # Regex to match [hh:mm:ss.xxx --> hh:mm:ss.xxx] and the following text
    pattern = r"\[(\d+):(\d+):([\d.]+)\.\d+ --> (\d+):(\d+):([\d.]+)\.\d+\]\s+(.+)"

    transcript_segments = []
    for match in re.finditer(pattern, transcript):
        text = match.group(7).strip()

        transcript_segments.append(text)

    return transcript_segments

def cleanup():
    delete_path = [Path(config.conf.base_path) / config.conf.data.diarize_cache, Path(config.conf.audio.files)]
    for path in delete_path:
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            logger.info(f"Cleared cache: {path}")
        else:
            logger.info(f"Folder not found: {path}")

