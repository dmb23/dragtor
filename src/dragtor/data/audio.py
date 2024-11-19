from pathlib import Path
import re
import shlex
import subprocess

from loguru import logger
import requests

from dragtor import config
from dragtor.data import DataLoader


class AudioLoader(DataLoader):
    """
    Load Audio data and transcribe it to text files.

    Attributes
        model_path (str): The model file path used for transcription. Defaults to audio.model under config.
        language(str): Language used in the audio file.
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({"cache_glob": "audio_*.txt"})
        super().__init__(*args, **kwargs)

        self.model_path = config.conf.audio.model
        self.language = config.conf.audio.lang

    def load_to_cache(self):
        """Load transcript from audio URL/path."""
        urls: dict[str, str] = config.conf.data.audio_urls
        logger.debug(f"loading audio urls:\n{urls}")

        counter = 0
        for url_name, url in urls.items():
            fpath = self._cache_dir / f"audio_{url_name}.txt"
            try:
                fpath.read_text(encoding="utf8")
                logger.debug(f"Already cached {url}")
            except IOError:
                # I actually don't know if it's always mp3 ...
                # NOTE: this could be changed into a tempfile if not needed for debugging
                raw_audio: Path = self._cache_dir / f"raw_audio_{url_name}.mp3"
                self._download_audio_file(url, raw_audio)
                # convert the mp3 to WAV
                wav_audio: Path = self._cache_dir / f"raw_audio_{url_name}.wav"
                self._convert_to_wav(raw_audio, wav_audio)
                # transcribe the WAV
                transcript: str = self._transcribe_audio(wav_audio)
                transcript = self._clean_transcript(transcript)
                fpath.write_text(transcript)
                counter += 1

        if counter:
            logger.info(f"Loaded and transcribed {counter} new audio file")
        else:
            logger.info("No new audio file configured to cache")

    def _download_audio_file(self, audio_url: str, targetfile: Path):
        """Download audio file from URL and return a tempfile."""
        response = requests.get(audio_url)
        response.raise_for_status()

        inputs = response.content
        with targetfile.open("wb") as f:
            f.write(inputs)
        logger.debug(f"Downloaded audio from {audio_url} to {targetfile}")

    def _convert_to_wav(self, input_file: Path, output_file: Path):
        """Converts audio file into .wav format using FFMPEG executable

        Hardcoded 16kHz sampling."""
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file.resolve()),
            "-ar",
            "16000",
            str(output_file.resolve()),
        ]
        try:
            subprocess.run(
                command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as e:
            logger.error("Error converting downloaded audio to WAV")
            raise e
        logger.debug(f"Converted audio input to WAV: {output_file}")

    def _transcribe_audio(self, wav_file: Path) -> str:
        """Transcribe .wav audio file using whisper.cpp executable"""
        command = shlex.join(
            [
                f"{config.conf.executables.whisper_project}/main",
                "-m",
                self.model_path,
                "-f",
                str(wav_file.resolve()),
                "-l",
                self.language,
            ]
        )
        try:
            result = subprocess.run(command, capture_output=True, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                "Whisper Transcription process failed for {wav_file}\n"
                "Make sure the transcription model is set correctly in the params!"
            )
            raise e
        decoded = result.stdout.decode("utf-8").strip()
        logger.debug(f"successfully transcribed {wav_file}")

        return decoded

    def _clean_transcript(self, transcript: str) -> str:
        """Clean transcript output coming from Whisper model.

        Whisper outputs in the format:
        [hh:mm:ss.xxx --> hh:mm:ss.xxx]   Text text text ...

        Whisper adds certain markers
        """
        # Regex to match [hh:mm:ss.xxx --> hh:mm:ss.xxx] and the following text
        pattern = r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s+(.+)"

        transcript_segments = []
        for match in re.finditer(pattern, transcript):
            text = match.group(1).strip()
            # text = re.sub(r"[^\w.,'?\-\s]", "", text)
            transcript_segments.append(text)

        full_text = "\n".join(transcript_segments)
        full_text = full_text.replace("[Music]", "")
        full_text = full_text.replace("[BLANK_AUDIO]", "")

        return full_text
