import subprocess
import requests
import re
import tempfile

from typing import Iterable, Any
from pathlib import Path
from loguru import logger
from requests.exceptions import RequestException
from dragtor import config

from dragtor.audio.diarization import _parse_transcript_to_diarization, _audio_diarize, _save_diarization_transcript
from dragtor.audio.transcription import _clean_filename, _download_audio_file, _convert_to_wav, _transcribe_audio, _save_transcript


class AudioLoader:
    """
    A class to handle audio transcription and audio diarization (optional), then saving the result to files.

    Attributes
        model_path (str): The model file path used for transcription. Defaults to audio.model under config.
        language(str): Language used in the audio file.
        outdir (str): Output path where the results are stored. Defaults to base_path/data.audio_cache under config.

    Methods
        load_audio_to_cache(audio_paths): Transcribes the audio URLs listed in config file and saves the result to a file.
        get_audio_cache: Retrieves all transcriptions into a list of string to be processed further (chunk, index, etc)
    """

    def __init__(self, outdir=None):
        self.model_path = config.conf.audio.model
        self.language = config.conf.audio.lang
        if outdir:
            self.outdir = Path(outdir)
        else:
            self.outdir = Path(config.conf.base_path) / config.conf.data.audio_cache

    def load_audio_to_cache(self, urls: Iterable[dict[str, Any]] | dict[str, Any]):
        """Load transcript from audio URL/path."""
        # diarize = config.conf.audio.diarize # TODO: Setting this on URL-level. Necessary?
        # Make sure output folder exists
        self.outdir.mkdir(exist_ok=True)

        if isinstance(urls, dict):
            urls = [urls]

        for entry in urls:
            url = entry["url"]
            diarize = entry["diarize"]

            # Check whether the provided URL is valid
            if url.startswith(("http://", "https://")):
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                except RequestException as e:
                    logger.error(f"{url} is not a valid URL")
                    continue
            # Check whether the provided file path exists
            elif not Path(url).exists():
                logger.error(f"{url} not found")
                continue

            self._transcribe_to_file(audio_path=url, diarize=diarize)


    def get_audio_cache(self) -> list[str]:
        """Get all previously cached audio transcript that was loaded to file"""
        full_texts = []
        for fpath in self.outdir.glob("*.txt"):
            logger.info(f"loading cached file {fpath}")
            full_texts.append(fpath.read_text(encoding="utf8"))

        return full_texts


    def _transcribe_to_file(self, audio_path: str, diarize: bool) -> bool:
        """
        Transcribe the audio file or URL provided in the config file and save it to a file.

        Args:
            audio_path (str): Path/URL to the audio to be transcribed.

        Returns:
            None: This function does not return any value. The transcription is saved as a file.
        """
        logger.info(f"Starting transcription for {audio_path} {'with diarize' if diarize else 'without diarize'}")

        # Set the output file name by extracting the last two parts and set file extension as txt
        file_name = _clean_filename("_".join(audio_path.split("/")[-2:]))
        if diarize:
            output_file = self.outdir / f"diarize_{file_name.rsplit('.', 1)[0]}.txt"
        else:
            output_file = self.outdir / f"{file_name.rsplit('.', 1)[0]}.txt"

        # If the same URL/audio file has been transcribed, it will read from existing transcription
        if output_file.exists():
            logger.info(f"Already cached {audio_path}")
            return True

        if audio_path.startswith(("http://", "https://")):
            with _download_audio_file(audio_path=audio_path) as temp_file:
                with tempfile.TemporaryDirectory(dir=config.conf.base_path) as tmpdir:
                    wav_file = _convert_to_wav(input_file=Path(temp_file.name), output_dir=Path(tmpdir), output_filename=file_name)
                    transcript = _transcribe_audio(wav_file=wav_file, model_path=self.model_path, language=self.language)
                    if diarize:
                        parsed_transcription = _parse_transcript_to_diarization(transcript)
                        parsed_diarization = _audio_diarize(wav_file)
                        _save_diarization_transcript(parsed_transcription, parsed_diarization, output_file)
                    else:
                        _save_transcript(transcript, output_file)
        else:
            with tempfile.TemporaryDirectory(dir=config.conf.base_path) as tmpdir:
                wav_file = _convert_to_wav(input_file=Path(audio_path), output_dir=Path(tmpdir), output_filename=file_name)
                transcript = _transcribe_audio(wav_file=wav_file, model_path=self.model_path, language=self.language)
                if diarize:
                    if diarize:
                        parsed_transcription = _parse_transcript_to_diarization(transcript)
                        parsed_diarization = _audio_diarize(wav_file)
                        _save_diarization_transcript(parsed_transcription, parsed_diarization, output_file)
                else:
                    _save_transcript(transcript, output_file)

        logger.info(f"Completed transcription for {audio_path}")
