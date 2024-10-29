import subprocess
import tempfile
import requests
import re

from typing import Iterable
from pathlib import Path
from loguru import logger
from requests.exceptions import RequestException
from dragtor import config


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

        # Make sure output folder exists
        self.outdir.mkdir(exist_ok=True)

    def load_audio_to_cache(self, urls: Iterable[str] | str):
        """Load transcript from audio URL/path."""
        if isinstance(urls, str):
            urls = [urls]

        for url in urls:
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

            self._transcribe_to_file(audio_path=url)


    def get_audio_cache(self) -> list[str]:
        """Get all previously cached audio transcript that was loaded to file"""
        full_texts = []
        for fpath in self.outdir.glob("*.txt"):
            logger.info(f"loading cached file {fpath}")
            full_texts.append(fpath.read_text(encoding="utf8"))

        return full_texts


    def _transcribe_to_file(self, audio_path: str) -> bool:
        """
        Transcribe the audio file or URL provided in the config file and save it to a file.

        Args:
            audio_path (str): Path/URL to the audio to be transcribed.

        Returns:
            None: This function does not return any value. The transcription is saved as a file.
        """
        logger.info(f"Starting transcription for {audio_path}")

        # Set the output file name by extracting the last two parts and set file extension as txt
        file_name = self._clean_filename("_".join(audio_path.split("/")[-2:]))
        output_file = self.outdir / f"{file_name.rsplit('.', 1)[0]}.txt"

        # If the same URL/audio file has been transcribed, it will read from existing transcription
        if output_file.exists():
            logger.info(f"Already cached {audio_path}")
            return True

        if audio_path.startswith(("http://", "https://")):
            with self._download_audio_file(audio_path=audio_path) as temp_file:
                with tempfile.TemporaryDirectory(dir=config.conf.base_path) as tmpdir:
                    wav_file = self._convert_to_wav(input_file=Path(temp_file.name), output_dir=Path(tmpdir), output_filename=file_name)
                    transcript = self._transcribe_audio(wav_file=wav_file)
                    self._save_transcript(transcript, output_file)
        else:
            with tempfile.TemporaryDirectory(dir=config.conf.base_path) as tmpdir:
                wav_file = self._convert_to_wav(input_file=Path(audio_path), output_dir=Path(tmpdir), output_filename=file_name)
                transcript = self._transcribe_audio(wav_file=wav_file)
                self._save_transcript(transcript, output_file)

        logger.info(f"Completed transcription for {audio_path}")


    def _download_audio_file(self, audio_path: str):
        """Download audio file from URL and return a tempfile."""
        inputs = requests.get(audio_path).content
        temp_file = tempfile.NamedTemporaryFile(delete=True, dir=config.conf.base_path)
        temp_file.write(inputs)
        temp_file.flush()
        return temp_file


    def _convert_to_wav(self, input_file: Path, output_dir: Path, output_filename: str) -> Path:
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


    def _transcribe_audio(self, wav_file) -> str:
        """Transcribe .wav audio file using transcription model."""
        command = f"{config.conf.executables.whisper_project}/main -m '{self.model_path}' -f '{str(wav_file)}' -l '{self.language}'"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Transcription process failed for {wav_file} with error {error.decode('utf-8')}. Make sure transcription model is available under models/")

        decoded_str = output.decode('utf-8').strip()
        processed_str = decoded_str.replace('[BLANK_AUDIO]', '').strip()
        return processed_str


    def _save_transcript(self, transcript, output_file: Path):
        """Write the transcript to a text file in the specified directory."""
        parsed_audio = self._parse_transcript(transcript)
        output_file.write_text(" ".join(parsed_audio))
        logger.debug(f"Transcript saved at {output_file}")


    def _parse_transcript(self, transcript: str) -> list[str]:
        """Remove timestamp from raw transcription output and collect it as a paragraph."""
        # Regex to match [hh:mm:ss.xxx --> hh:mm:ss.xxx] and the following text
        pattern = r"\[(\d+):(\d+):([\d.]+)\.\d+ --> (\d+):(\d+):([\d.]+)\.\d+\]\s+(.+)"

        transcript_segments = []
        for match in re.finditer(pattern, transcript):
            text = match.group(7).strip()
            clean_text = re.sub(r"[^\w.,'?\-\s]", "", text)
            transcript_segments.append(clean_text)

        return transcript_segments


    def _clean_filename(self, name: str) -> str:
        """Cleans file name to remove illegal characters."""
        return re.sub(r'[<>:"/\\|?*]', "_", name)
