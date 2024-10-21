from dragtor.audio.audio_utils import preprocess_audio_file, parse_transcript, cleanup
from dragtor.audio.audio_diarization_utils import audio_diarize, parse_audio_transcript, parse_diarization, align_transcription_with_speakers

import subprocess
import requests
from pathlib import Path
from loguru import logger
from requests.exceptions import RequestException
from dragtor import config


class AudioLoader:
    """
    A class to handle audio transcription and audio diarization (optional), then saving the result to files.

    Attributes
        model (str): The model file path used for transcription. Defaults to audio.model under config.
        outdir (str): Output path where the results are stored. Defaults to base_path/data.audio_cache under config.

    Methods
        transcribe_to_file(audio_path): Transcribes the audio and saves the result to a file. Set diarize=True to use diarization functionality.
    """

    def __init__(self, outdir=None):
        self.model_path = config.conf.audio.model
        self.outdir: Path
        if outdir:
            self.outdir = Path(outdir)
        else:
            self.outdir = Path(config.conf.base_path) / config.conf.data.audio_cache

    def transcribe_to_file(self, audio_path: str, language: str = None, diarize: bool = False, num_speakers: int = None,
                           min_speakers: int = None, max_speakers: int = None) -> None:
        """
        Transcribe the audio file or URL provided in the config file and save it to a file.

        Input:
            audio_path (str): Path/URL to the audio to be transcribed.
            output_dir (str, optional): Directory where the transcription file will be saved. Defaults to "data/audio_transcript".
            language (str, optional): The language of spoken audio. Default to "en" (English).
            diarize (bool, optional): Whether to perform speaker diarization. Default to False.

        Returns:
            None: This function does not return any value. The transcription is saved as a file.
        """
        # Set default values from config if not provided
        language = language or config.conf.audio.lang
        num_speakers = num_speakers or config.conf.audio.num_speakers
        min_speakers = min_speakers or config.conf.audio.min_speakers
        max_speakers = max_speakers or config.conf.audio.max_speakers

        self.outdir.mkdir(exist_ok=True)

        try:
            # Check whether the provided URL is valid
            if audio_path.startswith(("http://", "https://")):
                response = requests.get(audio_path)
                response.raise_for_status()
            # Check whether the provided file path exists
            elif not audio_path.startswith(("http://", "https://")) and not Path(audio_path).exists():
                raise FileNotFoundError(f"{audio_path} not found")
        except RequestException as e:
            logger.error(f"{audio_path} is not a valid URL")
        except FileNotFoundError as e:
            logger.error(f"{e}: {audio_path}")

        # If the same URL/audio file has been transcribed, it will read from existing transcription
        expected_output_file = self.outdir / f"{Path(audio_path).parts[-2]}_{Path(audio_path).stem}.txt"
        if expected_output_file.exists():
            logger.info(f"Already cached {audio_path}")

        audio_path = str(preprocess_audio_file(audio_path))

        logger.info(f"Transcribing {audio_path}")
        command = [
            "transcribe", "-m", self.model_path, "-f", audio_path, "-l", language,
        ]
        process = subprocess.Popen(" ".join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        decoded_str = output.decode('utf-8').strip()
        processed_str = decoded_str.replace('[BLANK_AUDIO]', '').strip()

        if diarize:
            diarize_str = audio_diarize(audio_path, num_speakers=num_speakers, min_speakers=min_speakers,
                                        max_speakers=max_speakers)
            # Standardize the timestamp format of transcription & speaker diarization output
            audio_segments = parse_audio_transcript(processed_str)
            diarize_segments = parse_diarization(diarize_str)
            # Match transcript with its speaker using timestamp
            align_transcription_with_speakers(audio_path, audio_segments, diarize_segments)
        else:
            # Remove timestamp and combine lines of transcript into one paragraph
            parsed_audio = parse_transcript(processed_str)

            output_file = self.outdir / f"{Path(audio_path).stem}.txt"
            output_file.write_text(" ".join(parsed_audio))
            logger.debug(f"Transcript saved at {output_file}")

        cleanup()

    def get_audio_cache(self) -> list[str]:
        """Get all previously cached audio transcript that was loaded to file"""
        full_texts = []
        for fpath in self.outdir.glob("*.txt"):
            logger.warning(fpath)
            full_texts.append(fpath.read_text(encoding="utf8"))

        return full_texts
