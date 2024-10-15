from dragtor.utils.audio_utils import *
from dragtor.utils.audio_diarization_utils import *

from pathlib import Path
from loguru import logger
from requests.exceptions import RequestException
from dragtor import config


class AudioLoader:
    def __init__(self, outdir=None):
        self.model_path = config.conf.audio.model
        self.outdir: Path
        if outdir:
            self.outdir = Path(outdir)
        else:
            self.outdir = Path(config.conf.base_path) / config.conf.data.audio_cache

    def transcribe(self, audio_path: str, language: str = "en", diarize: bool = False, num_speakers: int = 1, min_speakers: int = 1, max_speakers: int = 1) -> str:
        self.outdir.mkdir(exist_ok=True)

        try:
            if audio_path.startswith(("http://", "https://")):
                response = requests.get(audio_path)
                response.raise_for_status()
            elif not audio_path.startswith(("http://", "https://")) and not Path(audio_path).exists():
                raise FileNotFoundError(f"{audio_path} not found")
        except RequestException as e:
            logger.error(f"{audio_path} is not a valid URL")
        except FileNotFoundError as e:
            logger.error(f"{e}: {audio_path}")
        else:
            expected_output_file = self.outdir / f"{Path(audio_path).parts[-2]}_{Path(audio_path).stem}.txt"

            if expected_output_file.exists():
                processed_str = expected_output_file.read_text()
                logger.info(f"Already cached {audio_path}")
            else:
                audio_path = preprocess_audio_file(audio_path).as_posix()

                logger.info(f"Transcribing {audio_path}")
                command = [
                    "transcribe", "-m", self.model_path, "-f", audio_path, "-l", language,
                ]
                process = subprocess.Popen(" ".join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, error = process.communicate()
                # logger.debug(f"{error=}")

                decoded_str = output.decode('utf-8').strip()
                processed_str = decoded_str.replace('[BLANK_AUDIO]', '').strip()

                if diarize:
                    diarize_str = audio_diarize(audio_path, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
                    # Standardize whisper.cpp & pyannote output
                    audio_segments = parse_audio_transcript(processed_str)
                    diarize_segments = parse_diarization(diarize_str)
                    # Match transcript & speaker
                    align_transcription_with_speakers(audio_path, audio_segments, diarize_segments)
                else:
                    parsed_audio = parse_transcript(processed_str)

                    output_file = self.outdir / f"{Path(audio_path).stem}.txt"
                    output_file.write_text(" ".join(parsed_audio))
                    logger.debug(f"Transcript saved at {output_file}")

                cleanup()

            return processed_str

    def get_audio_cache(self) -> list[str]:
        """Get all previously cached text that was loaded to file"""
        full_texts = []
        for fpath in self.outdir.glob("*.txt"):
            logger.warning(fpath)
            full_texts.append(fpath.read_text(encoding="utf8"))

        return full_texts


# pl = AudioLoader()
# for url in config.conf.data.audio_urls:
#     pl.transcribe(url, diarize=False)

# MANUAL DIARIZATION
# with open("whisper-cpp/transcripts/default_tc.txt") as f:
#     processed = f.readlines()
#     processed_str = "\n".join(processed)
# audio_segments = parse_audio_transcript(processed_str)
#
# with open("pyannote/default_tc.txt") as f:
#     diarize = f.readlines()
#     diarize_str = "\n".join(diarize)
#
# diarize_segments = parse_diarization(diarize_str)
# align_transcription_with_speakers("audio_files/default_tc.wav", audio_segments, diarize_segments)