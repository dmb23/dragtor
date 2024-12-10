# pyright: reportIncompatibleVariableOverride = false
from pathlib import Path
import re
import shlex
import subprocess
from typing import Optional

import feedparser
from loguru import logger
from pydantic import BaseModel, Field
import requests

from dragtor import config
from dragtor.data import DataLoader
from dragtor.data.data import Document


class PodcastInformation(BaseModel):
    """Information about a podcast from iTunes"""

    name: str = Field(..., description="Name of the podcast", alias="collectionName")
    description: Optional[str] = Field(None, description="Podcast description")
    artwork_url: Optional[str] = Field(
        None, description="URL to podcast artwork", alias="artworkUrl100"
    )
    feed_url: Optional[str] = Field(None, description="RSS feed URL", alias="feedUrl")
    artist: Optional[str] = Field(None, alias="artistName")


class EpisodeInformation(BaseModel):
    podcast: PodcastInformation = Field(..., description="Corresponding podcast")
    title: str
    url: str
    guest: Optional[str]
    description: Optional[str] = Field(None, description="Episode description")
    release_date: Optional[str] = Field(None, description="Release date of episode")
    duration: Optional[str] = Field(None, description="Duration of episode")

    @property
    def ident(self) -> str:
        ident = f"{self.podcast.name}__{self.title}".replace("/", "_")
        return ident


def search_podcasts(podcast_title: str, episode_query: str) -> EpisodeInformation:
    """
    Search for podcast episodes using the iTunes Search API.
    `episode_query` needs to be verbatim in either the title or the description.

    Args:
        podcast_title: Name of the podcast to search for
        episode_query: Search terms to find specific episode

    Returns:
        PodcastInformation object with podcast and episode details

    Raises:
        ValueError: If no matching podcasts found
        requests.RequestException: If API request fails
    """
    # First search for the podcast
    base_url = "https://itunes.apple.com/search"
    podcast_params = {"term": podcast_title, "entity": "podcast", "limit": 1}

    response = requests.get(base_url, params=podcast_params)
    response.raise_for_status()

    results = response.json()
    if not results.get("results"):
        raise ValueError(f"No podcast found matching '{podcast_title}'")

    for k, v in results["results"][0].items():
        logger.debug(f"{k}: {v}")

    podcast = PodcastInformation.model_validate(results["results"][0])

    logger.debug(podcast)

    if not podcast.feed_url:
        raise ValueError(f"No RSS feed found for podcast '{podcast_title}'")

    feed = feedparser.parse(podcast.feed_url)
    for entry in feed.entries:
        if episode_query.lower() in entry.title.lower() or (
            hasattr(entry, "description") and episode_query.lower() in entry.description.lower()
        ):
            for k, v in sorted(entry.items()):
                logger.debug(f"{k}: {v}")
            guest = entry.author
            if guest == podcast.artist:
                guest = ""
            url = ""
            for l in entry.links:
                if l.type == "audio/mpeg":
                    url = l.href
                    break
            if url == "":
                raise ValueError(
                    f"Found no link to audio file for episode {episode_query}\n"
                    f"Available links: {entry.links}"
                )
            episode = EpisodeInformation(
                podcast=podcast,
                title=entry.title,
                guest=guest,
                url=url,
                description=entry.get("description"),
                release_date=entry.get("published"),
                duration=entry.get("itunes_duration"),
            )
            return episode
    raise ValueError(f"No episode found matching '{episode_query}' in podcast '{podcast_title}'")


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
        self._fname_template = "audio_{ident}.txt"

    def _get_podcast_episodes_from_config(self) -> list[EpisodeInformation]:
        podcast_config = config.conf.data.podcasts

        episodes = []
        for name in podcast_config:
            for url_query in podcast_config[name]:
                ep = search_podcasts(name, url_query)
                episodes.append(ep)

        return episodes

    def load_to_cache(self):
        """Load transcript from audio URL/path."""
        episodes = self._get_podcast_episodes_from_config()

        counter = 0
        for ep in episodes:
            logger.debug(f"loading podcast episode {ep.podcast} -> {ep.title}")

            fpath = self._cache_dir / self._fname_template.format(ident=ep.ident)
            try:
                fpath.read_text(encoding="utf8")
                logger.debug(f"Already cached {ep.title}")

            except IOError:
                # I actually don't know if it's always mp3 ...
                # NOTE: this could be changed into a tempfile if not needed for debugging
                raw_audio: Path = self._cache_dir / f"raw_audio_{ep.ident}.mp3"
                self._download_audio_file(ep.url, raw_audio)
                # convert the mp3 to WAV
                wav_audio: Path = self._cache_dir / f"raw_audio_{ep.ident}.wav"
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

    def get_cache(self) -> list[Document]:
        """Get all previously cached text that was loaded to file"""
        docs = []
        for ep in self._get_podcast_episodes_from_config():
            fpath = self._cache_dir / self._fname_template.format(ident=ep.ident)
            metadata = ep.model_dump()
            doc = Document(
                content=fpath.read_text(encoding="utf8"),
                title=ep.title,
                id=ep.ident,
                author=ep.podcast.artist,
                metadata=metadata,
            )
            logger.debug(f"loading cached file for {ep.title}")
            docs.append(doc)

        return docs

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
