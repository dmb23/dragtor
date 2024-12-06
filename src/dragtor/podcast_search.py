from typing import Optional

import feedparser
from loguru import logger
from pydantic import BaseModel, Field
import requests


class PodcastInformation(BaseModel):
    """Information about a podcast from iTunes"""

    podcast_name: str = Field(..., description="Name of the podcast")
    description: Optional[str] = Field(None, description="Podcast description")
    artwork_url: Optional[str] = Field(None, description="URL to podcast artwork")
    feed_url: Optional[str] = Field(None, description="RSS feed URL")
    artist: Optional[str]


class EpisodeInformation(BaseModel):
    podcast: PodcastInformation = Field(..., description="Corresponding podcast")
    title: str
    episode_nr: int
    description: Optional[str] = Field(None, description="Episode description")
    release_date: Optional[str] = Field(None, description="Release date of episode")
    duration: Optional[str] = Field(None, description="Duration of episode")


def search_podcasts(podcast_title: str, episode_query: str) -> EpisodeInformation:
    """
    Search for podcast episodes using the iTunes Search API.

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

    podcast = PodcastInformation.model_validate_json(results["results"][0])

    logger.debug(podcast)

    if not podcast.feed_url:
        raise ValueError(f"No RSS feed found for podcast '{podcast_title}'")

    feed = feedparser.parse(podcast.feed_url)
    for entry in feed.entries:
        if episode_query.lower() in entry.title.lower() or (
            hasattr(entry, "description") and episode_query.lower() in entry.description.lower()
        ):
            episode = EpisodeInformation(
                podcast=podcast,
                title=entry.title,
                episode_nr=len(feed.entries) - feed.entries.index(entry),
                description=entry.get("description"),
                release_date=entry.get("published"),
                duration=entry.get("itunes_duration"),
            )
            return episode
    raise ValueError(f"No episode found matching '{episode_query}' in podcast '{podcast_title}'")
