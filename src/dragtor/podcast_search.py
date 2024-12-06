import requests
from typing import Optional, List
from pydantic import BaseModel, Field


class PodcastInformation(BaseModel):
    """Information about a podcast episode from iTunes"""
    podcast_name: str = Field(..., description="Name of the podcast")
    episode_title: Optional[str] = Field(None, description="Title of the specific episode")
    description: Optional[str] = Field(None, description="Episode or podcast description")
    artwork_url: Optional[str] = Field(None, description="URL to podcast artwork")
    release_date: Optional[str] = Field(None, description="Release date of episode")
    duration: Optional[str] = Field(None, description="Duration of episode")
    feed_url: Optional[str] = Field(None, description="RSS feed URL")


def search_podcasts(podcast_title: str, episode_query: str) -> PodcastInformation:
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
    podcast_params = {
        "term": podcast_title,
        "entity": "podcast",
        "limit": 1
    }
    
    response = requests.get(base_url, params=podcast_params)
    response.raise_for_status()
    
    results = response.json()
    if not results.get("results"):
        raise ValueError(f"No podcast found matching '{podcast_title}'")
        
    podcast = results["results"][0]
    
    # Then search for specific episode
    episode_params = {
        "term": episode_query,
        "collectionId": podcast["collectionId"],
        "entity": "podcastEpisode",
        "limit": 1
    }
    
    response = requests.get(base_url, params=episode_params)
    response.raise_for_status()
    
    results = response.json()
    episode = results.get("results", [{}])[0]
    
    return PodcastInformation(
        podcast_name=podcast.get("collectionName", ""),
        episode_title=episode.get("trackName"),
        description=episode.get("description", podcast.get("description")),
        artwork_url=podcast.get("artworkUrl600"),
        release_date=episode.get("releaseDate"),
        duration=episode.get("trackTimeMillis"),
        feed_url=podcast.get("feedUrl")
    )
