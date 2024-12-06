from pydantic import BaseModel


class PodcastInformation(BaseModel):
    pass


def search_podcasts(podcast_title: str, episode_query: str) -> PodcastInformation:
    """TODO: query public podcast index URL to get metadata for specific episode"""
    pass
