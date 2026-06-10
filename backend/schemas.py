"""Pydantic request models for the recommender API."""
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    # Either user_id (existing dataset user) OR ratings (new / synthetic user).
    user_id: int | None = Field(default=None, description="Existing dataset userId.")
    ratings: dict[int, float] | None = Field(
        default=None, description="movieId -> rating, for a cold-start / synthetic user.")
    model: str = "dual"
    k: int = Field(default=10, ge=1, le=50)
    genres: list[str] | None = None
    year_min: int | None = None
    year_max: int | None = None


class CompareRequest(BaseModel):
    user_id: int | None = None
    ratings: dict[int, float] | None = None
    models: list[str] = Field(default_factory=lambda: ["dual", "stacked", "content_genome"])
    k: int = Field(default=10, ge=1, le=50)
    genres: list[str] | None = None
    year_min: int | None = None
    year_max: int | None = None


class ExplainRequest(BaseModel):
    user_id: int | None = None
    ratings: dict[int, float] | None = None
    movie_id: int
    model: str = "dual"
