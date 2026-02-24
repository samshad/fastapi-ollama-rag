from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """
    Represents a chunk of text retrieved from the vector database,
    including its relevance score.
    """

    text: str
    metadata: dict
    similarity_score: float = Field(
        ..., description="Cosine similarity score (1.0 is exact match)"
    )
