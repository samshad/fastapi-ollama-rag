from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """
    Strict data contract representing a localized segment of a larger document.
    """

    text: str = Field(..., description="The text content of the chunk.")
    chunk_index: int = Field(
        ..., description="The sequential order of this chunk in the document."
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Metadata inherited from the parent document "
        "(e.g., source file, page_count).",
    )
