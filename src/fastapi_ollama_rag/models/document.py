from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    """
    Strict data contract for extracted document content.
    """

    text: str = Field(
        ..., description="The complete extracted raw text from the document."
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Dictionary containing document metadata "
        "(e.g., author, title, page_count).",
    )
