from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    source: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)


class DocumentReceipt(BaseModel):
    source: str
    chunks_stored: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    # top_k is the maximum number of semantically nearest chunks supplied to the LLM.
    top_k: int = Field(default=4, ge=1, le=20)


class SourceMatch(BaseModel):
    source: str
    chunk_index: int
    content: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceMatch]


class HealthResponse(BaseModel):
    status: str
    database: str
    ai_provider: str
    provider_configured: bool
