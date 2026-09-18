from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from rag_app.config import get_settings
from rag_app.database import Database
from rag_app.models import (
    AskRequest,
    AskResponse,
    DocumentIn,
    DocumentReceipt,
    HealthResponse,
    SourceMatch,
)
from rag_app.providers import (
    AIProvider,
    ProviderNotConfiguredError,
    ProviderRequestError,
    create_provider,
)
from rag_app.rag import NoDocumentsError, RagService
from rag_app.repository import DocumentRepository


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    database = Database(settings.database_url)
    await database.open()

    provider = create_provider(settings)
    app.state.database = database
    app.state.provider = provider
    app.state.provider_name = settings.ai_provider
    app.state.rag = RagService(
        provider,
        DocumentRepository(database),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    try:
        yield
    finally:
        await database.close()


app = FastAPI(title="Simple PostgreSQL RAG", version="0.1.0", lifespan=lifespan)


def get_rag_service(request: Request) -> RagService:
    return cast(RagService, request.app.state.rag)


RagDependency = Annotated[RagService, Depends(get_rag_service)]


@app.exception_handler(ProviderNotConfiguredError)
async def provider_not_configured(
    _request: Request, error: ProviderNotConfiguredError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(error)},
    )


@app.exception_handler(ProviderRequestError)
async def provider_request_failed(_request: Request, error: ProviderRequestError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(error)},
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    database = cast(Database, request.app.state.database)
    provider = cast(AIProvider, request.app.state.provider)
    database_ok = await database.is_healthy()
    return HealthResponse(
        status="ok" if database_ok else "degraded",
        database="connected" if database_ok else "unavailable",
        ai_provider=cast(str, request.app.state.provider_name),
        provider_configured=provider.configured,
    )


@app.post("/documents", response_model=DocumentReceipt, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    document: DocumentIn,
    rag: RagDependency,
) -> DocumentReceipt:
    try:
        count = await rag.ingest(document.source, document.content)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    return DocumentReceipt(source=document.source, chunks_stored=count)


@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    rag: RagDependency,
) -> AskResponse:
    try:
        answer, matches = await rag.ask(request.question, request.top_k)
    except NoDocumentsError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    return AskResponse(
        answer=answer,
        sources=[
            SourceMatch(
                source=match.source,
                chunk_index=match.chunk_index,
                content=match.content,
                similarity=match.similarity,
            )
            for match in matches
        ],
    )
