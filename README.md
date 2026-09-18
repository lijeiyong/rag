# RAG with FastAPI, Gemini, PostgreSQL, and pgvector

This repository is a small, intentionally explicit RAG (retrieval-augmented generation) project. It stores document chunks and their Gemini embeddings in PostgreSQL, retrieves the chunks closest to a question with pgvector, and gives only those chunks to a text-generation model. Gemini is the default, while the OpenAI provider remains available behind one configuration switch.

## The RAG flow

```text
Ingestion: document -> overlapping chunks -> embeddings -> PostgreSQL/pgvector

Question: question -> embedding -> cosine search -> relevant chunks -> LLM -> answer
```

RAG does not retrain the language model. It adds relevant, application-owned context to each question at request time. The returned `sources` make it possible to see which stored chunks supported an answer.

## Run it in a VS Code dev container

Prerequisites: Docker Desktop, VS Code, and the Dev Containers extension.

1. Copy `.env.example` to `.env` and add a `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey).
2. Open this repository in VS Code.
3. Run **Dev Containers: Reopen in Container** from the command palette.
4. In the container terminal, start the API:

   ```bash
   uvicorn rag_app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. Open <http://localhost:8000/docs> for the interactive API documentation.

The dev container starts two services from `compose.yaml`:

- `app`: Python 3.12, the project dependencies, and VS Code tooling.
- `db`: PostgreSQL 16 with the pgvector extension already installed.

The named Docker volume preserves database data across container rebuilds. Run `docker compose down -v` from the host only when you intentionally want to delete that data.

## Try the API

Store a document (re-ingesting the same `source` replaces its previous chunks):

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"source":"handbook","content":"RAG retrieves relevant passages before asking a language model to answer. pgvector adds vector similarity search to PostgreSQL."}'
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What does pgvector do?","top_k":4}'
```

`top_k` is the maximum number of nearest document chunks retrieved from PostgreSQL and placed in the model's prompt. The default is 4 (allowed range: 1-20). A larger value can recover more supporting context, but it also adds prompt tokens, latency, and possibly irrelevant text.

Useful endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Check the API, database, and API-key configuration |
| `POST` | `/documents` | Chunk, embed, and store a document |
| `POST` | `/ask` | Retrieve related chunks and answer from them |

## How PostgreSQL becomes a vector database

The `db` container is ordinary PostgreSQL plus the pgvector extension. [`sql/001_init.sql`](sql/001_init.sql) creates a `vector(1536)` column: every row stores one text chunk and its 1,536-number embedding. Texts with similar meanings have nearby embeddings.

The query in [`src/rag_app/repository.py`](src/rag_app/repository.py) orders rows with pgvector's cosine-distance operator, `<=>`. Smaller distance means more similar; the API exposes `1 - distance` as an intuitive similarity score. An HNSW index accelerates that nearest-neighbor lookup as the table grows.

The vector size and embedding model are coupled. This project requests 1,536 dimensions from `gemini-embedding-2`, matching the SQL column. If you change the embedding model or dimensions, also recreate or migrate the vector column.

Embeddings produced by different providers or models do not share a meaningful coordinate space. When switching between Gemini and OpenAI, clear the old database vectors and re-ingest the documents:

```bash
docker compose down -v
docker compose up -d
```

The first command deletes the local PostgreSQL volume, so only run it when discarding the indexed data is intentional.

## Project layout

```text
.devcontainer/       Reproducible VS Code container definition
sql/                 PostgreSQL extension, table, and vector index
src/rag_app/
  chunking.py        Turns a document into overlapping retrieval units
  config.py          Environment-based settings
  database.py        Async PostgreSQL connection pool
  main.py            FastAPI routes and application lifecycle
  providers.py       Switchable Gemini/OpenAI embeddings and generation
  rag.py             Ingestion and retrieve-then-generate orchestration
  repository.py      pgvector persistence and similarity search
tests/               Fast unit tests with fake providers (no API key needed)
```

## Development commands

Run these inside the dev container:

```bash
pytest
ruff check .
ruff format --check .
mypy src
```

## Configuration and `.env`

The values in `config.py` are defaults, not hardcoded overrides. Calling `Settings()` makes Pydantic Settings load `.env`; real process environment variables take priority over that file. Docker Compose also reads the repository's `.env` for `${VARIABLE}` substitutions and injects the resulting values into the `app` container. `main.py` calls `get_settings()`, then passes those resolved values into the selected provider and database.

This means the normal dev-container path is:

```text
.env -> Docker Compose substitution -> app environment -> Settings -> provider/database
```

Available variables:

| Variable | Default | Meaning |
|---|---|---|
| `AI_PROVIDER` | `gemini` | Use `gemini` or `openai` |
| `GEMINI_API_KEY` | empty | Required when Gemini is selected |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Creates searchable vectors |
| `GEMINI_ANSWER_MODEL` | `gemini-3.8-flash` | Writes an answer from retrieved text |
| `OPENAI_API_KEY` | empty | Required only when `AI_PROVIDER=openai` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Creates searchable vectors |
| `OPENAI_ANSWER_MODEL` | `gpt-5-mini` | Writes an answer from retrieved text |
| `EMBEDDING_DIMENSIONS` | `1536` | Must match the SQL vector column |
| `DATABASE_URL` | `postgresql://rag:rag@db:5432/rag` | PostgreSQL connection string |
| `POSTGRES_PORT` | `5433` | Optional host port for database tools |

Do not commit `.env`; it is ignored by Git.

## Tokens and cost

Embedding is not token-free: the embedding API tokenizes the document chunks during ingestion and the question during retrieval. Answer generation then consumes input tokens (instructions, question, and retrieved chunks) and output tokens; some models may also account for thinking/reasoning tokens.

Google currently lists text input for Gemini Embedding as free on its free tier, within rate limits, and separately prices paid-tier usage. So embeddings still consume tokens/quota even when their price is $0. Check the current [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) and [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) before relying on the free tier. Google also states that free-tier content may be used to improve its products, so do not upload sensitive documents without reviewing the data-use terms.
