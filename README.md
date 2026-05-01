# AI Knowledge Base Assistant

Lokalny backend RAG w Pythonie. Aplikacja pozwala dodawać dokumenty wiedzy,
indeksować je przez embeddingi z Ollamy, zapisywać wektory w PostgreSQL/pgvector
i zadawać pytania, na które model odpowiada na podstawie znalezionych źródeł.

Projekt jest zbudowany jako portfolio backend/AI engineering: pokazuje podział
na warstwy API, services, repositories, schemas i models, lokalne modele przez
Ollamę, pgvector similarity search oraz testy bez zależności od zewnętrznego API.

## Funkcje

- `GET /api/v1/health` - healthcheck aplikacji.
- `POST /api/v1/documents` - dodanie dokumentu tekstowego lub Markdown.
- `GET /api/v1/documents` - lista dokumentów bez pełnej treści.
- `DELETE /api/v1/documents/{document_id}` - usunięcie dokumentu i jego chunków.
- `POST /api/v1/documents/{document_id}/index` - chunking, embeddingi i zapis chunków.
- `POST /api/v1/questions` - prosty pipeline RAG: retrieval, prompt, odpowiedź i źródła.
- `GET /` - prosty panel webowy do pracy z dokumentami i pytaniami.

## Stack

- Python 3.12+
- FastAPI
- Pydantic v2
- SQLAlchemy 2.x
- Alembic
- PostgreSQL
- pgvector
- Docker Compose
- Ollama
- httpx
- pytest
- ruff

## Architektura

```text
app/
  api/
    dependencies.py       # składanie zależności FastAPI
    routes/               # warstwa HTTP
  core/                   # konfiguracja i logging
  db/                     # SQLAlchemy engine, session, Base
  models/                 # modele bazy danych
  repositories/           # zapytania do bazy
  schemas/                # kontrakty API Pydantic
  services/               # logika aplikacyjna i AI
```

Najważniejszy przepływ:

```text
document -> chunks -> embeddings -> pgvector
question -> question embedding -> similar chunks -> prompt -> Ollama chat model -> answer + sources
```

Warstwy są rozdzielone celowo:

- routes obsługują HTTP i mapowanie błędów,
- services realizują logikę aplikacyjną,
- repositories rozmawiają z bazą,
- schemas definiują request/response,
- models mapują tabele PostgreSQL.

## Wymagania lokalne

- Python 3.12+
- Docker i Docker Compose
- Ollama działająca lokalnie na `http://localhost:11434`
- Model embeddingów:

```bash
ollama pull nomic-embed-text
```

- Model chat:

```bash
ollama pull llama3.1:8b
```

Domyślny embedding model `nomic-embed-text` zwraca wektory o wymiarze `768`.
Kolumna `document_chunks.embedding` ma typ `vector(768)`, więc zmiana modelu
embeddingów na inny wymiar wymaga migracji bazy.

## Uruchomienie lokalne w WSL

```bash
cd /mnt/d/nauka/project3
source .venv/bin/activate
cp .env.example .env
docker compose up -d db
python -m alembic upgrade head
python -m uvicorn app.main:app --reload
```

API będzie dostępne pod:

- `http://localhost:8000/`
- `http://localhost:8000/api/v1/health`
- `http://localhost:8000/docs`

PostgreSQL z Compose używa hostowego portu `5433`, żeby nie kolidować z innymi
lokalnymi projektami na `5432`.

## Uruchomienie przez Docker Compose

```bash
docker compose up --build
```

W kontenerze API łączy się z bazą przez `db:5432`. Ollama nadal działa na
hoście, więc Compose ustawia `OLLAMA_BASE_URL` na `http://host.docker.internal:11434`.

Migracje najprościej uruchamiać lokalnie:

```bash
source .venv/bin/activate
python -m alembic upgrade head
```

## Przykładowy scenariusz

Najwygodniej użyć panelu webowego:

```text
http://localhost:8000/
```

Panel pozwala dodać dokument, odświeżyć listę, uruchomić indeksację i zadać pytanie.
Podczas generowania odpowiedzi pokazuje prosty stan oczekiwania.

Ten sam przepływ można wykonać curlami:

### 1. Healthcheck

```bash
curl http://localhost:8000/api/v1/health
```

### 2. Dodanie dokumentu

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "RAG notes",
    "content": "RAG retrieves relevant context from a knowledge base and then uses a language model to answer based on that context.",
    "content_type": "text/plain",
    "metadata": {"source": "manual-test"}
  }'
```

Odpowiedź zawiera `id` dokumentu. Użyj go w następnym kroku.

### 3. Indeksacja dokumentu

```bash
curl -X POST http://localhost:8000/api/v1/documents/<document_id>/index
```

Ten krok:

- dzieli dokument na chunki,
- generuje embeddingi przez Ollamę,
- zapisuje chunki i embeddingi w PostgreSQL/pgvector,
- ustawia status dokumentu na `indexed`.

### 4. Zadanie pytania

```bash
curl -X POST http://localhost:8000/api/v1/questions \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does RAG do?",
    "top_k": 5
  }'
```

Przykładowy kształt odpowiedzi:

```json
{
  "question": "What does RAG do?",
  "answer": "RAG retrieves relevant context from a knowledge base and uses it to answer the question.",
  "chat_model": "llama3.1:8b",
  "embedding_model": "nomic-embed-text",
  "sources": [
    {
      "document_id": "11111111-1111-1111-1111-111111111111",
      "chunk_id": "22222222-2222-2222-2222-222222222222",
      "chunk_index": 0,
      "content": "RAG retrieves relevant context from a knowledge base...",
      "distance": 0.12,
      "similarity": 0.88
    }
  ],
  "source_count": 1
}
```

`distance` to dystans kosinusowy z pgvector: im niższy, tym lepiej.
`similarity` jest liczone jako `1.0 - distance`: im wyższe, tym lepiej.

## Testy i linting

```bash
cd /mnt/d/nauka/project3
source .venv/bin/activate
python -m pytest -q
python -m ruff check .
```

Aktualnie testy obejmują:

- healthcheck,
- walidację dokumentów,
- endpointy dokumentów i pytań,
- modele bazy,
- chunking,
- embedding service,
- retrieval service,
- RAG service,
- klienta Ollamy,
- składanie zależności FastAPI.

Testy jednostkowe mockują Ollamę i serwisy tam, gdzie nie chcemy zależeć od
lokalnego modelu ani sieci.

## Konfiguracja

Najważniejsze zmienne z `.env.example`:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/ai_kb
EMBEDDING_DIMENSION=768
CHUNK_SIZE_CHARS=1000
CHUNK_OVERLAP_CHARS=150
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

## Decyzje techniczne

- SQLAlchemy + Pydantic zamiast SQLModel: wyraźny podział modeli bazy i kontraktów API.
- Synchroniczne SQLAlchemy: prostsze do nauki, debugowania i uruchomienia lokalnego.
- Ollama zamiast płatnego API: brak kosztów i lokalne działanie.
- pgvector w PostgreSQL: similarity search bez osobnego vector database.
- Prosty chunking po słowach: wystarczający do pierwszej wersji i łatwy do omówienia.
- Synchroniczna indeksacja: czytelny pipeline bez kolejek i background jobs.

## Trade-offs

- Indeksacja czeka na embeddingi w ramach requestu, więc duże dokumenty mogą trwać długo.
- Chunking jest prosty i nie rozumie struktury Markdown, nagłówków ani tabel.
- Brak rerankingu wyników retrieval.
- Brak streamingu odpowiedzi z modelu.
- Brak autoryzacji i multi-tenancy, bo projekt skupia się na lokalnym backendzie RAG.
- Testy są głównie jednostkowe i endpointowe; cięższe testy integracyjne z prawdziwą bazą
  można dodać później.

## Future improvements

- Lepszy chunking dla Markdown i dokumentów technicznych.
- Reranking źródeł przed generowaniem odpowiedzi.
- Background jobs dla indeksacji dużych dokumentów.
- Streaming odpowiedzi z Ollamy.
- Testy integracyjne z PostgreSQL/pgvector.
- Prosty frontend do dodawania dokumentów i zadawania pytań.
- Obsługa plików PDF, HTML lub wielu formatów wejściowych.

## Opis do CV / portfolio

AI Knowledge Base Assistant to lokalny backend RAG zbudowany w Pythonie,
FastAPI, PostgreSQL/pgvector i Ollama. Projekt implementuje pełny przepływ:
dodawanie dokumentów, chunking, generowanie embeddingów, similarity search,
budowę promptu i generowanie odpowiedzi ze źródłami. Kod jest podzielony na
warstwy API, services, repositories, schemas i models oraz pokryty testami
jednostkowymi i endpointowymi.

## Status projektu

Zrealizowane etapy:

- architektura projektu,
- szkielet FastAPI,
- PostgreSQL + pgvector + Alembic,
- dodawanie i listowanie dokumentów,
- chunking i indeksacja przez Ollamę,
- retrieval,
- generowanie odpowiedzi RAG,
- uporządkowanie struktury aplikacji,
- testy,
- dokumentacja portfolio.
