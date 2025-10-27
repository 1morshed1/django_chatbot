# Django RAG Chatbot 🤖

A production-ready AI chatbot with document-aware responses using Retrieval-Augmented Generation (RAG). Built with Django, PostgreSQL (pgvector), and Groq API for real-time streaming conversations.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Django](https://img.shields.io/badge/Django-4.2-green.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **📄 Document Upload & Processing** - Upload .txt files for semantic knowledge retrieval
- **💬 Conversational AI** - Chat with AI assistant about uploaded documents
- **⚡ Real-Time Streaming** - Token-by-token streaming responses via Server-Sent Events (SSE)
- **🔍 Semantic Search** - pgvector-powered similarity search with cosine distance
- **🧠 Memory Management** - Intelligent context window management with token budgets
- **🔄 LangGraph Orchestration** - Clean workflow management for RAG pipeline
- **📊 Chat History** - Persistent conversation sessions with message tracking
- **🎯 High Performance** - Batch embedding generation, connection pooling, optimized queries

## 📸 Demo

| Document Upload | Streaming Response |
|:---------------:|:------------------:|
| ![Document Upload](screenshot/Screenshot%20from%202025-10-26%2011-13-56.png) | ![Streaming Response](screenshot/Screenshot%20from%202025-10-26%2011-15-02.png) |

| Document Processing | Chat History | API Endpoints |
|:-------------------:|:------------:|:-------------:|
| ![Document Processing](screenshot/Screenshot%20from%202025-10-26%2011-15-35.png) | ![Chat History](screenshot/Screenshot%20from%202025-10-26%2011-17-46.png) | ![API Endpoints](screenshot/Screenshot%20from%202025-10-26%2011-18-26.png) |

| Health Check | Session Management | Document Context |
|:------------:|:------------------:|:----------------:|
| ![Health Check](screenshot/Screenshot%20from%202025-10-26%2011-28-49.png) | ![Session Management](screenshot/Screenshot%20from%202025-10-26%2011-29-00.png) | ![Document Context](screenshot/Screenshot%20from%202025-10-26%2011-29-02.png) |

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │ (Browser/API Client)
└──────┬──────┘
       │ HTTP/SSE
       ▼
┌─────────────────────────────────────────┐
│     Django REST Framework (API)         │
├─────────────────────────────────────────┤
│  • /api/document/        (Upload)       │
│  • /api/chat/           (Streaming)     │
│  • /api/session/<id>/   (History)       │
│  • /api/health/         (Status)        │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         LangGraph Orchestration         │
│  (Retrieval → Context → Generation)     │
└──────┬──────────────────────────────────┘
       │
       ├─────────────┬─────────────┐
       ▼             ▼             ▼
  ┌────────┐   ┌──────────┐  ┌──────────┐
  │ Groq   │   │PostgreSQL│  │Embedding │
  │ LLM    │   │ pgvector │  │ Service  │
  └────────┘   └──────────┘  └──────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | Django 4.2 + DRF | REST API & business logic |
| **Database** | PostgreSQL 14+ | Relational data & vector storage |
| **Vector Store** | pgvector | Semantic similarity search |
| **LLM** | Groq API (Llama 3.3-70B) | Natural language generation |
| **Embeddings** | SentenceTransformers | Document & query embeddings |
| **Orchestration** | LangGraph | RAG workflow management |
| **Tokenizer** | GPT-2 (HuggingFace) | Token counting & chunking |

## 📋 Prerequisites

- **Python** 3.10 or higher
- **PostgreSQL** 14 or higher
- **pgvector** extension for PostgreSQL
- **Groq API Key** ([Get one here](https://console.groq.com))
- **Git** for version control

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd django_chatbot
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up PostgreSQL Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE ragdb;

# Connect to the database
\c ragdb

# Enable pgvector extension
CREATE EXTENSION vector;

# Exit psql
\q
```

### 5. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor
```

**Required environment variables:**

```env
# Django Settings
SECRET_KEY=your-secret-key-here-generate-a-long-random-string
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/ragdb

# Groq API
GROQ_API_KEY=your-groq-api-key-here
```

### 6. Run Database Migrations

```bash
python manage.py migrate
```

### 7. Create Superuser (Optional)

```bash
python manage.py createsuperuser
```

### 8. Start Development Server

```bash
python manage.py runserver 8001
```

The server will start at `http://localhost:8001/`

### 9. Verify Installation

```bash
# Health check
curl http://localhost:8001/api/health/
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "embedding_service": "loaded",
  "embedding_dim": 384,
  "groq_api_configured": true
}
```

## 📖 API Documentation

### Base URL

```
http://localhost:8001/api/
```

### Endpoints

#### 1. Create Chat Session

Create a new conversation session.

**Endpoint:** `POST /api/session/create/`

**Request:**
```bash
curl -X POST http://localhost:8001/api/session/create/
```

**Response:**
```json
{
  "id": 1,
  "created_at": "2025-10-26T10:00:00Z",
  "updated_at": "2025-10-26T10:00:00Z",
  "messages": []
}
```

---

#### 2. Upload Document

Upload a .txt document for knowledge retrieval.

**Endpoint:** `POST /api/document/`

**Request:**
```bash
curl -X POST http://localhost:8001/api/document/ \
  -F "file=@document.txt"
```

**Response:**
```json
{
  "document_id": 1,
  "filename": "document.txt",
  "status": "processed",
  "total_chunks": 25,
  "processing_time": "2.34s"
}
```

**Constraints:**
- File format: `.txt` only
- Max file size: 5MB
- Encoding: UTF-8

---

#### 3. Chat with Streaming

Send a message and receive streaming responses.

**Endpoint:** `POST /api/chat/`

**Request:**
```bash
curl -N -X POST http://localhost:8001/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": 1,
    "message": "What are the key points in the document?"
  }'
```

**Response (Server-Sent Events):**
```
data: {"type":"start"}

data: {"type":"content","content":"The"}

data: {"type":"content","content":" document"}

data: {"type":"content","content":" discusses"}

...

data: {"type":"done","message_id":123}
```

**Parameters:**
- `session_id` (required): ID of the chat session
- `message` (required): User's message/question

---

#### 4. Get Chat History

Retrieve conversation history for a session.

**Endpoint:** `GET /api/session/<session_id>/messages/`

**Request:**
```bash
curl http://localhost:8001/api/session/1/messages/
```

**Response:**
```json
{
  "id": 1,
  "created_at": "2025-10-26T10:00:00Z",
  "updated_at": "2025-10-26T10:05:00Z",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "What is in the document?",
      "created_at": "2025-10-26T10:01:00Z"
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "The document discusses...",
      "created_at": "2025-10-26T10:01:05Z"
    }
  ]
}
```

---

#### 5. Health Check

Check system health status.

**Endpoint:** `GET /api/health/`

**Request:**
```bash
curl http://localhost:8001/api/health/
```

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "embedding_service": "loaded",
  "embedding_dim": 384,
  "groq_api_configured": true
}
```

## 🔧 Configuration

### Database Settings

Edit `config/settings.py` for database configuration:

```python
DATABASES = {
    'default': {
        **env.db(),
        'CONN_MAX_AGE': 600,  # Connection pooling (10 minutes)
    }
}
```

### Context Window Management

Edit `rag/services/context.py` to adjust token budgets:

```python
class ContextManager:
    MAX_CONTEXT_TOKENS = 128000      # Llama 3.3 context window
    SYSTEM_PROMPT_TOKENS = 500       # System instructions
    DOCUMENT_CONTEXT_TOKENS = 5120   # Retrieved document chunks
    CURRENT_MESSAGE_TOKENS = 500     # User query
    RESERVED_FOR_RESPONSE = 8000     # LLM output space
    SAFETY_BUFFER = 2000             # Prevent truncation
```

### Document Chunking

Edit `rag/services/chunking.py` to adjust chunking strategy:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,           # Tokens per chunk
    chunk_overlap=50,         # Overlap to preserve context
    length_function=count_tokens,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
)
```

### Retrieval Settings

Edit retrieval parameters in `rag/services/retrieval.py`:

```python
def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,              # Number of chunks to retrieve
    min_similarity: float = 0.5  # Minimum similarity threshold (0-1)
):
    # ...
```

## 📁 Project Structure

```
django_chatbot/
├── config/                      # Django project settings
│   ├── __init__.py
│   ├── settings.py             # Main settings file
│   ├── urls.py                 # Root URL configuration
│   ├── wsgi.py                 # WSGI configuration
│   └── asgi.py                 # ASGI configuration
│
├── rag/                        # Main application
│   ├── migrations/             # Database migrations
│   │   └── 0001_initial.py
│   ├── services/               # Business logic layer
│   │   ├── __init__.py
│   │   ├── agent.py           # LangGraph orchestration
│   │   ├── chunking.py        # Document chunking
│   │   ├── context.py         # Context management
│   │   ├── embeddings.py      # Embedding service
│   │   └── retrieval.py       # Vector search
│   ├── __init__.py
│   ├── admin.py               # Django admin configuration
│   ├── apps.py                # App configuration
│   ├── models.py              # Database models
│   ├── serializers.py         # DRF serializers
│   ├── urls.py                # App URL routes
│   └── views.py               # API endpoints
│
├── .env                        # Environment variables (not in git)
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── manage.py                  # Django management script
├── README.md                  # This file
├── RESEARCH_AND_ANALYSIS.md   # System design document
└── requirements.txt           # Python dependencies
```

## 🧪 Testing

### Manual Testing

```bash
# 1. Create a test document
echo "This is a test document about artificial intelligence." > test.txt

# 2. Create session
SESSION_ID=$(curl -s -X POST http://localhost:8001/api/session/create/ | jq -r '.id')

# 3. Upload document
curl -X POST http://localhost:8001/api/document/ -F "file=@test.txt"

# 4. Chat (streaming)
curl -N -X POST http://localhost:8001/api/chat/ \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": $SESSION_ID, \"message\": \"What is in the document?\"}"

# 5. Get history
curl http://localhost:8001/api/session/$SESSION_ID/messages/
```

### Using Python

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8001/api"

# Create session
response = requests.post(f"{BASE_URL}/session/create/")
session_id = response.json()['id']
print(f"Session ID: {session_id}")

# Upload document
with open('test.txt', 'rb') as f:
    files = {'file': f}
    response = requests.post(f"{BASE_URL}/document/", files=files)
    print(response.json())

# Chat with streaming
data = {
    "session_id": session_id,
    "message": "What is in the document?"
}

response = requests.post(
    f"{BASE_URL}/chat/",
    json=data,
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            if data['type'] == 'content':
                print(data['content'], end='', flush=True)

print("\n")
```

## 🚀 Deployment

### Production Checklist

- [ ] Set `DEBUG=False` in .env
- [ ] Use strong `SECRET_KEY`
- [ ] Configure `ALLOWED_HOSTS`
- [ ] Set up HTTPS/SSL
- [ ] Use production database credentials
- [ ] Enable connection pooling
- [ ] Set up logging
- [ ] Configure static files
- [ ] Add rate limiting
- [ ] Set up monitoring
- [ ] Configure backups

### Using Gunicorn

1. Install Gunicorn:
```bash
pip install gunicorn
```

2. Create `gunicorn.conf.py`:
```python
bind = "0.0.0.0:8000"
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
```

3. Run with Gunicorn:
```bash
gunicorn config.wsgi:application -c gunicorn.conf.py
```

### Environment Variables for Production

```env
SECRET_KEY=<generate-strong-random-key>
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

DATABASE_URL=postgresql://user:password@db-host:5432/ragdb

GROQ_API_KEY=<your-production-api-key>

```

## 🔍 Troubleshooting

### Issue: Models not loading

**Problem:** Embedding model or tokenizer fails to load

**Solution:**
```bash
# Check internet connection
ping huggingface.co

# Manually download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Or skip model loading during migrations
export SKIP_MODEL_LOADING=True
python manage.py migrate
unset SKIP_MODEL_LOADING
```

### Issue: pgvector extension not found

**Problem:** `ERROR: type "vector" does not exist`

**Solution:**
```bash
# Install pgvector
# On Ubuntu/Debian:
sudo apt-get install postgresql-14-pgvector

# On macOS:
brew install pgvector

# Then enable in database:
psql -U postgres -d ragdb -c "CREATE EXTENSION vector;"
```

### Issue: Slow vector search

**Problem:** Queries take >1 second with many documents

**Solution:**
```sql
-- Create vector index (after loading documents)
CREATE INDEX idx_chunks_embedding ON document_chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### Issue: Connection header error

**Problem:** `AssertionError: Hop-by-hop header, 'Connection: keep-alive', not allowed`

**Solution:** Remove `Connection` header from streaming response (this is handled in the code).

### Issue: Streaming not working

**Problem:** Full response appears at once, not token-by-token

**Solution:** Ensure you're using `run_agent_streaming()` in `views.py`, not `run_agent()`.

## 📊 Performance Tips

### Database Optimization

```python
# Use select_related for foreign keys
messages = ChatMessage.objects.select_related('session').all()

# Use prefetch_related for reverse relations
sessions = ChatSession.objects.prefetch_related('messages').all()

# Bulk operations
DocumentChunk.objects.bulk_create(chunks, batch_size=100)
```

### Caching

```python
# Add to settings.py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}

# Use in views
from django.core.cache import cache

def retrieve_with_cache(query):
    cache_key = f"query:{hash(query)}"
    result = cache.get(cache_key)
    if result is None:
        result = retrieve_relevant_chunks(query)
        cache.set(cache_key, result, timeout=300)
    return result
```

### Connection Pooling

Already configured in `settings.py`:
```python
DATABASES = {
    'default': {
        'CONN_MAX_AGE': 600,  # Keep connections for 10 minutes
    }
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Code formatting
black .

# Linting
flake8
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Django** - Web framework
- **PostgreSQL & pgvector** - Database and vector storage
- **Groq** - LLM API provider
- **Sentence Transformers** - Embedding models
- **LangChain/LangGraph** - RAG orchestration

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ for Qtec Solution Limited**

**Version:** 1.0.0  
**Last Updated:** October 26, 2025
