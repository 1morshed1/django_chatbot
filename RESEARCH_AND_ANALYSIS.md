# RESEARCH_AND_ANALYSIS_MVP.md (Simplified for Quick Launch)

## 1. System Architecture Overview

### Simplified Architecture (No Redis/Celery)

```
┌─────────────┐
│   Client    │ (Streamlit/React Frontend)
│  (Browser)  │
└──────┬──────┘
       │ HTTP/SSE
       ▼
┌─────────────────────────────────────────────────────┐
│           Django REST Framework (API Layer)          │
├─────────────────────────────────────────────────────┤
│  • /api/chat/          (POST - Stream responses)    │
│  • /api/document/      (POST - Upload & process)    │
│  • /api/session/<id>/  (GET - Retrieve history)     │
│  • /api/health/        (GET - Health check)         │
└──────┬──────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Orchestration                 │
├─────────────────────────────────────────────────────┤
│  Node 1: Query Processing & Embedding               │
│  Node 2: Semantic Search (Top-K cosine)             │
│  Node 3: Context Builder (History + Documents)      │
│  Node 4: LLM Invocation (Groq - Streaming)          │
│  Node 5: Response Saver (Save to DB)                │
└──────┬──────────────────────────────────────────────┘
       │
       ├─────────────────┬─────────────────┐
       ▼                 ▼                 ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│   Groq API  │  │  PostgreSQL  │  │   Django     │
│   (LLM)     │  │  (pgvector)  │  │   Cache      │
└─────────────┘  └──────────────┘  └──────────────┘
                        │
                        ├── ChatSession
                        ├── ChatMessage
                        ├── Document
                        └── DocumentChunk (vectors)
```

### Request Flow

**Chat Flow (Synchronous):**
1. User sends message via POST `/api/chat/`
2. Django validates session and saves user message
3. Check Django cache for similar queries (in-memory)
4. LangGraph orchestration:
   - Generate query embedding
   - Search pgvector for relevant chunks (Top-5)
   - Build context with history (token-limited)
   - Stream from Groq API via SSE
   - Save assistant response to database
5. Client receives streamed response
6. Cache result in Django cache

**Document Upload Flow (Synchronous - Immediate Processing):**
1. User uploads file via POST `/api/document/`
2. Django validates file (type, size <5MB)
3. Save Document record with `status='processing'`
4. **Immediately process** (2-5 seconds for typical docs):
   - Read file content
   - Chunk document (semantic boundaries, 512 tokens)
   - Generate embeddings batch
   - Save chunks to database
   - Update status to `processed`
5. Return success response with document details

**Trade-off:** User waits 2-5 seconds during upload, but simpler architecture and immediate availability.

---

## 2. Technology Stack (Simplified)

### Core Technologies

| Component | Technology | Why? |
|-----------|-----------|------|
| **Backend** | Django 4.2 + DRF | Mature, great ORM, built-in admin |
| **Database** | PostgreSQL + pgvector | Single DB for relations + vectors |
| **LLM** | Groq (Llama 3.1) | Fast, affordable, 128K context |
| **Embeddings** | sentence-transformers | Free, local, fast |
| **Orchestration** | LangGraph | Clean workflow management |
| **Cache** | Django in-memory | Built-in, zero config |

### What We're NOT Using (Yet)

❌ **Redis** - Django's in-memory cache is sufficient for <100 users  
❌ **Celery** - Synchronous processing fast enough for <10MB files  
❌ **Docker** - Can deploy directly or add later  
❌ **Reranking** - Start with basic cosine similarity  

### When to Add Them Later

```python
# Add Redis when:
- >100 concurrent users
- Cache needs to persist across server restarts
- Need distributed caching

# Add Celery when:
- Document processing takes >10 seconds
- Need background jobs
- Handling concurrent uploads

# Add Reranking when:
- Retrieval quality needs improvement
- Users report irrelevant results
```

---

## 3. Document Processing (Simplified)

### Single Tokenizer Approach

```python
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load once at startup (in Django AppConfig)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def count_tokens(text: str) -> int:
    """Count tokens using LLM tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))

# Semantic chunking with token-based sizing
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=count_tokens,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
)

def chunk_document(text: str) -> list[dict]:
    """
    Chunk document and generate embeddings.
    
    Returns:
        List of dicts with 'text', 'token_count', 'embedding'
    """
    chunks = splitter.split_text(text)
    
    # Batch generate embeddings (faster)
    embeddings = embedding_model.encode(
        chunks, 
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False
    )
    
    return [
        {
            'text': chunk,
            'token_count': count_tokens(chunk),
            'embedding': embedding.tolist(),
            'chunk_index': i
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
```

### Database Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_size INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    uploaded_at TIMESTAMP DEFAULT NOW(),
    total_chunks INTEGER DEFAULT 0,
    CONSTRAINT valid_status CHECK (status IN ('pending', 'processing', 'processed', 'failed'))
);

-- Document chunks with vectors
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(384),
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_chunk UNIQUE (document_id, chunk_index)
);

-- Indexes
CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_documents_status ON documents(status);

-- Vector index (create after inserting data)
CREATE INDEX idx_chunks_embedding ON document_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Synchronous Upload View

```python
# views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
import time
import logging

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

@api_view(['POST'])
def upload_document(request):
    """
    Upload and immediately process document.
    Processing takes 2-5 seconds for typical documents.
    """
    if 'file' not in request.FILES:
        return Response(
            {"error": "No file provided"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    uploaded_file = request.FILES['file']
    
    # Validate size
    if uploaded_file.size > MAX_FILE_SIZE:
        return Response(
            {"error": f"File too large. Maximum {MAX_FILE_SIZE // (1024*1024)}MB"},
            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        )
    
    # Validate type (simple check)
    if not uploaded_file.name.endswith('.txt'):
        return Response(
            {"error": "Only .txt files supported"},
            status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        )
    
    # Create document record
    document = Document.objects.create(
        filename=uploaded_file.name,
        file_size=uploaded_file.size,
        status='processing'
    )
    
    try:
        start_time = time.time()
        
        # Read content
        content = uploaded_file.read().decode('utf-8')
        logger.info(f"Read file {document.id}: {len(content)} characters")
        
        # Chunk and embed (takes 2-5 seconds)
        chunks = chunk_document(content)
        logger.info(f"Created {len(chunks)} chunks for document {document.id}")
        
        # Bulk create chunks
        chunk_objects = [
            DocumentChunk(
                document=document,
                content=chunk['text'],
                embedding=chunk['embedding'],
                chunk_index=chunk['chunk_index'],
                token_count=chunk['token_count']
            )
            for chunk in chunks
        ]
        DocumentChunk.objects.bulk_create(chunk_objects, batch_size=100)
        
        # Update document status
        document.status = 'processed'
        document.total_chunks = len(chunks)
        document.save()
        
        processing_time = time.time() - start_time
        logger.info(f"Processed document {document.id} in {processing_time:.2f}s")
        
        return Response({
            "document_id": document.id,
            "filename": document.filename,
            "status": "processed",
            "total_chunks": len(chunks),
            "processing_time": f"{processing_time:.2f}s"
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        logger.exception(f"Failed to process document {document.id}")
        
        document.status = 'failed'
        document.error_message = str(e)
        document.save()
        
        return Response(
            {"error": f"Processing failed: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
```

---

## 4. Retrieval Strategy (Simplified)

### Basic Top-K Cosine Similarity

**Start with simple retrieval - add reranking later if needed:**

```python
from django.db import connection
import numpy as np

class EmbeddingService:
    """Singleton for embedding model."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            from sentence_transformers import SentenceTransformer
            cls._instance.model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2'
            )
        return cls._instance
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate normalized embedding."""
        return self.model.encode(text, normalize_embeddings=True)

# Global instance
embedding_service = EmbeddingService()


def retrieve_relevant_chunks(query: str, top_k: int = 5, min_similarity: float = 0.5) -> list[dict]:
    """
    Retrieve most relevant chunks using cosine similarity.
    
    Args:
        query: User's question
        top_k: Number of chunks to return
        min_similarity: Minimum similarity threshold (0-1)
    
    Returns:
        List of dicts with chunk content and metadata
    """
    # Generate query embedding
    query_embedding = embedding_service.generate_embedding(query)
    
    # Search using pgvector
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT 
                dc.id,
                dc.content,
                dc.token_count,
                dc.chunk_index,
                d.filename,
                d.id as document_id,
                1 - (dc.embedding <=> %s::vector) as similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.status = 'processed'
                AND 1 - (dc.embedding <=> %s::vector) >= %s
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s
        """, [
            query_embedding.tolist(),
            query_embedding.tolist(),
            min_similarity,
            query_embedding.tolist(),
            top_k
        ])
        
        results = [
            {
                'id': row[0],
                'content': row[1],
                'token_count': row[2],
                'chunk_index': row[3],
                'filename': row[4],
                'document_id': row[5],
                'similarity': float(row[6])
            }
            for row in cursor.fetchall()
        ]
    
    return results


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks for LLM prompt."""
    if not chunks:
        return "No relevant context found in uploaded documents."
    
    context_parts = ["=== RELEVANT DOCUMENT CONTEXT ===\n"]
    
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"""
[Source {i}: {chunk['filename']}, Chunk {chunk['chunk_index']}, Relevance: {chunk['similarity']:.2f}]
{chunk['content']}
---""")
    
    context_parts.append("\n=== END CONTEXT ===")
    return "\n".join(context_parts)
```

---

## 5. Chat Memory Management

### Token-Based History Limiting

```python
from transformers import AutoTokenizer
from typing import List, Dict

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')

class ContextManager:
    """Manage context window budget."""
    
    # Groq Llama 3.1 supports 128K tokens
    MAX_CONTEXT_TOKENS = 128000
    SYSTEM_PROMPT_TOKENS = 500
    DOCUMENT_CONTEXT_TOKENS = 5120  # ~10 chunks × 512 tokens
    CURRENT_MESSAGE_TOKENS = 500
    RESERVED_FOR_RESPONSE = 8000
    SAFETY_BUFFER = 2000
    
    @classmethod
    def calculate_history_budget(cls) -> int:
        """Calculate available tokens for chat history."""
        used = (
            cls.SYSTEM_PROMPT_TOKENS +
            cls.DOCUMENT_CONTEXT_TOKENS +
            cls.CURRENT_MESSAGE_TOKENS +
            cls.RESERVED_FOR_RESPONSE +
            cls.SAFETY_BUFFER
        )
        return cls.MAX_CONTEXT_TOKENS - used  # ~111,880 tokens available!
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens in text."""
        return len(tokenizer.encode(text, add_special_tokens=False))
    
    @classmethod
    def get_history_for_context(cls, messages: List[Dict]) -> List[Dict]:
        """
        Get chat history that fits within token budget.
        Works backwards from most recent messages.
        
        Args:
            messages: List of {'role': str, 'content': str}
        
        Returns:
            Filtered messages within budget
        """
        budget = cls.calculate_history_budget()
        selected_messages = []
        total_tokens = 0
        
        # Start from most recent
        for message in reversed(messages):
            msg_tokens = cls.count_tokens(message['content'])
            
            if total_tokens + msg_tokens > budget:
                break
            
            selected_messages.insert(0, message)
            total_tokens += msg_tokens
        
        return selected_messages


def build_prompt(
    user_message: str,
    chat_history: List[Dict],
    relevant_chunks: List[Dict]
) -> List[Dict]:
    """
    Build complete prompt for LLM.
    
    Returns:
        Messages list in chat format
    """
    system_message = {
        "role": "system",
        "content": """You are an AI assistant with access to uploaded documents.

INSTRUCTIONS:
- Answer questions using the document context provided below
- If the answer is not in the context, clearly state: "I don't have that information in the uploaded documents"
- Cite sources when referencing documents (e.g., "According to [filename]...")
- Be concise but comprehensive
- If unsure, acknowledge uncertainty"""
    }
    
    messages = [system_message]
    
    # Add document context if available
    if relevant_chunks:
        doc_context = format_context(relevant_chunks)
        messages.append({
            "role": "system",
            "content": doc_context
        })
    
    # Add filtered chat history
    history = ContextManager.get_history_for_context(chat_history)
    messages.extend(history)
    
    # Add current message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    return messages
```

---

## 6. Streaming Implementation

### Server-Sent Events (SSE)

```python
# views.py
from django.http import StreamingHttpResponse
from rest_framework.decorators import api_view
import json
import logging

logger = logging.getLogger(__name__)

@api_view(['POST'])
def chat_stream(request):
    """
    Stream chat responses via SSE.
    
    POST data: {"session_id": "uuid", "message": "user question"}
    """
    session_id = request.data.get('session_id')
    user_message = request.data.get('message')
    
    if not session_id or not user_message:
        return Response(
            {"error": "session_id and message required"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    def event_stream():
        try:
            # Get session
            session = ChatSession.objects.get(id=session_id)
            
            # Save user message
            ChatMessage.objects.create(
                session=session,
                role='user',
                content=user_message
            )
            
            # Build context
            history = list(
                session.messages
                .order_by('-created_at')[:50]
                .values('role', 'content')
            )
            history.reverse()  # Oldest first
            
            chunks = retrieve_relevant_chunks(user_message, top_k=10)
            messages = build_prompt(user_message, history, chunks)
            
            # Stream from LLM
            full_response = []
            for token in stream_from_groq(messages):
                full_response.append(token)
                yield f"data: {json.dumps({'type': 'content', 'content': token})}\n\n"
            
            # Save complete response
            assistant_msg = ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=''.join(full_response)
            )
            
            yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_msg.id})}\n\n"
            
        except ChatSession.DoesNotExist:
            logger.error(f"Session {session_id} not found")
            yield f"data: {json.dumps({'type': 'error', 'error': 'Session not found'})}\n\n"
        
        except Exception as e:
            logger.exception("Error in chat stream")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    response = StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    
    return response


def stream_from_groq(messages: List[Dict]):
    """
    Stream tokens from Groq API.
    
    Yields:
        Token strings
    """
    from groq import Groq
    import os
    
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    try:
        stream = client.chat.completions.create(
            model='llama-3.1-70b-versatile',  # or 'llama-3.1-8b-instant' for speed
            messages=messages,
            stream=True,
            max_tokens=8000,
            temperature=0.7
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"Groq error: {e}")
        yield f"\n\n[Error generating response: {str(e)}]"
```

---

## 7. Django Cache Setup (Instead of Redis)

### Configuration

```python
# settings.py

# Option 1: In-memory cache (fastest, lost on restart)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'rag-cache',
        'OPTIONS': {
            'MAX_ENTRIES': 1000  # Limit memory usage
        }
    }
}

# Option 2: Database cache (persistent, slower)
# CACHES = {
#     'default': {
#         'BACKEND': 'django.core.cache.backends.db.DatabaseCache',
#         'LOCATION': 'cache_table',
#     }
# }
# Run: python manage.py createcachetable

# Option 3: File-based cache (persistent, medium speed)
# CACHES = {
#     'default': {
#         'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache',
#         'LOCATION': '/var/tmp/django_cache',
#     }
# }
```

### Usage

```python
from django.core.cache import cache
import hashlib

def get_cache_key(prefix: str, text: str) -> str:
    """Generate cache key from text."""
    text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
    return f"{prefix}:{text_hash}"

def cache_embedding(text: str, embedding: list):
    """Cache embedding for 1 hour."""
    key = get_cache_key('emb', text)
    cache.set(key, embedding, timeout=3600)

def get_cached_embedding(text: str) -> list | None:
    """Retrieve cached embedding."""
    key = get_cache_key('emb', text)
    return cache.get(key)

# Usage in retrieval
def retrieve_relevant_chunks_cached(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve with caching."""
    
    # Check embedding cache
    cached_embedding = get_cached_embedding(query)
    if cached_embedding:
        query_embedding = cached_embedding
    else:
        query_embedding = embedding_service.generate_embedding(query)
        cache_embedding(query, query_embedding.tolist())
    
    # Rest of retrieval logic...
    return retrieve_relevant_chunks(query, top_k)
```

---

## 8. Simple Rate Limiting

```python
# middleware.py
from django.core.cache import cache
from django.http import JsonResponse
from django.utils import timezone

class SimpleRateLimitMiddleware:
    """Rate limit using Django cache."""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        if request.path.startswith('/api/chat/'):
            # Use user ID or IP address
            user_id = getattr(request.user, 'id', None)
            identifier = user_id if user_id else request.META.get('REMOTE_ADDR')
            
            key = f"rate_limit:{identifier}"
            
            # Get request count
            count = cache.get(key, 0)
            
            # Limit: 20 requests per minute
            if count >= 20:
                return JsonResponse(
                    {
                        "error": "Rate limit exceeded. Please wait before sending more messages.",
                        "retry_after": 60
                    },
                    status=429
                )
            
            # Increment counter
            cache.set(key, count + 1, timeout=60)
        
        return self.get_response(request)

# Add to settings.py
# MIDDLEWARE = [
#     ...
#     'app.middleware.SimpleRateLimitMiddleware',
# ]
```

---

## 9. Complete Requirements (Minimal)

```txt
# requirements.txt - Minimal for MVP

# Core Django & Database
Django==4.2.7
djangorestframework==3.14.0
psycopg2-binary==2.9.9
django-cors-headers==4.3.1
django-environ==0.10.0
pgvector==0.2.4

# LangGraph & AI Orchestration
langgraph>=0.2,<0.3
langchain>=0.3,<0.4
langchain-core>=0.3,<0.4

# LLM Provider
groq==0.9.0

# Embeddings & NLP
sentence-transformers==2.5.1
transformers==4.35.2
torch>=2.2.0,<2.6

# Text Processing
numpy>=1.26.0,<2.0
python-dotenv==1.0.0

# Development
pytest==7.4.2
pytest-django==4.5.2
black==23.9.1
flake8==6.1.0

# Production
gunicorn==21.2.0
whitenoise==6.5.0
```

---

## 10. Quick Start Guide

### Step 1: Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cat > .env << EOF
DEBUG=True
SECRET_KEY=your-secret-key-here-change-in-production
DATABASE_URL=postgresql://user:password@localhost/ragdb
GROQ_API_KEY=your-groq-api-key
ALLOWED_HOSTS=localhost,127.0.0.1
EOF

# Create database
createdb ragdb
psql ragdb -c "CREATE EXTENSION vector;"

# Run migrations
python manage.py migrate

# Load models (first run will download ~80MB for embeddings)
python manage.py shell
>>> from sentence_transformers import SentenceTransformer
>>> SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
>>> exit()

# Create superuser
python manage.py createsuperuser

# Run server
python manage.py runserver
```

### Step 2: Test Upload

```bash
# Create test file
echo "Django is a Python web framework. It follows the MVC pattern." > test.txt

# Upload document
curl -X POST http://localhost:8000/api/document/ \
  -F "file=@test.txt"

# Response:
# {
#   "document_id": 1,
#   "filename": "test.txt",
#   "status": "processed",
#   "total_chunks": 1,
#   "processing_time": "2.34s"
# }
```

### Step 3: Test Chat

```bash
# Create session
curl -X POST http://localhost:8000/api/session/create/

# Chat (with SSE)
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<session-id>",
    "message": "What is Django?"
  }'
```

---

## 11. Project Structure

```
rag_project/
├── manage.py
├── requirements.txt
├── .env
├── .gitignore
│
├── config/                 # Django project settings
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── apps/
│   └── rag/               # Main RAG app
│       ├── __init__.py
│       ├── apps.py        # Load models on startup
│       ├── models.py      # DB models
│       ├── views.py       # API views
│       ├── urls.py
│       ├── serializers.py
│       │
│       ├── services/      # Business logic
│       │   ├── __init__.py
│       │   ├── embeddings.py
│       │   ├── chunking.py
│       │   ├── retrieval.py
│       │   └── context.py
│       │
│       └── middleware.py  # Rate limiting
│
└── tests/
    ├── __init__.py
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_views.py
```

### Key Files

**apps.py - Load Models on Startup:**
```python
from django.apps import AppConfig

class RagConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.rag'
    
    def ready(self):
        """Load embedding model and tokenizer on startup."""
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer
        
        # Load once, reuse across requests
        SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
        
        print("✓ Models loaded successfully")
```

**models.py:**
```python
from django.db import models
from pgvector.django import VectorField

class Document(models.Model):
    filename = models.CharField(max_length=255)
    file_size = models.IntegerField()
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('processing', 'Processing'),
            ('processed', 'Processed'),
            ('failed', 'Failed')
        ],
        default='pending'
    )
    error_message = models.TextField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    total_chunks = models.IntegerField(default=0)
    
    class Meta:
        db_table = 'documents'
        ordering = ['-uploaded_at']


class DocumentChunk(models.Model):
    document = models.ForeignKey(
        Document, 
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    content = models.TextField()
    embedding = VectorField(dimensions=384)  # all-MiniLM-L6-v2 dimension
    chunk_index = models.IntegerField()
    token_count = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'document_chunks'
        unique_together = ['document', 'chunk_index']
        indexes = [
            models.Index(fields=['document', 'chunk_index']),
        ]


class ChatSession(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'chat_sessions'


class ChatMessage(models.Model):
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name='messages'
    )
    role = models.CharField(max_length=10)  # 'user' or 'assistant'
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'chat_messages'
        ordering = ['created_at']
```

---

## 12. Production Deployment (Simple)

### Using Gunicorn + Nginx

**gunicorn.conf.py:**
```python
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4  # 2-4 × CPU cores
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Allow 2min for document processing
keepalive = 5
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
```

**nginx.conf:**
```nginx
upstream django {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 10M;
    
    location /static/ {
        alias /path/to/staticfiles/;
    }
    
    location /media/ {
        alias /path/to/media/;
    }
    
    location / {
        proxy_pass http://django;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # SSE specific
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
}
```

**Run:**
```bash
# Collect static files
python manage.py collectstatic --noinput

# Start gunicorn
gunicorn config.wsgi:application -c gunicorn.conf.py

# Or use systemd service
sudo systemctl start rag-app
```

---

## 13. Monitoring & Debugging

### Simple Logging

```python
# settings.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{levelname}] {asctime} {name} - {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/django.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
        'apps.rag': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
    },
}
```

### Health Check Endpoint

```python
# views.py
from django.http import JsonResponse
from django.db import connection

def health_check(request):
    """Simple health check."""
    try:
        # Test DB connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        
        # Test embedding service
        from apps.rag.services.embeddings import embedding_service
        test_embedding = embedding_service.generate_embedding("test")
        
        return JsonResponse({
            "status": "healthy",
            "database": "connected",
            "embedding_service": "loaded",
            "embedding_dim": len(test_embedding)
        })
    except Exception as e:
        return JsonResponse({
            "status": "unhealthy",
            "error": str(e)
        }, status=500)
```

---

## 14. Performance Tips

### Database Optimization

```python
# Use select_related and prefetch_related
messages = ChatMessage.objects.select_related('session').all()

# Bulk operations
DocumentChunk.objects.bulk_create(chunks, batch_size=100)

# Index your vector column properly
# CREATE INDEX idx_chunks_embedding ON document_chunks 
# USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### Batch Embedding Generation

```python
# Instead of one at a time:
# for chunk in chunks:
#     embedding = model.encode(chunk)

# Do batch:
embeddings = model.encode(
    [chunk['text'] for chunk in chunks],
    batch_size=32,  # Process 32 at once
    show_progress_bar=False
)
```

### Connection Pooling

```python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'ragdb',
        'CONN_MAX_AGE': 600,  # Keep connections for 10 minutes
    }
}
```

---

## 15. Testing

### Basic Tests

```python
# tests/test_chunking.py
import pytest
from apps.rag.services.chunking import chunk_document

def test_chunk_document():
    text = "This is a test sentence. " * 100
    chunks = chunk_document(text)
    
    assert len(chunks) > 0
    assert all(chunk['token_count'] <= 512 for chunk in chunks)
    assert all(len(chunk['embedding']) == 384 for chunk in chunks)

# tests/test_retrieval.py
import pytest
from apps.rag.services.retrieval import retrieve_relevant_chunks

@pytest.mark.django_db
def test_retrieval():
    # Assuming test data exists
    chunks = retrieve_relevant_chunks("test query", top_k=5)
    
    assert len(chunks) <= 5
    assert all('content' in chunk for chunk in chunks)
    assert all('similarity' in chunk for chunk in chunks)

# Run tests
# pytest --verbose
```

---

## 16. When to Upgrade

### Add Celery When:
- Document processing takes >10 seconds
- Need to handle >5 concurrent uploads
- Want background processing for large files

### Add Redis When:
- >100 concurrent users
- Cache hit rate matters for cost/performance
- Need cache persistence across restarts
- Need distributed rate limiting

### Add Reranking When:
- Users report irrelevant results
- Need better retrieval precision
- Have processing budget for extra step

### Migrate Vector DB When:
- >1M document chunks
- pgvector queries become slow (>2s)
- Need specialized vector features

---

## 17. Summary

### What You're Building

✅ **MVP RAG System with:**
- Synchronous document upload (2-5s processing)
- Semantic search with pgvector
- Streaming chat responses
- Token-based context management
- Simple caching and rate limiting
- Production-ready deployment

### Implementation Timeline

**Week 1: Core**
- Django setup + PostgreSQL
- Document upload & chunking
- Embedding generation
- Basic retrieval

**Week 2: Chat**
- SSE streaming
- Context management
- LangGraph integration
- Simple UI

**Week 3: Polish**
- Rate limiting
- Logging & monitoring
- Tests
- Deployment

### Total Complexity

- **Lines of code:** ~1,500
- **Services to run:** 2 (Django + PostgreSQL)
- **External dependencies:** 1 (Groq API)
- **Time to deploy:** ~2-3 weeks

This is a **simple, production-ready system** you can actually build and deploy quickly. Add complexity (Redis, Celery, reranking) only when you need it!

---

**Document Version:** MVP 1.0  
**Target:** Quick Launch with Essential Features  
**Philosophy:** Start simple, scale when needed
