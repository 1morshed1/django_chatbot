# RESEARCH_AND_ANALYSIS.md

## 1. System Architecture Overview

### High-Level Architecture

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
│  • /api/document/      (POST - Upload documents)    │
│  • /api/session/<id>/  (GET - Retrieve history)     │
└──────┬──────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Orchestration Layer           │
├─────────────────────────────────────────────────────┤
│  Node 1: Semantic Search (Find relevant chunks)     │
│  Node 2: Context Builder (History + Documents)      │
│  Node 3: LLM Invocation (Groq API)                  │
│  Node 4: Response Processor (Save to DB)            │
└──────┬──────────────────────────────────────────────┘
       │
       ├─────────────────┬─────────────────┐
       ▼                 ▼                 ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│   Groq API  │  │  PostgreSQL  │  │  Embedding   │
│   (LLM)     │  │  (pgvector)  │  │   Service    │
└─────────────┘  └──────────────┘  └──────────────┘
                        │
                        ├── ChatSession
                        ├── ChatMessage
                        ├── Document
                        └── DocumentChunk (with vectors)
```

### Request Flow

**Chat Flow:**
1. User sends message via POST `/api/chat/`
2. Django validates session and saves user message
3. LangGraph orchestrates the workflow:
   - **Step 1:** Generate query embedding from user message
   - **Step 2:** Perform semantic search against DocumentChunk vectors
   - **Step 3:** Retrieve last N tokens of chat history
   - **Step 4:** Build prompt with context (history + relevant chunks)
   - **Step 5:** Stream tokens from Groq LLM via SSE
   - **Step 6:** Save complete assistant response to database
4. Client receives streamed response in real-time

**Document Upload Flow:**
1. User uploads .txt file via POST `/api/document/`
2. Django validates file type and saves Document record
3. Background/synchronous process:
   - Split document into fixed token-size chunks (512 tokens with 50 token overlap)
   - Generate embeddings using `all-MiniLM-L6-v2`
   - Store chunks with embeddings in DocumentChunk table
4. Return success response with document ID

---

## 2. Technology Justifications

### Django
**Rationale:**
- **Mature ORM:** Built-in PostgreSQL support with excellent migration management
- **Django REST Framework (DRF):** Industry-standard for building REST APIs with serializers, viewsets, and authentication
- **Ecosystem:** Rich plugin ecosystem (django-cors-headers, django-environ, etc.)
- **Streaming Support:** Built-in `StreamingHttpResponse` for SSE implementation
- **Production-Ready:** Battle-tested in enterprise environments with good documentation

**Alternatives Considered:** FastAPI (async-first, but Django's maturity and ORM won out for this use case)

### PostgreSQL with pgvector
**Rationale:**
- **Single Database Solution:** Store relational data (sessions, messages) and vector embeddings in one place, reducing operational complexity
- **pgvector Extension:** Native support for vector similarity search with indexing (IVFFlat, HNSW)
- **ACID Compliance:** Ensures data consistency for chat history
- **Cost-Effective:** No need for separate vector database (Pinecone, Weaviate, etc.)
- **Performance:** Sufficient for moderate scale (thousands of documents); can scale with proper indexing

**Trade-off:** Specialized vector DBs might be faster for millions of vectors, but pgvector is simpler and sufficient here.

### LangGraph
**Rationale:**
- **Orchestration:** Manages complex multi-step workflows (retrieve → build context → call LLM → save)
- **State Management:** Maintains state across nodes, making it easy to pass context, history, and documents
- **Streaming Support:** Native integration with LangChain's streaming capabilities
- **Debuggability:** Clear graph structure makes workflow easier to understand and debug
- **Extensibility:** Easy to add new nodes (e.g., guardrails, query rewriting, multi-document comparison)

**Alternatives Considered:** Direct LLM calls (too simplistic for complex workflows), custom orchestration (reinventing the wheel)

### Groq (LLM Provider)
**Rationale:**
- **Speed:** Groq's LPU inference is extremely fast, providing excellent user experience for streaming
- **Free Tier:** Generous free tier with high rate limits for development and testing
- **Chat Inference:** Full support for chat completion APIs with streaming
- **Model Options:** Access to Llama 3, Mixtral, and other open-source models
- **Cost-Effective:** After free tier, pricing is competitive compared to OpenAI/Anthropic

**Model Choice:** `llama-3.1-70b-versatile` for best quality

**Trade-off:** Less customization than self-hosted models, but significantly simpler deployment and maintenance.

---

## 3. Document Retrieval Plan

### Chunking Strategy: Fixed Token Size

**Implementation Details:**
- **Chunk Size:** 512 tokens per chunk
- **Overlap:** 50 tokens between consecutive chunks to maintain context continuity
- **Tokenizer Strategy:** Dual-tokenizer approach
  - **For chunking/embeddings:** Use `AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')`
  - **For LLM context management:** Use `AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')`
  - **Reasoning:** Ensures chunking aligns with embedding model while LLM context budgeting uses correct tokenization

**Chunking Implementation:**
```python
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# For document chunking (aligns with embedding model)
embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# For LLM context management  
llm_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

def chunk_with_embedding_tokenizer(text, chunk_size=512, overlap=50):
    tokens = embedding_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = embedding_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append({
            'text': chunk_text,
            'embedding_tokens': len(chunk_tokens),
            'llm_tokens': len(llm_tokenizer.encode(chunk_text))
        })
        start = end - overlap
        
    return chunks
```

### Embedding Storage: PostgreSQL with pgvector

**Schema Design:**
```sql
CREATE EXTENSION vector;

CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    content TEXT NOT NULL,
    embedding vector(384),  -- all-MiniLM-L6-v2 produces 384-dim vectors
    chunk_index INTEGER,
    token_count INTEGER,        -- Embedding tokens (for chunking validation)
    llm_token_count INTEGER,    -- LLM tokens (for context budgeting)
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_chunks_embedding ON document_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_chunks_document ON document_chunks(document_id, chunk_index);
```

**Indexing Strategy:**
- Use IVFFlat index for approximate nearest neighbor search (faster than exact search)
- `lists = 100` for up to ~10,000 chunks (adjust based on data size: lists ≈ sqrt(total_rows))
- For larger datasets (>100k chunks), consider HNSW indexing (more accurate but slower builds)

### Embedding Model: sentence-transformers/all-MiniLM-L6-v2

**Rationale:**
- **Free & Open Source:** No API costs, run locally
- **Fast:** 384-dimensional embeddings, quick inference (~5ms per text on CPU)
- **Quality:** Good balance between speed and semantic understanding
- **Small Model Size:** ~80MB, easy to deploy
- **Multilingual:** Decent support for non-English languages

**Implementation Pattern:**
```python
from sentence_transformers import SentenceTransformer

# Load once at startup
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def generate_embedding(text: str):
    return embedding_model.encode(text, normalize_embeddings=True)
```

**Alternative Considered:** OpenAI embeddings (better quality but costly at scale and requires API calls)

### Retrieval Strategy: Top-K Cosine Similarity

**Implementation:**
- **K Value:** Retrieve top 3-5 most relevant chunks (configurable)
- **Similarity Metric:** Cosine similarity (angle between vectors, normalized)
- **Threshold:** Optional minimum similarity score (e.g., 0.7) to filter irrelevant chunks
- **Reranking:** For advanced implementation, could add cross-encoder reranking for better precision

**Query Process:**
1. Generate embedding for user's query using the same model
2. Execute pgvector similarity search:
   ```sql
   SELECT 
       content, 
       1 - (embedding <=> query_vector::vector) as similarity,
       document_id,
       chunk_index
   FROM document_chunks
   ORDER BY embedding <=> query_vector::vector
   LIMIT 5;
   ```
3. Filter chunks by similarity threshold (if configured)
4. Inject into LLM prompt with source metadata

**Context Injection Format:**
```
Relevant Document Context:
---
[Document: filename.txt, Chunk 1, Similarity: 0.89]
{chunk content}

[Document: filename.txt, Chunk 2, Similarity: 0.85]
{chunk content}
---

Use the above context to answer the following question. If the answer is not in the context, say so.
```

---

## 4. Chat Memory Design

### History Management: Token-Based Limit

**Strategy:**
- **Token Budget:** Reserve 2000 tokens for chat history in the context window
- **Model Context Window:** Groq's Llama 3.1 supports 128k tokens, but we'll be conservative with total usage
- **Calculation Approach:**
  1. Start with most recent message
  2. Work backwards through history
  3. Accumulate messages until reaching token budget
  4. Truncate older messages if budget exceeded

**Token Allocation (Example for 8k context window):**
```
System Prompt:              ~500 tokens
Document Chunks (5×512):    ~2560 tokens
Chat History:               ~2000 tokens
User's Current Message:     ~200 tokens
Reserved for Response:      ~2740 tokens
─────────────────────────────────────────
Total:                      8000 tokens
```

**For Llama 3.1 (128k context), we can be more generous:**
```
System Prompt:              ~500 tokens
Document Chunks (10×512):   ~5120 tokens
Chat History:               ~8000 tokens  (larger history)
User's Current Message:     ~500 tokens
Reserved for Response:      ~4000 tokens
─────────────────────────────────────────
Total:                      ~18,120 tokens (leaving room for safety)
```

### Context Formatting for LLM

**Prompt Structure:**
```
SYSTEM: You are an AI assistant with access to uploaded documents. 
Answer questions based on the provided context. If the information 
is not in the context, clearly state that you don't have that information.

CONTEXT FROM DOCUMENTS:
[Document chunks here with metadata]

CONVERSATION HISTORY:
User: [message 1]
Assistant: [response 1]
User: [message 2]
Assistant: [response 2]

CURRENT USER MESSAGE:
User: [current message]