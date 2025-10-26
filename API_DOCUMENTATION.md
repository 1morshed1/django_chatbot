# API Documentation

## Overview

Django RAG Chatbot REST API provides endpoints for document-aware conversational AI using Retrieval-Augmented Generation (RAG).

**Base URL:** `http://localhost:8001/api/`  
**Content-Type:** `application/json`  
**Authentication:** None (add authentication in production)

---

## Endpoints

### 1. Health Check

Check the health status of the system and verify all services are running.

**Endpoint:** `GET /api/health/`

**Request:**
```bash
curl http://localhost:8001/api/health/
```

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "database": "connected",
  "embedding_service": "loaded",
  "embedding_dim": 384,
  "groq_api_configured": true
}
```

**Response Fields:**
- `status` (string): Overall system status (`healthy` or `unhealthy`)
- `database` (string): Database connection status
- `embedding_service` (string): Embedding model status
- `embedding_dim` (integer): Embedding vector dimensions
- `groq_api_configured` (boolean): Groq API key presence

**Error Response:** `500 Internal Server Error`
```json
{
  "status": "unhealthy",
  "error": "Error message here"
}
```

---

### 2. Create Chat Session

Create a new conversation session for managing chat history.

**Endpoint:** `POST /api/session/create/`

**Request:**
```bash
curl -X POST http://localhost:8001/api/session/create/
```

**Response:** `201 Created`
```json
{
  "id": 1,
  "created_at": "2025-10-26T10:00:00.000000Z",
  "updated_at": "2025-10-26T10:00:00.000000Z",
  "messages": []
}
```

**Response Fields:**
- `id` (integer): Unique session identifier
- `created_at` (datetime): Session creation timestamp
- `updated_at` (datetime): Last update timestamp
- `messages` (array): List of messages (empty for new session)

**Use Case:**
Create a session before starting a conversation. Use the returned `id` in subsequent chat requests.

---

### 3. Upload Document

Upload a text document for semantic knowledge retrieval.

**Endpoint:** `POST /api/document/`

**Request:**
```bash
curl -X POST http://localhost:8001/api/document/ \
  -F "file=@/path/to/document.txt"
```

**Request Parameters:**
- `file` (file, required): Text file to upload

**File Constraints:**
- **Format:** `.txt` only
- **Max Size:** 5MB
- **Encoding:** UTF-8

**Response:** `201 Created`
```json
{
  "document_id": 1,
  "filename": "document.txt",
  "status": "processed",
  "total_chunks": 25,
  "processing_time": "2.34s"
}
```

**Response Fields:**
- `document_id` (integer): Unique document identifier
- `filename` (string): Original filename
- `status` (string): Processing status (`processed`, `failed`)
- `total_chunks` (integer): Number of chunks created
- `processing_time` (string): Time taken to process

**Error Responses:**

`400 Bad Request` - No file provided
```json
{
  "error": "No file provided"
}
```

`413 Payload Too Large` - File exceeds 5MB
```json
{
  "error": "File too large. Maximum 5MB"
}
```

`415 Unsupported Media Type` - Invalid file type
```json
{
  "error": "Only .txt files supported"
}
```

`500 Internal Server Error` - Processing failed
```json
{
  "error": "Processing failed: <error message>"
}
```

**Processing Details:**
1. File is validated (size, type, encoding)
2. Content is split into chunks (512 tokens, 50 overlap)
3. Embeddings are generated for each chunk (384D vectors)
4. Chunks are stored in database with vectors
5. Document status is updated to `processed`

**Typical Processing Times:**
- 1MB document: 2-3 seconds
- 5MB document: 8-12 seconds

---

### 4. Stream Chat Response

Send a message and receive real-time streaming responses using Server-Sent Events (SSE).

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

**Request Body:**
```json
{
  "session_id": 1,
  "message": "What are the key points in the document?"
}
```

**Request Fields:**
- `session_id` (integer, required): Valid session ID from `/api/session/create/`
- `message` (string, required): User's question or message

**Response:** `200 OK` with `Content-Type: text/event-stream`

**SSE Stream Format:**

```
data: {"type":"start"}

data: {"type":"content","content":"The"}

data: {"type":"content","content":" document"}

data: {"type":"content","content":" discusses"}

data: {"type":"content","content":" three"}

data: {"type":"content","content":" main"}

data: {"type":"content","content":" topics"}

...

data: {"type":"done","message_id":123}
```

**Event Types:**

1. **Start Event**
```json
{"type": "start"}
```
Indicates streaming has begun.

2. **Content Event**
```json
{"type": "content", "content": "token"}
```
Contains a token from the LLM response. Multiple content events are streamed.

3. **Done Event**
```json
{"type": "done", "message_id": 123}
```
Indicates streaming is complete. Includes the saved message ID.

4. **Error Event**
```json
{"type": "error", "error": "Error message"}
```
Indicates an error occurred during processing.

**Error Responses:**

`400 Bad Request` - Missing or invalid parameters
```json
{
  "error": "session_id required"
}
```
```json
{
  "error": "message required and cannot be empty"
}
```

`404 Not Found` - Session doesn't exist
```json
{
  "type": "error",
  "error": "Session not found"
}
```

**Client Implementation Examples:**

**JavaScript (Fetch API):**
```javascript
const response = await fetch('http://localhost:8001/api/chat/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    session_id: 1,
    message: 'What is in the document?'
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      
      if (data.type === 'content') {
        console.log(data.content);
      } else if (data.type === 'done') {
        console.log('Streaming complete');
      }
    }
  }
}
```

**Python (requests):**
```python
import requests
import json

response = requests.post(
    'http://localhost:8001/api/chat/',
    json={
        'session_id': 1,
        'message': 'What is in the document?'
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            if data['type'] == 'content':
                print(data['content'], end='', flush=True)
            elif data['type'] == 'done':
                print('\nStreaming complete')
```

**Processing Flow:**
1. User message is saved to database
2. Chat history is retrieved (last 50 messages)
3. Query embedding is generated
4. Relevant document chunks are retrieved (top 10)
5. Prompt is built with context and history
6. LLM streams tokens in real-time
7. Complete response is saved to database

**Performance:**
- **Time to First Token (TTFT):** ~200-500ms
- **Tokens per Second:** ~50-100 tokens/sec
- **Average Response Time:** 5-15 seconds (depending on length)

---

### 5. Get Chat History

Retrieve all messages from a conversation session.

**Endpoint:** `GET /api/session/<session_id>/messages/`

**Request:**
```bash
curl http://localhost:8001/api/session/1/messages/
```

**Response:** `200 OK`
```json
{
  "id": 1,
  "created_at": "2025-10-26T10:00:00.000000Z",
  "updated_at": "2025-10-26T10:05:00.000000Z",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "What is in the document?",
      "created_at": "2025-10-26T10:01:00.000000Z"
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "The document discusses three main topics: artificial intelligence, machine learning, and natural language processing.",
      "created_at": "2025-10-26T10:01:05.000000Z"
    },
    {
      "id": 3,
      "role": "user",
      "content": "Tell me more about machine learning",
      "created_at": "2025-10-26T10:02:00.000000Z"
    },
    {
      "id": 4,
      "role": "assistant",
      "content": "Machine learning is a subset of artificial intelligence...",
      "created_at": "2025-10-26T10:02:10.000000Z"
    }
  ]
}
```

**Response Fields:**
- `id` (integer): Session ID
- `created_at` (datetime): Session creation time
- `updated_at` (datetime): Last message time
- `messages` (array): List of message objects
  - `id` (integer): Message ID
  - `role` (string): Message role (`user` or `assistant`)
  - `content` (string): Message content
  - `created_at` (datetime): Message timestamp

**Error Response:** `404 Not Found`
```json
{
  "error": "Session not found"
}
```

**Use Case:**
- Display conversation history in UI
- Resume conversations
- Export chat transcripts
- Analyze conversation patterns

---

## Rate Limiting

**Current:** No rate limiting (add in production)

**Recommended Production Limits:**
- Document upload: 10 requests/hour per IP
- Chat: 100 requests/hour per IP
- Other endpoints: 1000 requests/hour per IP

---

## Error Handling

All API endpoints follow consistent error response format:

```json
{
  "error": "Error message describing what went wrong"
}
```

**HTTP Status Codes:**
- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Resource not found
- `413 Payload Too Large` - File size exceeded
- `415 Unsupported Media Type` - Invalid file type
- `500 Internal Server Error` - Server error

---

## Best Practices

### Document Upload
1. **Validate files client-side** before uploading
2. **Handle processing time** - Show loading indicator
3. **Check document status** before querying
4. **Implement retry logic** for failed uploads

### Chat Streaming
1. **Use EventSource API** for browsers (simpler than fetch)
2. **Handle reconnection** if connection drops
3. **Buffer partial responses** for smoother UI
4. **Show typing indicators** during streaming
5. **Implement timeout** for long responses

### Session Management
1. **Create session once** per conversation
2. **Reuse session ID** for related messages
3. **Store session ID** client-side (localStorage)
4. **Handle session expiration** gracefully

### Error Handling
1. **Check response status** before parsing
2. **Display user-friendly** error messages
3. **Implement retry logic** with exponential backoff
4. **Log errors** for debugging

---

## Performance Optimization

### Client-Side
- Debounce user input
- Implement request cancellation
- Cache document IDs
- Prefetch session data

### Server-Side
- Already implemented:
  - Connection pooling
  - Batch embedding generation
  - Vector indexing (add via migration)
  - Streaming responses

---

## Security Considerations

**⚠️ Production Requirements:**

1. **Add Authentication**
   - Implement token-based auth (JWT)
   - Require API keys for all endpoints
   - Use HTTPS only

2. **Rate Limiting**
   - Implement per-user/IP limits
   - Prevent abuse and DoS attacks

3. **Input Validation**
   - Sanitize all user inputs
   - Validate file contents
   - Limit message lengths

4. **CORS Configuration**
   - Restrict origins in production
   - Use credentials properly

---

## Versioning

**Current Version:** v1  
**Base Path:** `/api/`

Future versions will use path versioning:
- v1: `/api/v1/`
- v2: `/api/v2/`

---

## Support

For issues or questions:
- Check [README.md](README.md) for setup
- Review [RESEARCH_AND_ANALYSIS.md](RESEARCH_AND_ANALYSIS.md) for architecture
- Open GitHub issue for bugs
- Contact development team

---

**Last Updated:** October 26, 2025
