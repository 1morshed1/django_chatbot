# rag/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import StreamingHttpResponse
from .models import Document, DocumentChunk, ChatSession, ChatMessage
from .serializers import DocumentSerializer, ChatSessionSerializer
from .services.chunking import chunk_document
from .services.agent import run_agent_streaming  # ✅ Import streaming version
import json
import time
import logging
import os

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 5 * 1024 * 1024


@api_view(['POST'])
def upload_document(request):
    """Upload and process document synchronously."""
    
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
    
    # Validate type
    if not uploaded_file.name.endswith('.txt'):
        return Response(
            {"error": "Only .txt files supported"},
            status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        )
    
    # Create document
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
        
        # Chunk and embed
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
        
        # Update document
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


@api_view(['POST'])
def create_session(request):
    """Create a new chat session."""
    session = ChatSession.objects.create()
    serializer = ChatSessionSerializer(session)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(['GET'])
def get_session(request, session_id):
    """Get chat session with history."""
    try:
        session = ChatSession.objects.get(id=session_id)
        serializer = ChatSessionSerializer(session)
        return Response(serializer.data)
    except ChatSession.DoesNotExist:
        return Response(
            {"error": "Session not found"},
            status=status.HTTP_404_NOT_FOUND
        )


@api_view(['POST'])
def chat_stream(request):
    """
    Stream chat responses via SSE with TRUE token-by-token streaming from Groq API.
    """
    
    session_id = request.data.get('session_id')
    user_message = request.data.get('message')
    
    # Validation
    if not session_id:
        return Response(
            {"error": "session_id required"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if not user_message or not user_message.strip():
        return Response(
            {"error": "message required and cannot be empty"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    def event_stream():
        """Generator for Server-Sent Events with TRUE streaming."""
        
        full_response = ""  # Accumulate for saving to DB
        
        try:
            # Get session
            try:
                session = ChatSession.objects.get(id=session_id)
            except ChatSession.DoesNotExist:
                logger.error(f"Session {session_id} not found")
                yield f"data: {json.dumps({'type': 'error', 'error': 'Session not found'})}\n\n"
                return
            
            # Save user message
            user_msg = ChatMessage.objects.create(
                session=session,
                role='user',
                content=user_message.strip()
            )
            logger.info(f"Saved user message {user_msg.id} to session {session_id}")
            
            # Get chat history (excluding the message we just added)
            history_qs = session.messages.exclude(id=user_msg.id).order_by('created_at')
            history = list(history_qs.values('role', 'content'))
            
            logger.info(f"Retrieved {len(history)} historical messages")
            
            # Send start event
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            
            # Stream tokens from LLM
            logger.info("Starting token stream from Groq API...")
            
            from .services.agent import run_agent_streaming
            
            for token in run_agent_streaming(user_message, history, session_id):
                # Accumulate full response
                full_response += token
                
                # Stream token to client
                yield f"data: {json.dumps({'type': 'content', 'content': token})}\n\n"
            
            logger.info(f"Streaming completed. Total response length: {len(full_response)} chars")
            
            # Save assistant response to database
            if full_response.strip():
                assistant_msg = ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content=full_response.strip()
                )
                logger.info(f"Saved assistant message {assistant_msg.id}")
                
                # Send completion event
                yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_msg.id})}\n\n"
            else:
                logger.warning("Empty response from LLM")
                yield f"data: {json.dumps({'type': 'error', 'error': 'Empty response from LLM'})}\n\n"
            
        except Exception as e:
            logger.exception("Error in chat stream")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    # Create streaming response
    response = StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream'
    )
    
    # Required headers for SSE (REMOVED Connection header!)
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'  # Disable nginx buffering
    # ❌ REMOVED: response['Connection'] = 'keep-alive'  # This causes the error!
    
    return response


@api_view(['GET'])
def health_check(request):
    """Health check endpoint."""
    from django.db import connection
    from .services.embeddings import embedding_service
    
    try:
        # Test DB
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        
        # Test embedding service
        test_embedding = embedding_service.generate_embedding("test")
        
        return Response({
            "status": "healthy",
            "database": "connected",
            "embedding_service": "loaded",
            "embedding_dim": len(test_embedding),
            "groq_api_configured": bool(os.getenv('GROQ_API_KEY'))
        })
    except Exception as e:
        logger.exception("Health check failed")
        return Response({
            "status": "unhealthy",
            "error": str(e)
        }, status=500)