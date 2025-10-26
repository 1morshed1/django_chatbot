# rag/services/retrieval.py
from django.db import connection
from .embeddings import embedding_service


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.5
) -> list:
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


def format_context(chunks: list) -> str:
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
