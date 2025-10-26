# rag/services/chunking.py

from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .embeddings import embedding_service


# Load tokenizer once - using GPT-2 (open, no auth required)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print("✓ Tokenizer loaded")


def count_tokens(text: str) -> int:
    """Count tokens using tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# Create splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=count_tokens,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
)


def chunk_document(text: str) -> list:
    """
    Chunk document and generate embeddings.
    
    Args:
        text: Document text content
    
    Returns:
        List of dicts with 'text', 'token_count', 'embedding', 'chunk_index'
    """
    # Split into chunks
    chunks = splitter.split_text(text)
    
    # Batch generate embeddings
    embeddings = embedding_service.batch_generate_embeddings(chunks)
    
    # Combine results
    return [
        {
            'text': chunk,
            'token_count': count_tokens(chunk),
            'embedding': embedding.tolist(),
            'chunk_index': i
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
