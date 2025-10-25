from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:
    """Singleton for embedding model."""
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print("Loading embedding model...")
            cls._model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2'
            )
            print("✓ Embedding model loaded")
        return cls._instance
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate normalized embedding for text."""
        return self._model.encode(text, normalize_embeddings=True)
    
    def batch_generate_embeddings(self, texts: list) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )


# Global instance
embedding_service = EmbeddingService()
