# rag/apps.py
from django.apps import AppConfig


class RagConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rag'
    
    def ready(self):
        """Load models on startup."""
        print("\n" + "="*60)
        print("Loading AI models...")
        print("="*60)
        
        # This will trigger model loading
        from .services.embeddings import embedding_service
        from .services.chunking import tokenizer
        
        print("="*60)
        print("✓ All models loaded successfully!")
        print("="*60 + "\n")
