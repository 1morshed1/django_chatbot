from transformers import AutoTokenizer
from .retrieval import format_context


# Load tokenizer - using GPT-2 (open, no auth required)
tokenizer = AutoTokenizer.from_pretrained('gpt2')


class ContextManager:
    """Manage context window budget."""
    
    MAX_CONTEXT_TOKENS = 128000
    SYSTEM_PROMPT_TOKENS = 500
    DOCUMENT_CONTEXT_TOKENS = 5120
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
        return cls.MAX_CONTEXT_TOKENS - used
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens in text."""
        return len(tokenizer.encode(text, add_special_tokens=False))
    
    @classmethod
    def get_history_for_context(cls, messages: list) -> list:
        """Get chat history that fits within token budget."""
        budget = cls.calculate_history_budget()
        selected_messages = []
        total_tokens = 0
        
        for message in reversed(messages):
            msg_tokens = cls.count_tokens(message['content'])
            
            if total_tokens + msg_tokens > budget:
                break
            
            selected_messages.insert(0, message)
            total_tokens += msg_tokens
        
        return selected_messages


def build_prompt(
    user_message: str,
    chat_history: list,
    relevant_chunks: list
) -> list:
    """Build complete prompt for LLM."""
    
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
    
    # Add document context
    if relevant_chunks:
        doc_context = format_context(relevant_chunks)
        messages.append({
            "role": "system",
            "content": doc_context
        })
    
    # Add filtered history
    history = ContextManager.get_history_for_context(chat_history)
    messages.extend(history)
    
    # Add current message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    return messages
