# rag/services/agent.py
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
import operator
import os

# Import your existing services
from .retrieval import retrieve_relevant_chunks, format_context
from .context import build_prompt

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    context: str
    query: str
    session_id: int

def retrieve_node(state: AgentState):
    """RAG retrieval node"""
    chunks = retrieve_relevant_chunks(state['query'], top_k=10)
    context = format_context(chunks)
    return {"context": context}

def generate_node(state: AgentState):
    """LLM generation node using direct Groq API"""
    import requests
    import json
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    
    # Convert context string back to list format for build_prompt
    context_str = state.get('context', '')
    
    # Create proper chunks list with all required fields
    if context_str and context_str != "No relevant context found in uploaded documents.":
        relevant_chunks = [{
            'content': context_str, 
            'filename': 'retrieved_documents', 
            'chunk_index': 0,
            'similarity': 1.0
        }]
    else:
        relevant_chunks = []
    
    # Build prompt messages
    prompt_messages = build_prompt(
        state['query'],
        state.get('messages', []),
        relevant_chunks
    )
    
    # Prepare API request
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": prompt_messages,
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.7,
        "max_tokens": 8000,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        assistant_message = result['choices'][0]['message']['content']
        
        return {"messages": [{"role": "assistant", "content": assistant_message}]}
        
    except Exception as e:
        print(f"Groq API error: {e}")
        return {"messages": [{"role": "assistant", "content": "I encountered an error while processing your request. Please try again."}]}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.set_entry_point("retrieve")

# Compile the agent
agent = workflow.compile()

# Helper function to run the agent
def run_agent(query: str, chat_history: list, session_id: int) -> str:
    """
    Run the LangGraph agent.
    
    Args:
        query: User's question
        chat_history: Previous messages [{"role": "user/assistant", "content": "..."}]
        session_id: Chat session ID
    
    Returns:
        Assistant's response text
    """
    initial_state = {
        "query": query,
        "messages": chat_history,
        "context": "",
        "session_id": session_id
    }
    
    result = agent.invoke(initial_state)
    
    # Extract the last assistant message
    if result.get("messages"):
        return result["messages"][-1]["content"]
    return "I apologize, but I couldn't generate a response."