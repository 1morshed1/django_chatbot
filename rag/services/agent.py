# rag/services/agent.py
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Iterator
import operator
import os
import requests
import json

from .retrieval import retrieve_relevant_chunks
from .context import build_prompt

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    context: list  # Changed from str to list
    query: str
    session_id: int

def retrieve_node(state: AgentState):
    """RAG retrieval node"""
    chunks = retrieve_relevant_chunks(state['query'], top_k=10)
    return {"context": chunks}  # Return list, not string

def generate_node(state: AgentState):
    """LLM generation node - non-streaming version for graph execution"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    
    # Build prompt messages
    prompt_messages = build_prompt(
        state['query'],
        state.get('messages', []),
        state.get('context', [])
    )
    
    # Non-streaming call for LangGraph
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
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        assistant_message = result['choices'][0]['message']['content']
        return {"messages": [{"role": "assistant", "content": assistant_message}]}
    except Exception as e:
        print(f"Groq API error: {e}")
        return {"messages": [{"role": "assistant", "content": "Error processing request."}]}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.set_entry_point("retrieve")

# Compile the agent
agent = workflow.compile()

def run_agent(query: str, chat_history: list, session_id: int) -> str:
    """
    Run the LangGraph agent (non-streaming).
    
    Args:
        query: User's question
        chat_history: Previous messages
        session_id: Chat session ID
        
    Returns:
        Assistant's response text
    """
    initial_state = {
        "query": query,
        "messages": chat_history,
        "context": [],
        "session_id": session_id
    }
    
    result = agent.invoke(initial_state)
    
    if result.get("messages"):
        return result["messages"][-1]["content"]
    return "I apologize, but I couldn't generate a response."


def run_agent_streaming(query: str, chat_history: list, session_id: int) -> Iterator[str]:
    """
    Run agent with TRUE streaming from Groq API.
    
    This function:
    1. Retrieves relevant documents
    2. Builds prompt with context
    3. Streams tokens directly from Groq API
    
    Args:
        query: User's question
        chat_history: Previous messages
        session_id: Chat session ID
        
    Yields:
        Token strings as they arrive from LLM
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Retrieve relevant chunks
        logger.info(f"Retrieving context for query: {query[:50]}...")
        chunks = retrieve_relevant_chunks(query, top_k=10)
        logger.info(f"Retrieved {len(chunks)} relevant chunks")
        
        # Step 2: Build prompt with context
        prompt_messages = build_prompt(query, chat_history, chunks)
        
        # Step 3: Stream from Groq API
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        
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
            "stream": True  # ✅ TRUE STREAMING
        }
        
        logger.info("Starting streaming request to Groq API...")
        
        response = requests.post(
            url, 
            headers=headers, 
            json=data, 
            stream=True,  # ✅ Enable streaming
            timeout=120
        )
        response.raise_for_status()
        
        # Stream tokens as they arrive
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        chunk_data = json.loads(data_str)
                        delta = chunk_data['choices'][0]['delta']
                        
                        # Yield content token if present
                        if 'content' in delta:
                            content = delta['content']
                            if content:
                                yield content
                                
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.warning(f"Error parsing stream chunk: {e}")
                        continue
        
        logger.info("Streaming completed successfully")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Groq API request failed: {e}")
        yield "\n\nI encountered an error while processing your request. Please try again."
    
    except Exception as e:
        logger.error(f"Unexpected error in streaming: {e}")
        yield "\n\nAn unexpected error occurred. Please try again."