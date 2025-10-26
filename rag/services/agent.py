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
    session_id: int  # Add this to track session

def retrieve_node(state: AgentState):
    """RAG retrieval node"""
    chunks = retrieve_relevant_chunks(state['query'], top_k=10)
    context = format_context(chunks)
    return {"context": context}

def generate_node(state: AgentState):
    """LLM generation node"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=8000,
        api_key=api_key
    )
    
    # Build proper message format
    prompt_messages = build_prompt(
        state['query'],
        state.get('messages', []),
        state.get('context', '')
    )
    
    # Convert to LangChain format if needed
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    
    lc_messages = []
    for msg in prompt_messages:
        if msg['role'] == 'system':
            lc_messages.append(SystemMessage(content=msg['content']))
        elif msg['role'] == 'user':
            lc_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            lc_messages.append(AIMessage(content=msg['content']))
    
    response = llm.invoke(lc_messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}

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