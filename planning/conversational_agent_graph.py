"""
Conversational Agent Graph for a Subject Matter Expert (SME)
This file defines a multi-agent system using LangGraph.

Architecture:
1.  State: A 'GraphState' object holds the conversation memory (`messages`)
    and the next node to route to (`next_node`).
2.  Router: A central 'router' node uses an LLM to look at the
    conversation and decide which specialized agent to call next.
3.  Worker Agents:
    - KnowledgeBase: Uses RAG (via sme_agent) to answer questions or
      generate content.
    - Researcher: Uses Tavily to search the web.
    - Writer: Uses tools to save content to PDF or Markdown.
    - GeneralChat: A simple LLM chain for chit-chat.
4.  Graph: The nodes are wired together. After each worker agent runs,
    control returns to the 'router' node to decide the next step.
"""

import os
import json
import ast
from typing import TypedDict, Annotated, List, Union
import sys

# --- ADDED: Fix for relative import ---
# Add the parent directory (project root) to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# --- END ADDITION ---

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub 
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# --- Local Imports ---
from rag_api.sme_agent import SMEAgent, PDFToolInput, SaveAsPDFTool, MDToolInput, SaveAsMDTool
from langchain_tavily import TavilySearch


# --- 1. Define the State (The System's Memory) ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        messages: The list of messages in the conversation.
        next_node: The next node to route to, as determined by the router.
    """
    messages: Annotated[list, lambda x, y: x + y]
    next_node: str


# ---  Define the "Worker" Agents ---

def create_agent_node(llm: ChatGoogleGenerativeAI, tools: list, system_prompt: str):
    """Helper function to create a new agent node."""
    
    # --- UPDATED: Use a chat-enabled prompt ---
    # This prompt is designed to accept 'input' and 'chat_history'
    prompt = hub.pull("hwchase17/react-chat")
    # --- END UPDATE ---

    prompt = prompt.partial(
        instructions=system_prompt,
        tools=tools,
        tool_names=", ".join([t.name for t in tools]),
    )
    
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor

def create_llm_node(llm: ChatGoogleGenerativeAI, system_prompt: str):
    """Helper function for a simple, no-tool LLM call."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain

# --- 4. Define the Graph Nodes ---
def call_rag_agent(state: GraphState):
    """Node for the RAG (Knowledge Base) agent."""
    messages = state['messages']
    # --- UPDATED: Pass 'input' and 'chat_history' to match the new prompt ---
    # This provides conversational memory to the agent
    input_data = {"input": messages[-1].content, "chat_history": messages[:-1]}
    response = rag_agent_executor.invoke(input_data)
    # --- END UPDATE ---
    return {"messages": [AIMessage(content=response['output'])]}

def call_research_agent(state: GraphState):
    """Node for the Web Research agent."""
    messages = state['messages']
    # --- UPDATED: Pass 'input' and 'chat_history' ---
    input_data = {"input": messages[-1].content, "chat_history": messages[:-1]}
    response = research_agent_executor.invoke(input_data)
    # --- END UPDATE ---
    return {"messages": [AIMessage(content=response['output'])]}

def call_writer_agent(state: GraphState):
    """Node for the Writing & Saving agent."""
    messages = state['messages']
    # --- UPDATED: Pass 'input' and 'chat_history' ---
    input_data = {"input": messages[-1].content, "chat_history": messages[:-1]}
    response = writer_agent_executor.invoke(input_data)
    # --- END UPDATE ---
    return {"messages": [AIMessage(content=response['output'])]}

def call_general_agent(state: GraphState):
    """Node for the General Chatbot."""
    # This node is simple and already uses the full message history
    messages = state['messages']
    response = general_agent_chain.invoke({"messages": messages})
    return {"messages": [AIMessage(content=response)]}


# --- 5. Define the "Router" (Conditional Edge) ---
class Router(BaseModel):
    """
    Route the conversation to the next appropriate agent or finish.
    """
    next_node: str = Field(
        description="The name of the next node to route to.",
        enum=["KnowledgeBase", "Researcher", "Writer", "GeneralChat", "END"]
    )

def define_router_llm(llm: ChatGoogleGenerativeAI):
    """Creates an LLM with tool-calling forced to use the Router model."""
    structured_llm = llm.with_structured_output(Router)
    
    # --- UPDATED: Stricter router prompt to prevent loops ---
    system_prompt = """You are an expert dispatcher routing conversation to the correct agent.
Based on the full conversation history, you must decide the next step.
The last message in the history is the *most important* for your decision.

- If the last message is an AIMessage (from an agent), it means a task just finished.
  Your default action should be to "END" unless the user's *original* request
  clearly requires another, *different* step.
- If the last message is a HumanMessage, route it to the best agent to handle it.
- Do not route to the same agent twice in a row.

Available agents:
- KnowledgeBase: Answers specific questions about data science OR generates content like assignments/quizzes using the internal RAG system.
- Researcher: Finds new or real-time information on the internet.
- Writer: Formats and saves content to files (PDF, Markdown). Use this *only if* content has just been generated or the user explicitly asks to save.
- GeneralChat: For simple conversation, greetings, or follow-ups like "thank you".
- END: Use this if the user's request is fully complete (e.g., they said "thanks" or the agent just answered a question).

Analyze the LAST message in the context of the entire conversation and choose the *single* best agent to route to next.
"""
    # --- END UPDATE ---
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    router_chain = prompt | structured_llm
    return router_chain

# --- THIS IS THE ROUTER *NODE* ---
def route_agent_node(state: GraphState):
    """
    This is now a NODE. It runs the router LLM and stores the
    decision in the 'next_node' field of the state.
    """
    messages = state['messages']
    print("--- ROUTER: Deciding next step... ---")
    router_decision = router_llm.invoke({"messages": messages})
    print(f"--- ROUTER: Decision: {router_decision.next_node} ---")
    
    # Return a dict to update the 'next_node' field in the state
    return {"next_node": router_decision.next_node}

# --- THIS IS THE *CONDITION* FOR THE EDGE ---
def should_route(state: GraphState) -> str:
    """
    This is the "conditional edge." It reads the 'next_node'
    field from the state and returns it to decide where to go.
    """
    return state["next_node"]


# --- Helper function to load API keys ---
def get_api_key_from_json(file_path: str, key_name: str):
    """Reads a specific API key from a JSON credentials file."""
    try:
        with open(file_path, 'r') as f:
            credentials = json.load(f)
        return credentials.get(key_name)
    except Exception as e:
        print(f"Error loading {key_name} from {file_path}: {e}")
        return None

# --- 6. Assemble and Run the Graph ---

# Global definitions for the agent executors
rag_agent_executor: AgentExecutor
research_agent_executor: AgentExecutor
writer_agent_executor: AgentExecutor
general_agent_chain: RunnablePassthrough
router_llm: RunnablePassthrough

if __name__ == "__main__":
    # --- A. Load Credentials & LLM ---
    print("Initializing SMEAgent... (This will load models and connect to Pinecone)")
    assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets.json")
    
    TAVILY_API_KEY = get_api_key_from_json(assets_path, "tavily_api_key")
    if TAVILY_API_KEY:
        os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    else:
        print("Warning: 'tavily_api_key' not found in assets.json. Web search will fail.")
        
    sme_agent = SMEAgent(assets_path=assets_path)
    GEMINI_API_KEY = sme_agent.GEMINI_API_KEY
    print("SMEAgent initialized successfully.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0)

    # --- B. Define REAL Tools based on SMEAgent ---
    @tool
    def knowledge_base_tool(query: str) -> str:
        """
        Use this tool to answer specific questions about data science concepts,
        course material, or internal knowledge.
        """
        return sme_agent._run_qa_rag(query)
    
    @tool
    def assignment_generator_tool(query: str) -> str:
        """
        Use this tool to generate the text content for an assignment, quiz,
        or study guide based on the internal knowledge.
        """
        return sme_agent._run_assignment_rag(query)

    save_as_pdf_tool = SaveAsPDFTool()
    save_as_md_tool = SaveAsMDTool()
    
    web_search_tool = TavilySearch(max_results=3)
    web_search_tool.name = "web_search_tool"


    # --- C. Instantiate Worker Agents ---
    # --- UPDATED: Stricter system prompts to prevent looping ---
    rag_agent_executor = create_agent_node(
        llm,
        [knowledge_base_tool, assignment_generator_tool],
        """You are an expert Q&A assistant and content generator.
Your goal is to answer questions or create content (like assignments) using your tools.
Use the 'chat_history' to understand context and resolve pronouns like 'it'.
**Your final answer MUST be only the factual information or generated content.
Do NOT engage in conversation. Do NOT say "you're welcome" or "thank you".**
"""
    )
    
    research_agent_executor = create_agent_node(
        llm,
        [web_search_tool], # Pass the real tool
        """You are an expert web researcher.
Find the most relevant and up-to-date information to answer the user's query.
Use the 'chat_history' to understand context and resolve pronouns.
**Your final answer MUST be only the factual information.
Do NOT engage in conversation. Do NOT say "you're welcome" or "thank you".**
"""
    )
    
    writer_agent_executor = create_agent_node(
        llm,
        [save_as_pdf_tool, save_as_md_tool],
        """You are an expert technical writer.
Use the context from the conversation history (e.g., a generated assignment) to format and save content as requested by the user.
**Your final answer MUST be only a confirmation of the save (e.g., "Successfully saved...").
Do NOT engage in conversation.**
"""
    )
    # --- END UPDATES ---
    
    general_agent_chain = create_llm_node(
        llm,
        "You are a helpful and friendly conversational assistant. Respond directly to the user's message."
    )
    
    # --- D. Instantiate the Router ---
    router_llm = define_router_llm(llm)

    # --- E. Build the Graph ---
    workflow = StateGraph(GraphState) 

    workflow.add_node("KnowledgeBase", call_rag_agent)
    workflow.add_node("Researcher", call_research_agent)
    workflow.add_node("Writer", call_writer_agent)
    workflow.add_node("GeneralChat", call_general_agent)
    workflow.add_node("router", route_agent_node) 
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",       
        should_route,   
        {
            "KnowledgeBase": "KnowledgeBase",
            "Researcher": "Researcher",
            "Writer": "Writer",
            "GeneralChat": "GeneralChat",
            "END": END,
        },
    )

    workflow.add_edge("KnowledgeBase", "router")
    workflow.add_edge("Researcher", "router")
    workflow.add_edge("Writer", "router")
    workflow.add_edge("GeneralChat", "router")

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    # --- F. Run the Agent ---
    print("\n--- Conversational Agent Graph Initialized ---")
    print("Type 'exit' to quit.")
    
    conversation_config = {"configurable": {"thread_id": "sme-convo-1"}}
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        events = graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=conversation_config
        )
        
        print("\nAgent:")
        final_message = None
        for event in events:
            if "messages" in event:
                final_message = event['messages'][-1]
        
        if final_message:
            final_message.pretty_print()