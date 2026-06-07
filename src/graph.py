import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from mcp import ClientSession
from contextlib import AsyncExitStack 
from langchain_mcp_adapters.tools import load_mcp_tools
from src.vector_store import LanceIndexingVault
from mcp.client.sse import sse_client

load_dotenv()

# =====================================================================
# OPTIMIZATION 1: Cache LLM instance globally to avoid reinitialization
# =====================================================================
_cached_llm = None

def get_cached_llm():
    """Returns a singleton LLM instance to avoid memory overhead per request."""
    global _cached_llm
    if _cached_llm is None:
        print("⚡ Initializing cached LLM instance...")
        _cached_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    return _cached_llm


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    focused_file_path: str
    mcp_session: ClientSession         # Holds our active protocol connection reference
    mcp_tools_cache: list              # Stores converted LangChain tools
    vault: LanceIndexingVault          # Reused vault instance for semantic search


# ----------------------------------------------------------------------
# 1. Top-Level Node Architecture
# ----------------------------------------------------------------------

def analysis_node(state: AgentState):
    messages = state["messages"]
    last_user_query = messages[-1].content
    workspace_root = state.get("focused_file_path", "")

    # Query LanceDB to pull target snippets matching the query context
    vault = state.get("vault")
    if vault is None:
        vault = LanceIndexingVault()
    contexts = vault.semantic_code_search(str(last_user_query), limit=3)

    context_blocks = []
    for c in contexts:
        # Resolve path using the active user workspace directory
        resolved_path = os.path.join(workspace_root, c['file_path']) if workspace_root and not os.path.isabs(
            c['file_path']) else c['file_path']
        context_blocks.append(
            f"--- Code from path: {resolved_path} ---\n{c['source_code']}")

    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are RepoIntel v2, an expert software architect executing inside a production "
        "MCP decoupled framework. Use your tools to read, edit, or test local source code.\n"
        "CRITICAL: When requested to write or modify code, always call 'update_local_file' with the "
        "EXACT file path provided in the context header blocks above.\n\n"
        f"Retrieved Code Semantics Context:\n{context_str}"
    )

    llm = get_cached_llm()
    langchain_tools = state.get("mcp_tools_cache", [])
    bound_llm = llm.bind_tools(langchain_tools) if langchain_tools else llm

    response = bound_llm.invoke([HumanMessage(content=system_prompt)] + list(messages))
    return {"messages": [response]}


async def execution_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    session = state["mcp_session"]  # Threaded through active graph context
    tool_outputs = []
    tool_calls = getattr(last_message, "tool_calls", []) or []

    for call in tool_calls:
        print(f"📡 MCP Client routing call to Server: {call['name']}...")
        result = await session.call_tool(call["name"], arguments=call["args"])
        tool_outputs.append(
            ToolMessage(content=str(result.content), tool_call_id=call["id"])
        )

    return {"messages": tool_outputs}


def validation_gate(state: AgentState):
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
    if tool_calls:
        return "execute"
    return END


# ----------------------------------------------------------------------
# 2. Graph Definition & Compilation
# ----------------------------------------------------------------------

workflow = StateGraph(AgentState)
workflow.add_node("analyze", analysis_node)
workflow.add_node("execute", execution_node)

workflow.set_entry_point("analyze")
workflow.add_conditional_edges(
    "analyze",
    validation_gate,
    {"execute": "execute", END: END}
)
workflow.add_edge("execute", "analyze")

compiled_graph = workflow.compile()


# ----------------------------------------------------------------------
# 3. Execution Router Interface
# ----------------------------------------------------------------------

# Standardize fallback to match your local bare-metal loopback root address
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:5001/")


class RealMCPClientRouter:
    """Manages a single, long-lived persistent connection session to the remote MCP microservice container."""

    def __init__(self):
        self.graph = compiled_graph
        self.exit_stack = None
        self.session = None
        self.mcp_tools = []
        self.vault = LanceIndexingVault()  # ◄ OPTIMIZATION 5: Reuse single vault instance


    # Inside src/graph.py -> RealMCPClientRouter class

    async def start_session(self):
        """Spins up the network client stream connection natively on gateway startup."""
        if self.session is not None:
            return  

        # Pull exact endpoint target directly
        target_endpoint = MCP_SERVER_URL.rstrip("/")
        print(f"🔌 Connecting to network MCP Stream endpoint: {target_endpoint} ...")
        self.exit_stack = AsyncExitStack()

        try:
            read_stream, write_stream = await self.exit_stack.enter_async_context(
                sse_client(target_endpoint)
            )

            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            await self.session.initialize()
            self.mcp_tools = await load_mcp_tools(self.session)
            print("🛰️ Remote network MCP session container bound successfully.")

        except Exception as e:
            print(f"❌ Structural Failure connecting to runtime service: {e}")
            await self.stop_session()
            # 💡 CRITICAL: Raise the error out of the startup hook! 
            # This stops the gateway server immediately so you don't run an offline mesh
            raise SystemExit("Stopping gateway because background MCP server is unreachable.")

    async def stop_session(self):
        """Tears down network streaming contexts gracefully on service stop events."""
        if self.exit_stack:
            print("🛑 Disconnecting remote MCP target streams safely...")
            await self.exit_stack.aclose()
            self.session = None
            self.exit_stack = None

    async def run_pipeline(self, initial_state: AgentState):
        """Runs the LangGraph framework injecting the persistent session state properties."""
        if not self.session:
            raise RuntimeError(
                "MCP Routing architecture engine is offline. Start session context first.")

        initial_state["mcp_session"] = self.session
        initial_state["mcp_tools_cache"] = self.mcp_tools
        initial_state["vault"] = self.vault  # ◄ OPTIMIZATION 5: Pass cached vault instance

        return await self.graph.ainvoke(initial_state)


mcp_router = RealMCPClientRouter()