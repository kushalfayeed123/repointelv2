import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack  # ◄ Added for persistent tracking

from langchain_mcp_adapters.tools import load_mcp_tools
from src.vector_store import LanceIndexingVault

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    focused_file_path: str
    mcp_session: ClientSession         # Holds our active protocol connection reference
    mcp_tools_cache: list              # Stores converted LangChain tools


# Configure runtime subprocess parameters for our MCP server
server_params = StdioServerParameters(
    command="uv",
    args=["run", "python", "-m", "src.mcp_server"]
)


# ----------------------------------------------------------------------
# 1. Top-Level Node Architecture (Lifted for stable graph scoping)
# ----------------------------------------------------------------------

def analysis_node(state: AgentState):
    messages = state["messages"]
    last_user_query = messages[-1].content
    workspace_root = state.get("focused_file_path", "")

    # Query LanceDB to pull target snippets matching the query context
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

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    langchain_tools = state.get("mcp_tools_cache", [])
    bound_llm = llm.bind_tools(langchain_tools) if langchain_tools else llm

    response = bound_llm.invoke(
        [HumanMessage(content=system_prompt)] + messages)
    return {"messages": [response]}


async def execution_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    session = state["mcp_session"]  # Threaded through active graph context
    tool_outputs = []

    for call in last_message.tool_calls:
        print(f"📡 MCP Client routing call to Server: {call['name']}...")
        result = await session.call_tool(call["name"], arguments=call["args"])
        tool_outputs.append(
            ToolMessage(content=str(result.content), tool_call_id=call["id"])
        )

    return {"messages": tool_outputs}


def validation_gate(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
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

class RealMCPClientRouter:
    """Manages a single, long-lived persistent connection session to the MCP server."""

    def __init__(self):
        self.graph = compiled_graph
        self.server_params = StdioServerParameters(
            command="uv",
            args=["run", "python", "-m", "src.mcp_server"]
        )
        self.exit_stack = None
        self.session = None
        self.mcp_tools = []

    async def start_session(self):
        """Spins up the background process once and keeps it alive."""
        if self.session is not None:
            return # Already connected
            
        print("🔌 Connecting to background MCP Server process...")
        self.exit_stack = AsyncExitStack()
        
        # Connect to the stdio transport layer
        read_stream, write_stream = await self.exit_stack.enter_async_context(
            stdio_client(self.server_params)
        )
        
        # Establish the formal MCP Client Session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        
        await self.session.initialize()
        # Pre-cache the tools so we don't have to reload them every chat request
        self.mcp_tools = await load_mcp_tools(self.session)
        print("🛰️ Persistent MCP session connection established cleanly.")

    async def stop_session(self):
        """Gracefully tears down the background subprocess pipes."""
        if self.exit_stack:
            print("🛑 Closing MCP background session...")
            await self.exit_stack.aclose()
            self.session = None
            self.exit_stack = None

    async def run_pipeline(self, initial_state: dict):
        """Executes your LangGraph without thrashing the OS with subprocess initializations."""
        if not self.session:
            raise RuntimeError("MCP routing engine is offline. Call start_session() first.")

        # Thread the already established streaming components straight into the context
        initial_state["mcp_session"] = self.session
        initial_state["mcp_tools_cache"] = self.mcp_tools

        return await self.graph.ainvoke(initial_state)

# Instantiate the instance once globally
mcp_router = RealMCPClientRouter()