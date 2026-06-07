import os
import asyncio
import traceback
from contextlib import asynccontextmanager
from typing import cast
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from src.workspace import SystemWorkspaceManager
from src.graph import AgentState, mcp_router

workspace_manager = None

# =====================================================================
# LIFESPAN MANAGER: Tells Uvicorn to open the port FIRST, then load tools
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global workspace_manager, mcp_router
    # Informative trace to monitor multi-worker fork distributions
    print(f"\n🚀 WORKER PROCESS INITIALIZED (PID: {os.getpid()})! Network port binding sequence cleared.")

    workspace_manager = SystemWorkspaceManager()

    # Defer LangGraph compilation to prevent memory spikes on startup
    from src.graph import mcp_router as instantiated_router
    mcp_router = instantiated_router

    async def connect_mesh_background():
        print(f"🤖 [PID {os.getpid()}] Connecting to internal decoupled networks in the background...")
        try:
            await mcp_router.start_session()
            print(f"✅ [PID {os.getpid()}] Microservice mesh connections online and ready for traffic.\n")
        except Exception as mesh_err:
            print(f"\n⚠️ [PID {os.getpid()}] BACKGROUND MESH CONNECTION DELAYED: {mesh_err}")
            print("🔄 System will automatically retry connection during standard chat route invocations.")

    asyncio.create_task(connect_mesh_background())

    yield

    if mcp_router:
        await mcp_router.stop_session()

app = FastAPI(title="RepoIntel API Gateway", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print("\n" + "💥" * 40)
        print(f"🔥 CRITICAL BACKEND ERROR DETECTED DURING REQUEST: {request.url.path}")
        print("💥" * 40)
        traceback.print_exc()
        print("💥" * 40 + "\n")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {str(e)}", "trace": traceback.format_exc()}
        )

# NOTE: If utilizing multiple workers in the future, these MUST be moved to SQLite/Redis
session_context = {
    "current_workspace": None,
    "chat_history": []
}

MAX_HISTORY_TURNS = 10

def trim_chat_history(history: list) -> list:
    """Keep only the last N turns to prevent memory explosion in long conversations."""
    if len(history) > MAX_HISTORY_TURNS:
        return history[-MAX_HISTORY_TURNS:]
    return history


class IngestRequest(BaseModel):
    repo_url: str


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    raise HTTPException(
        status_code=404, detail="Frontend index.html file not found inside root path.")


@app.post("/api/ingest")
async def ingest_repository(payload: IngestRequest):
    if workspace_manager is None:
        raise HTTPException(
            status_code=503, detail="Server workspace manager is still initializing.")
    if not payload.repo_url.strip():
        raise HTTPException(
            status_code=400, detail="Provided repository URL is empty.")
    try:
        loop = asyncio.get_event_loop()
        workspace_path = await loop.run_in_executor(
            None, workspace_manager.process_github_repository, payload.repo_url
        )
        session_context["current_workspace"] = workspace_path
        return {"status": "success", "workspace_path": workspace_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def process_user_query(payload: ChatRequest):
    if mcp_router is None:
        raise HTTPException(
            status_code=503, detail="The AI engine failed initialization on startup.")
    if not payload.message.strip():
        raise HTTPException(
            status_code=400, detail="Query message cannot be blank.")

    session_context["chat_history"].append(HumanMessage(content=payload.message))

    trimmed_history = trim_chat_history(session_context["chat_history"])

    initial_graph_state = cast(AgentState, {
        "messages": trimmed_history,
        "focused_file_path": session_context["current_workspace"] or os.getcwd(),
        "mcp_session": mcp_router.session,
        "mcp_tools_cache": mcp_router.mcp_tools
    })

    try:
        final_state = await mcp_router.run_pipeline(initial_graph_state)
        agent_reply = final_state["messages"][-1].content
        session_context["chat_history"].append(AIMessage(content=agent_reply))
        return {"response": agent_reply}
    except Exception as e:
        root_exception = e
        while hasattr(root_exception, "__cause__") and root_exception.__cause__:
            root_exception = root_exception.__cause__

        error_message = f"[{type(root_exception).__name__}]: {str(root_exception)}"

        print("\n" + "❌" * 40)
        print(f"🚨 ABSOLUTE ROOT PIPELINE CRASH EXPOSED: {error_message}")
        print("❌" * 40)
        traceback.print_exc()
        print("❌" * 40 + "\n")

        raise HTTPException(status_code=500, detail=f"Pipeline error: {error_message}")


if __name__ == "__main__":
    import uvicorn
    import importlib.util 
    
    target_port = int(os.getenv("PORT", 8000))
    
    # 💡 PRODUCTION TUNING: Render free tiers allocate 1 CPU core. Setting workers to 1 
    # eliminates multi-process state drifting completely and preserves clean in-memory state.
    workers_count = int(os.getenv("WEB_CONCURRENCY", 1))
    
    # Check if the module is inspectable without natively running an 'import'
    if importlib.util.find_spec("uvloop") is not None:
        chosen_loop = "uvloop"
        chosen_http = "httptools"
        print("⚡ Using high-performance engine configurations (uvloop + httptools)")
    else:
        chosen_loop = "asyncio"
        chosen_http = "auto"
        print("🐢 Falling back to standard engine configurations (asyncio)")

    print(f"⚙️ Booting RepoIntel Gateway Engine on interface 0.0.0.0:{target_port} with {workers_count} worker(s)...")
    
    uvicorn.run(
        "server:app",           
        host="0.0.0.0",         
        port=target_port,       
        workers=workers_count,  
        loop=chosen_loop,       
        http=chosen_http,       
        access_log=False,       
        log_level="warning"     
    )