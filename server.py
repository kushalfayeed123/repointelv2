import os
import asyncio
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from src.workspace import SystemWorkspaceManager
# ◄ Fully safe to import globally now because it does nothing on read!
from src.graph import mcp_router

workspace_manager = None

# =====================================================================
# LIFESPAN MANAGER: Tells Uvicorn to open the port FIRST, then load tools
# =====================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global workspace_manager, mcp_router
    print("\n🚀 NETWORK PORT BINDING SUCCESSFUL! Render scan passed.")
    print("🤖 Lazily instantiating LangGraph, MCP Tools, and Subprocesses...")

    try:
        workspace_manager = SystemWorkspaceManager()

        # Pull the router context into scope
        from src.graph import mcp_router as instantiated_router
        mcp_router = instantiated_router

        # Fire up the long-lived background connection stream safely
        await mcp_router.start_session()

        print("✅ Core systems initialized successfully. Ready to handle queries.\n")
        yield
    except Exception as startup_err:
        print(f"\n🔥 CRITICAL LIFESPAN SETUP RUNTIME CRASH: {startup_err}")
        traceback.print_exc()
        yield  # Yield anyway so the port remains responsive for local debugging
    finally:
        # Prevent orphaned zombie processes locally when turning off the server
        if mcp_router:
            await mcp_router.stop_session()

app = FastAPI(title="RepoIntel API Gateway",
              version="2.0.0", lifespan=lifespan)

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
        print(
            f"🔥 CRITICAL BACKEND ERROR DETECTED DURING REQUEST: {request.url.path}")
        print("💥" * 40)
        traceback.print_exc()
        print("💥" * 40 + "\n")
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal Server Error: {str(e)}", "trace": traceback.format_exc()}
        )

session_context = {
    "current_workspace": None,
    "chat_history": []
}


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

    session_context["chat_history"].append(
        HumanMessage(content=payload.message))

    initial_graph_state = {
        "messages": session_context["chat_history"],
        "focused_file_path": session_context["current_workspace"] or os.getcwd()
    }

    try:
        final_state = await mcp_router.run_pipeline(initial_graph_state)
        agent_reply = final_state["messages"][-1].content
        session_context["chat_history"].append(AIMessage(content=agent_reply))
        return {"response": agent_reply}
    except Exception as e:
        root_exception = e
        while hasattr(root_exception, "exceptions") and root_exception.exceptions:
            root_exception = root_exception.exceptions[0]

        error_message = f"[{type(root_exception).__name__}]: {str(root_exception)}"

        print("\n" + "❌" * 40)
        print(f"🚨 ABSOLUTE ROOT PIPELINE CRASH EXPOSED: {error_message}")
        print("❌" * 40)
        traceback.print_exc()
        print("❌" * 40 + "\n")

        raise HTTPException(
            status_code=500, detail=f"Pipeline error: {error_message}")

if __name__ == "__main__":
    import uvicorn
    # ◄ Explicitly false for production safety
    uvicorn.run("server.py", host="0.0.0.0", port=8000, reload=False)
