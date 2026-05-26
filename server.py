# server.py
import os
import asyncio
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse  # ◄ Added for serving index.html
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from src.workspace import SystemWorkspaceManager
from src.graph import mcp_router

app = FastAPI(title="RepoIntel API Gateway", version="2.0.0")

# Keep CORS broad so local testing environments don't conflict
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# NEW: Global Exception Middleware to print error logs to terminal
# =====================================================================


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        # Print the massive, detailed stack trace directly into your command line
        print("\n" + "💥" * 40)
        print(
            f"🔥 CRITICAL BACKEND ERROR DETECTED DURING REQUEST: {request.url.path}")
        print("💥" * 40)
        traceback.print_exc()  # This prints the file name, line number, and error name
        print("💥" * 40 + "\n")

        # Format the exception message cleanly as JSON for your frontend script
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal Server Error: {str(e)}", "trace": traceback.format_exc()}
        )

workspace_manager = SystemWorkspaceManager()

session_context = {
    "current_workspace": None,
    "chat_history": []
}


class IngestRequest(BaseModel):
    repo_url: str


class ChatRequest(BaseModel):
    message: str

# =====================================================================
# NEW: Serve your index.html directly at the root URL (/)
# =====================================================================


@app.get("/")
async def serve_frontend():
    """Serves the vanilla HTML interface straight from the project folder."""
    # Looks for index.html in the directory where server.py is running
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    raise HTTPException(
        status_code=404, detail="Frontend index.html file not found inside root path.")


@app.post("/api/ingest")
async def ingest_repository(payload: IngestRequest):
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
        raise HTTPException(
            status_code=500, detail=f"Pipeline error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 so external cloud infrastructure environments can route ports smoothly
    uvicorn.run("server.py", host="0.0.0.0", port=8000, reload=True)
