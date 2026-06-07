# src/mcp_server_sse.py
import os
import subprocess
from fastmcp import FastMCP

# Initialize the official FastMCP orchestration frame
mcp = FastMCP("RepoIntelRuntimeServer")

# ----------------------------------------------------------------------
# Register Your Tool Definitions Cleanly
# ----------------------------------------------------------------------

@mcp.tool()
def read_local_file(path: str) -> str:
    """Reads and returns the complete text content of a local file safely."""
    workspace_path = os.path.join(os.getcwd(), "workspace_cache", os.path.basename(path))
    if not os.path.exists(workspace_path):
        return f"Error: Target file reference at '{path}' does not exist inside sandbox."
    with open(workspace_path, "r", encoding="utf-8") as f:
        return f.read()
    
@mcp.tool()
def update_local_file(path: str, modifications: str) -> str:
    """Overwrites or outputs structural code changes directly to a target local path."""
    workspace_path = os.path.join(os.getcwd(), "workspace_cache", os.path.basename(path))
    os.makedirs(os.path.dirname(workspace_path), exist_ok=True)
    with open(workspace_path, "w", encoding="utf-8") as f:
        f.write(modifications)
    return f"Success: File systems sync completed cleanly for {path}."

@mcp.tool()
def execute_test_suite(test_target_path: str) -> str:
    """Runs a local test suite via pytest to verify your changes."""
    workspace_path = os.path.join(os.getcwd(), "workspace_cache", os.path.basename(test_target_path))
    try:
        result = subprocess.run(
            ["pytest", workspace_path],
            capture_output=True, text=True, timeout=30
        )
        return f"STDOUT: \n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as err:
        return f"Execution Fault Encountered: {str(err)}"

# 💡 THE NATIVE PRODUCTION WAY: Bind explicitly to your target port and loopback host
if __name__ == "__main__":
    # 1. Read Render's assigned port dynamically, fallback to 5001 for local development
    target_port = int(os.getenv("PORT", 5001))
    
    # 2. Bind to 0.0.0.0 to expose the service to Render's routing mesh
    # Local development still works seamlessly over http://127.0.0.1:5001
    mcp.run(
        transport="sse", 
        host="0.0.0.0", 
        port=target_port
    )