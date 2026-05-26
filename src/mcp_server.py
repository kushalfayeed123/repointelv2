import os
import subprocess
from fastmcp import FastMCP


# Initialize FactMCP Server context

mcp = FastMCP("RepoIntelRuntimeServer")

@mcp.tool()
def read_local_file(path: str) -> str:
    """Reads and returns the complete text content of a local file safely."""
    if not os.path.exists(path):
        return f"Error: Target file reference at '{path}' does not exist."
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
@mcp.tool()
def update_local_file(path: str, modifications: str) -> str:
    """Overwrites or outputs structural code changes directly to a target local path."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(modifications)
    return f"Success: File systems sync completed cleanly for {path}."

@mcp.tool()
def execute_test_suite(test_target_path:str) -> str:
    """Runs a local test suite via pytest to verify your changes."""
    try:
        result = subprocess.run(
            ["uv", "run", "pytest", test_target_path],
            capture_output=True, text=True, timeout=30
        )
        return f"STDOUT: \n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as err:
        return f"Execution Failure: {str(err)}"
    
if __name__== "__main__":
    # Launch server over standard input/output(studio) channels
    mcp.run()