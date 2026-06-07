#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Ensure uv is accessible in the PATH environment if running in loose shells
export PATH="/bin:/usr/local/bin:$PATH"

echo "🚀 Initializing RepoIntel Multi-Service Launcher..."
echo "📊 SERVICE_TYPE evaluated as: '${SERVICE_TYPE}'"

# Check the environment variable injected by the cloud provider
if [ "$SERVICE_TYPE" = "mcp" ]; then
    echo "🛰️ Target match found: Launching Decoupled MCP Tool Server on port ${PORT:-5001}..."
    exec uv run src/mcp_server_sse.py
else
    echo "🧠 Target fallback: Launching Gateway Orchestrator API on port ${PORT:-8000}..."
    exec uv run server.py
fi