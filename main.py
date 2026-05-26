# main.py
import os
import asyncio
from src.ingestion import LocalASTEngine
from src.vector_store import LanceIndexingVault
from src.graph import mcp_router
from src.evaluation import assess_agent_health
from langchain_core.messages import HumanMessage


async def execute_system_diagnostic():
    print("=" * 70)
    print("🧪 RepoIntel v2: Automated Integration Diagnostic Pipeline")
    print("=" * 70)

    # 1. Setup predictable local sandbox environment
    mock_file_name = "diagnostic_calculator.py"
    mock_code_content = """# Automated Sandbox Sample Code
def compute_square_root(value: float) -> float:
    \"\"\"Calculates square roots for positive floats safely.\"\"\"
    if value < 0:
        raise ValueError("Cannot calculate square root of a negative number.")
    return value ** 0.5
"""
    print(
        f"\n✨ Phase 1: Materializing Sandbox Environment at '{mock_file_name}'...")
    with open(mock_file_name, "w", encoding="utf-8") as f:
        f.write(mock_code_content)

    try:
        # 2. Extract structural syntax elements via AST
        print("✨ Phase 2: Evaluating Syntax Architecture via Native AST Engine...")
        payload = LocalASTEngine.extract_structures(
            mock_file_name, mock_code_content)

        # 3. Commit elements into LanceDB
        print("✨ Phase 3: Synchronizing Extracted Payloads into Local LanceDB Store...")
        vault = LanceIndexingVault()
        vault.index_file_payload(payload)

        # 4. Fire up the LangGraph + MCP Server processing loop
        print("✨ Phase 4: Spin-up MCP Server Subprocess & Routing Query through Graph...")
        diagnostic_query = "Analyze diagnostic_calculator.py and tell me what function is inside."

        initial_state = {
            "messages": [HumanMessage(content=diagnostic_query)],
            "focused_file_path": os.getcwd()  # Sets the active project context path
        }

        final_state = await mcp_router.run_pipeline(initial_state)
        agent_output = final_state["messages"][-1].content

        print("\n🤖 Diagnostic Agent Response Output:")
        print("-" * 50)
        print(agent_output)
        print("-" * 50)

        # 5. Run the newly fixed Ragas Evaluation Layer
        print("\n✨ Phase 5: Executing Guardrail Benchmarking via Refactored Ragas Loop...")
        retrieved_contexts = vault.semantic_code_search(
            diagnostic_query, limit=1)

        health_report = assess_agent_health(
            diagnostic_query, agent_output, retrieved_contexts)
        print(f"\n📋 Integration Pipeline Health Metrics Summary:")
        print(f"   ➔ Status: {health_report['status']}")
        print(
            f"   ➔ Faithfulness Score: {health_report['faithfulness_score']:.2f}")
        print(
            f"   ➔ Answer Relevance Score: {health_report['response_relevance_score']:.2f}")

    finally:
        # 6. Housekeeping cleanup
        print("\n✨ Phase 6: Cleaning up generated sandbox environments...")
        if os.path.exists(mock_file_name):
            os.remove(mock_file_name)
        print("✅ Diagnostic testing completed successfully.")

if __name__ == "__main__":
    asyncio.run(execute_system_diagnostic())
