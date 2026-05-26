# tests/test_pipeline.py
import pytest
import textwrap
from src.ingestion import LocalASTEngine

def test_ast_engine_handles_async_methods():
    """Verifies that the AST engine correctly identifies and processes async function nodes."""
    async_code_sample = """
    async def fetch_remote_telemetry(node_hash: str) -> dict:
        \"\"\"Simulates remote network call synchronization bindings.\"\"\"
        return {"node_hash": node_hash, "latency": 45}
    """
    # Clean up outer indents so the code block starts cleanly at column 0 for ast.parse
    cleaned_code = textwrap.dedent(async_code_sample).strip()
    
    payload = LocalASTEngine.extract_structures("test_async.py", cleaned_code)
    
    assert len(payload.standalone_functions) == 1
    target_function = payload.standalone_functions[0]
    assert target_function.name == "fetch_remote_telemetry"
    assert target_function.is_async is True
    assert "node_hash" in target_function.args
    assert "Simulates remote network" in target_function.docstring


def test_ast_engine_handles_classes_and_imports():
    """Verifies that imports and class architectures are cleanly isolated."""
    code_sample = """
    from datetime import datetime
    import os

    class TelemetryProcessor:
        \"\"\"Processes backend metrics.\"\"\"
        def process(self, payload):
            return True
    """
    cleaned_code = textwrap.dedent(code_sample).strip()
    
    payload = LocalASTEngine.extract_structures("processor.py", cleaned_code)
    
    # Assert Imports
    assert "datetime.datetime" in payload.top_level_imports
    assert "os" in payload.top_level_imports
    
    # Assert Classes and inner methods
    assert len(payload.classes) == 1
    target_class = payload.classes[0]
    assert target_class.name == "TelemetryProcessor"
    assert target_class.docstring == "Processes backend metrics."
    
    assert len(target_class.methods) == 1
    assert target_class.methods[0].name == "process"
    assert "self" in target_class.methods[0].args