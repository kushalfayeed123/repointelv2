# src/ingestion.py
import os
import textwrap
from typing import List, Optional
from pydantic import BaseModel, Field
from tree_sitter import Parser
import tree_sitter_languages

class FunctionUnit(BaseModel):
    name: str = Field(description="Identifier of the function or method component.")
    source_code: str = Field(description="The exact text snippet of the code block.")
    language: str = Field(description="The source language classification (e.g., python, javascript).")

class ClassUnit(BaseModel):
    name: str = Field(description="Identifier of the defined structural class/object model.")
    source_code: str = Field(description="The entire structural text block enclosing the class.")
    language: str = Field(description="The programming language classification.")

class CodebasePayload(BaseModel):
    file_path: str = Field(description="Relative path destination of the parsed source module.")
    top_level_imports: List[str] = Field(default_factory=list, description="Extracted external references or imports.")
    classes: List[ClassUnit] = Field(default_factory=list)
    standalone_functions: List[FunctionUnit] = Field(default_factory=list)

class LocalASTEngine:
    """Uses tree-sitter queries to extract universal programming symbols across multiple targets."""

    EXTENSION_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".java": "java"
    }

    # Universal Tree-Sitter queries mapping syntax nodes to tags
    LANGUAGE_QUERIES = {
        "python": {
            "functions": "(function_def name: (identifier) @func_name) @func_node",
            "classes": "(class_def name: (identifier) @class_name) @class_node"
        },
        "javascript": {
            "functions": [
                "(function_declaration name: (identifier) @func_name) @func_node",
                "(method_definition name: (property_identifier) @func_name) @func_node"
            ],
            "classes": "(class_declaration name: (identifier) @class_name) @class_node"
        },
        "go": {
            "functions": [
                "(function_declaration name: (identifier) @func_name) @func_node",
                "(method_declaration name: (field_identifier) @func_name) @func_node"
            ],
            "classes": "(type_spec name: (identifier) @class_name type: (struct_type)) @class_node"
        },
        "rust": {
            "functions": "(function_item name: (identifier) @func_name) @func_node",
            "classes": "(struct_item name: (type_identifier) @class_name) @class_node"
        }
    }

    @classmethod
    def extract_structures(cls, file_path: str, source_text: str) -> CodebasePayload:
        ext = os.path.splitext(file_path)[1].lower()
        lang_id = cls.EXTENSION_MAP.get(ext)

        # Fallback handling for plain-text, configuration profiles, or unrecognized targets
        if not lang_id or lang_id not in cls.LANGUAGE_QUERIES:
            return cls._generate_fallback_payload(file_path, source_text)

        # Normalize formatting layout constraints safely
        source_text = textwrap.dedent(source_text).strip()
        source_text = source_text.replace("\r\n", "\n")
        source_bytes = bytes(source_text, "utf8")

        parser = Parser()
        try:
            language_binding = tree_sitter_languages.get_language(lang_id)
            parser.set_language(language_binding)
        except Exception:
            return cls._generate_fallback_payload(file_path, source_text)

        tree = parser.parse(source_bytes)
        root_node = tree.root_node

        extracted_functions = []
        extracted_classes = []
        queries = cls.LANGUAGE_QUERIES[lang_id]

        # 1. Process Multi-Language Functions via capture map groupings
        func_queries = queries["functions"] if isinstance(queries["functions"], list) else [queries["functions"]]
        for q_str in func_queries:
            query = language_binding.query(q_str)
            captures = query.captures(root_node)
            
            node_map = {}
            for node, tag in captures:
                if tag not in node_map:
                    node_map[tag] = []
                node_map[tag].append(node)

            if "func_node" in node_map and "func_name" in node_map:
                for f_node, f_name_node in zip(node_map["func_node"], node_map["func_name"]):
                    code_segment = source_bytes[f_node.start_byte:f_node.end_byte].decode("utf-8")
                    name_str = source_bytes[f_name_node.start_byte:f_name_node.end_byte].decode("utf-8")
                    
                    extracted_functions.append(FunctionUnit(
                        name=name_str,
                        source_code=code_segment,
                        language=lang_id
                    ))

        # 2. Process Multi-Language Classes
        class_queries = queries["classes"] if isinstance(queries["classes"], list) else [queries["classes"]]
        for q_str in class_queries:
            query = language_binding.query(q_str)
            captures = query.captures(root_node)
            
            node_map = {}
            for node, tag in captures:
                if tag not in node_map:
                    node_map[tag] = []
                node_map[tag].append(node)

            if "class_node" in node_map and "class_name" in node_map:
                for c_node, c_name_node in zip(node_map["class_node"], node_map["class_name"]):
                    code_segment = source_bytes[c_node.start_byte:c_node.end_byte].decode("utf-8")
                    name_str = source_bytes[c_name_node.start_byte:c_name_node.end_byte].decode("utf-8")
                    
                    extracted_classes.append(ClassUnit(
                        name=name_str,
                        source_code=code_segment,
                        language=lang_id
                    ))

        # If it's a known code extension but yielded no blocks, wrap the file context cleanly
        if not extracted_functions and not extracted_classes and source_text.strip():
            return cls._generate_fallback_payload(file_path, source_text, lang_id)

        return CodebasePayload(
            file_path=file_path,
            classes=extracted_classes,
            standalone_functions=extracted_functions
        )

    @staticmethod
    def _generate_fallback_payload(file_path: str, source_text: str, lang: str = "plaintext") -> CodebasePayload:
        """Fallback framework encapsulating text streams gracefully."""
        return CodebasePayload(
            file_path=file_path,
            standalone_functions=[
                FunctionUnit(
                    name="full_module",
                    source_code=source_text,
                    language=lang
                )
            ]
        )

# Ensure our alias pointer is active for workspace.py structural targets
UniversalASTEngine = LocalASTEngine