# src/vector_store.py
import lancedb
from lancedb.pydantic import LanceModel, Vector
from src.ingestion import CodebasePayload

class CodeSnippetSchema(LanceModel):
    id: str
    file_path: str
    type: str          
    name: str
    meta_tags: str     
    source_code: str
    vector: Vector(384)

class LanceIndexingVault:
    """Manages transactional indexing and semantic search over a localDB instance."""

    def __init__(self, db_path: str = "db/repointel_lance"):
        self.db = lancedb.connect(db_path)
        self._encoder = None  # ◄ Deferred/Lazy initialization pointer

    @property
    def encoder(self):
        """Loads the embedding engine context ONLY when an embedding is explicitly requested."""
        if self._encoder is None:
            print("🧠 Loading vector embedding model lazily into active application RAM...")
            from langchain_huggingface import HuggingFaceEmbeddings
            self._encoder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        return self._encoder

    def index_file_payload(self, payload: CodebasePayload):
        table_name = "codebase_index"
        records = []
        texts_to_embed = []
        metadata_list = []

        # OPTIMIZATION 4: Batch embedding calls - collect all texts first
        for func in payload.standalone_functions:
            tags = f"function {func.name} imports: {', '.join(payload.top_level_imports)}"
            texts_to_embed.append(func.source_code)
            metadata_list.append({
                "id": f"{payload.file_path}::{func.name}",
                "file_path": payload.file_path,
                "type": "function",
                "name": func.name,
                "meta_tags": tags,
                "source_code": func.source_code
            })

        for cls in payload.classes:
            tags = f"class {cls.name} methods: {', '.join([m.name for m in cls.methods])}"
            combined_class_text = f"class {cls.name}:\n" + "\n".join([m.source_code for m in cls.methods])
            texts_to_embed.append(combined_class_text)
            metadata_list.append({
                "id": f"{payload.file_path}::{cls.name}",
                "file_path": payload.file_path,
                "type": "class",
                "name": cls.name,
                "meta_tags": tags,
                "source_code": combined_class_text
            })

        # Batch embed all texts at once instead of individual calls
        if texts_to_embed:
            vectors = self.encoder.embed_documents(texts_to_embed)
            for metadata, vector in zip(metadata_list, vectors):
                records.append({**metadata, "vector": vector})

        if not records:
            return

        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            table.add(records)
        else:
            self.db.create_table(table_name, data=records, schema=CodeSnippetSchema)

    def semantic_code_search(self, query: str, limit: int = 3) -> list:
        table_name = "codebase_index"
        if table_name not in self.db.table_names():
            return []

        query_vector = self.encoder.embed_query(query)  # ◄ Uses lazy property accessor
        table = self.db.open_table(table_name)
        results = table.search(query_vector).limit(limit).to_list()
        return results