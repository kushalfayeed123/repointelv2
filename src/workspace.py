# src/workspace.py
import os
import stat
import shutil
import tempfile
from git import Repo
from src.ingestion import UniversalASTEngine
from src.vector_store import LanceIndexingVault


class SystemWorkspaceManager:
    """Clones, extracts, and vectorizes code components from Git endpoints or local paths."""

    def __init__(self):
        self.vault = LanceIndexingVault()

    def process_and_index_directory(self, root_dir: str):
        """Recursively parses all source code files under a given directory root path."""
        print(f"📁 Ingesting structures from: {root_dir}")
        
        # =====================================================================
        # FIX: Drop the existing database table index so old embeddings are discarded!
        # =====================================================================
        table_name = "codebase_index"
        if table_name in self.vault.db.table_names():
            print("🧹 Flushing old vector store database cache...")
            self.vault.db.drop_table(table_name)

        indexed_count = 0

        # Change your loop to parse polyglot code files based on extensions mapped in your engine
        supported_extensions = (".py", ".js", ".ts", ".go", ".rs", ".cpp", ".java")

        for root, _, files in os.walk(root_dir):
            for file in files:
                # FIX: Stop restricting parsing to only .py files if ingestion engine handles any source code
                if file.lower().endswith(supported_extensions):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, root_dir)

                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            source_code = f.read()

                        payload = UniversalASTEngine.extract_structures(
                            relative_path, source_code)
                        self.vault.index_file_payload(payload)
                        indexed_count += 1
                    except Exception as e:
                        print(
                            f"⚠️ Failed to parse structural tree for {relative_path}: {e}")

        print(
            f"✅ Ingestion cycle closed. Indexed {indexed_count} source modules successfully.")

    @staticmethod
    def _remove_readonly(func, path, excinfo):
        """Clear the read-only bit on Windows so shutil.rmtree can clean up git caches."""
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def process_github_repository(self, repo_url: str):
        """Clones a public Git repository into a local cache folder to allow the agent to make edits."""
        local_workspace = os.path.join(os.getcwd(), "workspace_cache")
        if os.path.exists(local_workspace):
            shutil.rmtree(local_workspace, onexc=self._remove_readonly)

        print(
            f"🐙 Cloning remote repository {repo_url} into local workspace target: {local_workspace}")
        Repo.clone_from(repo_url, local_workspace, depth=1)
        self.process_and_index_directory(local_workspace)
        return local_workspace