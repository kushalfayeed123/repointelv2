# interface.py
import asyncio
import os
import sys
import shutil
from langchain_core.messages import HumanMessage
from src.workspace import SystemWorkspaceManager
from src.graph import mcp_router
from src.vector_store import LanceIndexingVault
from src.evaluation import assess_agent_health

async def run_main_application_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 72)
    print("⚙️  RepoIntel v2: Production MCP Workspace Control Center")
    print("=" * 72)
    
    manager = SystemWorkspaceManager()
    focused_workspace_path = os.getcwd()
    
    print("\n[1] Index a Local Project Directory Path")
    print("[2] Ingest & Clone a Public GitHub Repository URL")
    print("[3] Provide raw Code Snippet string directly")
    print("[4] Skip indexing phase (Use existing database schema index)")
    
    selection = input("\nSelect an entry ingestion strategy (1-4): ").strip()
    
    if selection == "1":
        path = input("📂 Enter absolute or relative path to local directory: ").strip()
        if not os.path.isdir(path):
            print("❌ Invalid directory reference. Exiting.")
            sys.exit(1)
        manager.process_and_index_directory(path)
        focused_workspace_path = os.path.abspath(path)
        
    elif selection == "2":
        repo_url = input("🐙 Enter target public GitHub Repository URL: ").strip()
        if not repo_url.startswith("http"):
            print("❌ Invalid source endpoint URL format. Exiting.")
            sys.exit(1)
        # Clones directly to an accessible local directory so edits write back correctly
        focused_workspace_path = manager.process_github_repository(repo_url)
        
    elif selection == "3":
        print("📝 Enter your raw Python code (Type 'EOF' on a new line when finished):")
        lines = []
        while True:
            line = input()
            if line.strip() == "EOF":
                break
            lines.append(line)
        raw_code = "\n".join(lines)
        
        snippet_dir = os.path.join(os.getcwd(), "snippet_workspace")
        os.makedirs(snippet_dir,存在_ok=True)
        focused_workspace_path = snippet_dir
        
        target_file = os.path.join(snippet_dir, "code_snippet.py")
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(raw_code)
            
        from src.ingestion import LocalASTEngine
        payload = LocalASTEngine.extract_structures("code_snippet.py", raw_code)
        manager.vault.index_file_payload(payload)
        print(f"✅ Code snippet initialized at: {target_file}")
        
    elif selection == "4":
        print("🛸 Accessing local database indexing boundaries...")
        focused_workspace_path = input("Enter working directory root path to verify: ").strip()
    else:
        print("❌ Unknown option selection choice.")
        sys.exit(1)

    print("\n" + "=" * 72)
    print("🤖 Agent Core Active Loop: Ask questions or request code changes.")
    print("💡 Enter 'exit' or 'quit' to close connection streams cleanly.")
    print("=" * 72 + "\n")
    
    conversation_messages = []
    vault = LanceIndexingVault()
    
    while True:
        try:
            user_prompt = input("👤 User ➔ ").strip()
            if not user_prompt:
                continue
            if user_prompt.lower() in ["exit", "quit"]:
                # Clean up any temporary workspace clones on exit if needed
                print("\n👋 Closing MCP sessions. Goodbye!")
                break
                
            conversation_messages.append(HumanMessage(content=user_prompt))
            print("🤖 RepoIntel is compiling contextual graph paths...")
            
            runtime_input = {
                "messages": conversation_messages,
                "focused_file_path": focused_workspace_path
            }
            
            final_state = await mcp_router.run_pipeline(runtime_input)
            
            agent_message = final_state["messages"][-1]
            agent_response = agent_message.content
            conversation_messages = list(final_state["messages"])
            
            print(f"\n🤖 RepoIntel v2 ➔\n{agent_response}\n")
            
            print("📋 Calculating real-time Ragas alignment validations...")
            contexts = vault.semantic_code_search(user_prompt, limit=2)
            
            if contexts:
                report = assess_agent_health(user_prompt, agent_response, contexts)
                status_color = "🟢" if report["status"] == "PASS" else "🔴"
                print(f"   [{status_color} Ragas Health Check] Faithfulness: {report['faithfulness_score']:.2f} | Relevance: {report['response_relevance_score']:.2f}\n")
            else:
                print("   [条 Check Skip] No matching database context captured for score parsing.\n")
                
        except KeyboardInterrupt:
            print("\n👋 Session closed cleanly.")
            break
        except Exception as err:
            print(f"❌ Core Error: {err}\n")

if __name__ == "__main__":
    asyncio.run(run_main_application_terminal())