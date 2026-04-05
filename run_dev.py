import subprocess
import os
import sys

def run_dev():
    print("🚀 Starting Optimized Development Server...")
    
    # Set environment variables for dev mode
    # Set DEV_MODE=true to skip heavy vector store preloading on startup
    env = os.environ.copy()
    env["DEV_MODE"] = "true"
    
    # Construct Uvicorn command with folder exclusions
    # We exclude 'data' (ChromaDB) to prevent infinite reload loops
    # We exclude 'frontend' to prevent reloads when editing the Flutter app
    command = [
        "uvicorn", "main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload-exclude", "data/*",
        "--reload-exclude", "frontend/*",
        "--reload-exclude", "myvenv/*",
        "--reload-exclude", "*.log",
    ]
    
    print(f"🔧 Command: {' '.join(command)}")
    print("💡 RAG will load on the FIRST request to save startup time.")
    
    try:
        subprocess.run(command, env=env)
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
        sys.exit(0)

if __name__ == "__main__":
    run_dev()
