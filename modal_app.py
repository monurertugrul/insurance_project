import modal
import os
import subprocess
import time
from modal_image import base_image


image = (
    base_image
    .env({"PYTHONPATH": "/root/app", "REBUILD": "2"})
    .add_local_dir("UI", "/root/app/UI")
    .add_local_dir("core", "/root/app/core")
    .add_local_dir("agents", "/root/app/agents")
    .add_local_dir(
        "/Users/mustafaonurertugrul/Documents/Projects/insurance_project/rag", 
        "/root/app/rag"
    )
    .add_local_file("modal_app.py", "/root/app/modal_app.py")
    .add_local_file("modal_image.py", "/root/app/modal_image.py")
)

app = modal.App("Medical-Insurance-Demo")

# ---------------------------------------------------------
# STREAMLIT SERVER 
# ---------------------------------------------------------
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("gemini-secret")],
    timeout=600
)
@modal.web_server(port=8000, startup_timeout=120)
def run():
    import subprocess
    subprocess.Popen(
        [
            "streamlit",
            "run",
            "/root/app/UI/app.py",
            "--server.port=8000",
            "--server.address=0.0.0.0",
            "--server.headless=true",
        ]
    )
    
    print("🚀 Launching Streamlit...")

# ---------------------------------------------------------
# THE ONLY ADDITION FOR RUN MODE
# ---------------------------------------------------------
@app.local_entrypoint()
def main():
    # Calling .local() on a web_server function triggers the deployment
    run.local()
    
    # We MUST keep the local process alive, or Modal kills the remote container 
    # the moment this script finishes.
    print("✅ App is running. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")