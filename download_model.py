from huggingface_hub import hf_hub_download
import shutil
import os

# Configuration (Same as app.py)
MODEL_REPO = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

print(f"Downloading {MODEL_FILE} from {MODEL_REPO}...")
try:
    # Download to cache first
    cached_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"Cached at: {cached_path}")

    # Copy to current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    destination = os.path.join(current_dir, MODEL_FILE)
    
    print(f"Copying to {destination}...")
    shutil.copy(cached_path, destination)
    
    print("-" * 50)
    print(f"SUCCESS! Model saved to: {destination}")
    print("-" * 50)
    print("WARNING: This file is likely larger than 100MB.")
    print("To push this to GitHub, you MUST use Git LFS (Large File Storage):")
    print("  1. Run: git lfs install")
    print("  2. Run: git lfs track \"*.gguf\"")
    print("  3. Run: git add .gitattributes")
    print("  4. Then add and commit your model file.")
    print("-" * 50)

except Exception as e:
    print(f"Error: {e}")
