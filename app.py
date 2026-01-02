from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_REPO = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILENAME = "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
# Get absolute path to ensure we find the file regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

print(f"Checking for model at: {MODEL_PATH}")
llm = None
try:
    # Check if model exists locally
    if os.path.exists(MODEL_PATH):
        print(f"✅ Found local model: {MODEL_PATH}")
        model_location = MODEL_PATH
    else:
        print(f"⚠️ Local model not found at {MODEL_PATH}. Downloading from {MODEL_REPO}...")
        model_location = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        print("Model downloaded to cache:", model_location)
    
    llm = Llama(
        model_path=model_location,
        n_ctx=2048,       # Increased context window
        n_gpu_layers=0,   # CPU only
        n_batch=512,      # Optimizes prompt processing speed
        n_threads=4,      # Uses multi-threading for faster generation
        verbose=False
    )
    print("Model loaded successfully")
except Exception as e:
    print("Model load failed:", e)


@app.get("/")
def root():
    return {"status": "running", "model_loaded": llm is not None}


@app.post("/chat")
async def chat(request: Request):
    if llm is None:
        return {"error": "Model not initialized"}

    body = await request.json()
    messages = body.get("messages")
    if not messages:
        return {"error": "messages missing"}

    messages = body.get("messages")
    if not messages:
        return {"error": "messages missing"}

    # Use native chat completion which handles templating correctly and efficiently
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=200,
        stop=["<|im_end|>", "<|endoftext|>"],
        temperature=0.7
    )
    
    return {"response": response["choices"][0]["message"]["content"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
