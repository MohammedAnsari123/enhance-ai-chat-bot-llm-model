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
MODEL_FILE = "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

print("Loading model...")
llm = None
try:
    # Check if model exists locally (pushed via Git LFS)
    if os.path.exists(MODEL_FILE):
        model_path = MODEL_FILE
        print(f"Found local model: {model_path}")
    else:
        print(f"Local model not found. Downloading from {MODEL_REPO}...")
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        print("Model downloaded to cache:", model_path)
    
    llm = Llama(
        model_path=model_path,
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
