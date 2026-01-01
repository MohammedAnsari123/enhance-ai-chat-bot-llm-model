from fastapi import FastAPI
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

app = FastAPI()

# Configuration
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

print(f"Downloading/Loading GGUF model from {MODEL_REPO}...")
try:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"Model downloaded to: {model_path}")

    # Use n_gpu_layers=0 for CPU inference
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0, 
        verbose=True
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    llm = None

@app.get("/")
def home():
    if llm is None:
        return {"status": "error", "message": "Model failed to load"}
    return {"status": "ok", "message": "Service is running (Qwen optimized)"}

@app.post("/chat")
def chat(request: dict):
    if llm is None:
        return {"error": "Model not initialized"}
        
    messages = request.get("messages", [])
    if not messages:
        return {"error": "Messages missing"}
    
    # Simple ChatML formatting for Qwen
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    output = llm(
        prompt, 
        max_tokens=512, 
        stop=["<|im_end|>"], 
        temperature=0.7
    )
    
    text = output["choices"][0]["text"]
    return { "text": text.strip() }
