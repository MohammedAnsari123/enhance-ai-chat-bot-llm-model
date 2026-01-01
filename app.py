from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from llama_cpp import Llama
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
# Switching to Qwen2.5-0.5B-Instruct-GGUF as requested
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf" 

model = None
load_error = None

print(f"Downloading/Loading GGUF model from {MODEL_REPO}...")
try:
    # Explicitly download the file
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"Model downloaded to: {model_path}")

    # Load using llama-cpp-python (Llama class)
    # n_ctx=2048 context window
    # n_gpu_layers=0 for CPU only
    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0,
        verbose=True
    )
    print("Qwen Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    load_error = str(e)
    model = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

@app.get("/")
async def health_check():
    if model is None:
        return {"status": "error", "message": "Model failed to load", "detail": load_error}
    return {"status": "ok", "message": "Service is running (Qwen optimized)"}

@app.post("/chat")
async def chat(data: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not initialized. Error: {load_error}")

    try:
        # Construct simplified ChatML-like prompt for Qwen
        # Qwen uses: <|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant
        messages_list = [{"role": m.role, "content": m.content} for m in data.messages]
        
        # Llama-cpp-python has a create_chat_completion method that handles formatting automatically
        # if the model metadata is correct, otherwise we can manual prompt.
        # But Qwen chat templates are standard. Let's try the high-level API first.
        
        response = model.create_chat_completion(
            messages=messages_list,
            max_tokens=data.max_tokens,
            temperature=data.temperature,
            top_p=0.9
        )
        
        # Extract content
        generated_text = response['choices'][0]['message']['content']
        
        return { "text": generated_text }
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
