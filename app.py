from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ctransformers import AutoModelForCausalLM
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
MODEL_REPO = "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF"
MODEL_FILE = "smollm2-360m-instruct-q4_k_m.gguf" 

model = None
load_error = None

print(f"Downloading/Loading GGUF model from {MODEL_REPO}...")
try:
    # Explicitly download the file first to ensure we have the local path
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print(f"Model downloaded to: {model_path}")

    # Load model with ctransformers using the local file path
    model = AutoModelForCausalLM.from_pretrained(
        model_path, # Pass exact file path
        model_type="llama", 
        context_length=2048,
        gpu_layers=0 # Force CPU
    )
    print("GGUF Model loaded successfully.")
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

def apply_chat_template(messages: List[dict]) -> str:
    """
    Manually apply ChatML template:
    <|im_start|>system
    ...<|im_end|>
    <|im_start|>user
    ...<|im_end|>
    <|im_start|>assistant
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

@app.get("/")
async def health_check():
    if model is None:
        return {"status": "error", "message": "Model failed to load", "detail": load_error}
    return {"status": "ok", "message": "Service is running (GGUF optimized)"}

@app.post("/chat")
async def chat(data: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not initialized. Error: {load_error}")

    try:
        messages_list = [{"role": m.role, "content": m.content} for m in data.messages]
        prompt = apply_chat_template(messages_list)
        
        # ctransformers generate returns a generator or text
        # simple generation:
        generated_text = model(
            prompt, 
            max_new_tokens=data.max_tokens,
            temperature=data.temperature,
            top_p=0.9
        )
        
        return { "text": generated_text }
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
