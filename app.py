from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ctransformers import AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware
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
# Use a GGUF quantized model repo.
# SmolLM2 360M Instruct GGUF. We use q8_0 or q4_k_m. q8_0 is better quality, q4 is smaller.
# Given 512MB RAM, we MUST use q4_k_m or similar.
MODEL_REPO = "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF"
MODEL_FILE = "smollm2-360m-instruct-q4_k_m.gguf" 

print(f"Loading GGUF model from {MODEL_REPO}...")
try:
    # Load model with ctransformers
    # gpu_layers=0 ensures CPU only.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        model_file=MODEL_FILE,
        model_type="llama", 
        context_length=2048,
    )
    print("GGUF Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
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
        return {"status": "error", "message": "Model failed to load"}
    return {"status": "ok", "message": "Service is running (GGUF optimized)"}

@app.post("/chat")
async def chat(data: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

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
