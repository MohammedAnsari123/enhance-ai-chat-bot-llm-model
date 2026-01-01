from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and tokenizer
# Use environment variable for model name if needed, or default to SmolLM2
model_name = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")

print(f"Loading model: {model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto" # Let accelerate handle device map (likely CPU on Render free tier)
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # We don't exit here so the app can still start and report health status, 
    # but actual chat requests will fail if model is None.
    model = None
    tokenizer = None

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
        return {"status": "error", "message": "Model failed to load"}
    return {"status": "ok", "message": "Service is running"}

@app.post("/chat")
async def chat(data: ChatRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        # Convert Pydantic messages to list of dicts for tokenizer
        messages_list = [{"role": m.role, "content": m.content} for m in data.messages]
        
        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=data.max_tokens,
                temperature=data.temperature,
                do_sample=True,
                top_p=0.9
            )
            
        # Decode only the new tokens
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Return in the format expected by the server (text field)
        return { "text": generated_text }
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
