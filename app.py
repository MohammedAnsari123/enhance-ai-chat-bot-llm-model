from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
except Exception as e:
    model = None
    load_error = str(e)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    if model is None:
        return {"status": "error", "detail": load_error}
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503)

    inputs = tokenizer(req.message, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.8
    )

    return {"reply": tokenizer.decode(outputs[0], skip_special_tokens=True)}
