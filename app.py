from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_REPO = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

print("Downloading model...")
llm = None
try:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print("Model path:", model_path)
    
    llm = Llama(
        model_path=model_path,
        n_ctx=1024,
        n_gpu_layers=0,
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

    prompt = ""
    for m in messages:
        prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    result = llm(prompt, max_tokens=200, stop=["<|im_end|>"])
    text = result["choices"][0]["text"].strip()
    return {"response": text}
