from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_REPO = "bartowski/SmolLM-135M-Instruct-GGUF"
MODEL_FILE = "smollm-135m-instruct-Q4_K_M.gguf"

print("Downloading and loading model...")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

model = Llama(
    model_path=model_path,
    n_threads=2,  # Render free tier = 2 CPU
    n_ctx=1024,
    n_gpu_layers=0
)

class ChatReq(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatReq):
    response = model(
        req.message,
        max_tokens=120,
        temperature=0.7,
        stop=["</s>"]
    )
    return {"reply": response["choices"][0]["text"]}

@app.get("/")
def root():
    return {"status": "ok"}
