from fastapi import FastAPI
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

MODEL_REPO = "qnguyen3/flan-t5-small-gguf"
MODEL_FILE = "flan-t5-small-q4_0.gguf"

try:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print("Model downloaded")

    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_threads=1,
        n_gpu_layers=0,
        verbose=False
    )
    print("Model loaded successfully")

except Exception as e:
    print("Model load error:", e)
    llm = None


@app.get("/")
def home():
    if llm is None:
        return {"status": "error", "message": "Model failed to load"}
    return {"status": "ok", "message": "LLM Chatbot Running"}


@app.post("/chat")
def chat(request: dict):
    if llm is None:
        return {"error": "Model not initialized"}

    prompt = request.get("prompt")
    if not prompt:
        return {"error": "Prompt missing"}

    output = llm(
        prompt,
        max_tokens=96,
        temperature=0.7
    )

    result = output["choices"][0]["text"].strip()
    return {"response": result}
