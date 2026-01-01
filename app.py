from fastapi import FastAPI
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

app = FastAPI()

MODEL_REPO = "qnguyen3/flan-t5-small-gguf"
MODEL_FILE = "flan-t5-small-q4_0.gguf"

print("Downloading model...")
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
print("Model downloaded:", model_path)

llm = Llama(
    model_path=model_path,
    n_threads=4,
    n_ctx=1024
)

@app.get("/")
def home():
    return {"message": "LLM Chatbot is Running"}

@app.post("/chat")
def chat(request: dict):
    prompt = request.get("prompt", "")
    if not prompt:
        return {"error": "Prompt missing"}
    output = llm(f"User: {prompt}\nAssistant:", max_tokens=128, temperature=0.7)
    text = output["choices"][0]["text"]
    return {"response": text.strip()}
