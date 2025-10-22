import ollama
from typing import Generator

# ------------------------
# Check Ollama status
# ------------------------
def is_ollama_running() -> bool:
    """
    Check if Ollama is running by trying to list models.
    Returns True if successful, False otherwise.
    """
    try:
        ollama.list()
        return True
    except Exception:
        return False

# ------------------------
# Get available models
# ------------------------
def get_llms() -> list[str]:
    """
    Get available models from Ollama.
    Returns a list of model names.
    """
    try:
        models = ollama.list()
        if "models" in models and models["models"]:
            return [m.model for m in models["models"]]
        else:
            return []
    except Exception:
        return []

# ------------------------
# Chat with selected model
# ------------------------
def chat_with_model(model: str, prompt: str) -> Generator[str, None, None]:
    """
    Send a user prompt to the selected Ollama model and return the response.
    Yields partial text chunks as they arrive.
    """
    try:
        # response = ollama.chat(
        #     model=model,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response["message"]["content"]
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in stream:
            # Each chunk is usually a dict with 'message' content
            content = chunk.get("message", {}).get("content")
            if content:
                yield content
    except Exception as e:
        yield f"❌ Error: {e}"