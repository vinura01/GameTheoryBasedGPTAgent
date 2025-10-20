
import os, requests, json

class LLMAgent:
    def __init__(self, model: str = "mistral", system_prompt: str = "You are a helpful assistant."):
        self.model = model
        self.system_prompt = system_prompt
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def chat(self, user_text: str) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip() or "(No content returned)"
        except Exception as e:
            return f"[LLM error] {e}"
