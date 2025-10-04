import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

def generate(prompt: str, max_tokens: int = 512, temperature: float = 0.0, **kwargs) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Set the env var or update this wrapper.")

    payload = {
        "model": "llama-3.1-8b-instant",  # Updated model name
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(GROQ_ENDPOINT, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        return str(data)
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"