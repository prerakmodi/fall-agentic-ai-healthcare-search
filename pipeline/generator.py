import requests
from pathlib import Path
from typing import List, Dict

PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(name: str) -> str:
    prompt_path = PROMPTS_DIR / f"{name}.md"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text().strip()

# An instance of interacting with the Ollama LLM
class OllamaGenerator:
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 1024
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.system_prompt = load_prompt("system")
        self.rag_template = load_prompt("rag_template")
        self.context_template = load_prompt("context_template")

    # format retrieved documents into context string
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            url_line = f"URL: {doc.get('url')}" if doc.get('url') else ""

            formatted = self.context_template.format(
                index=i,
                title=doc.get('title', 'Unknown'),
                source_type=doc.get('source', 'Unknown'),
                url_line=url_line,
                content=doc.get('text', '')
            )
            context_parts.append(formatted)

        return "\n".join(context_parts)

    # build the prompt with context and query
    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        context = self.format_context(retrieved_docs)

        return self.rag_template.format(
            context=context,
            query=query
        )

    # have ollama generate a response
    def generate(
        self,
        query: str,
        retrieved_docs: List[Dict],
        stream: bool = False
    ) -> str:
        prompt = self.build_prompt(query, retrieved_docs)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except requests.exceptions.RequestException as e:
            return f"error communicating with Ollama: {str(e)}"

    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                return self.model.split(":")[0] in model_names
            return False
        except:
            return False

# Have LLM generate simple content without RAG context
def simple_generate(query: str, model: str = "llama3.2") -> str:
    generator = OllamaGenerator(model=model)

    if not generator.check_health():
        return "error: ollama is not running or model not found. Run 'ollama serve' first."

    payload = {
        "model": model,
        "prompt": query,
        "system": generator.system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512
        }
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"Error: {str(e)}"
