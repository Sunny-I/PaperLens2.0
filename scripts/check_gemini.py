"""List available chat and embedding models without printing credentials or raw provider errors."""
import os
import sys
import tomllib
from pathlib import Path
from google import genai

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from pipeline.configuration import resolve_api_key
secrets_path = root / '.streamlit/secrets.toml'
secrets = tomllib.loads(secrets_path.read_text()) if secrets_path.exists() else {}
try:
    key, _ = resolve_api_key(secrets)
except ValueError as exc:
    print(str(exc))
    sys.exit(1)
try:
    with genai.Client(api_key=key, vertexai=False, http_options={'timeout': 20000}) as client:
        available = list(client.models.list())
        models = [m.name for m in available if 'generateContent' in (m.supported_actions or [])]
        embeddings = [m.name for m in available if 'embedContent' in (m.supported_actions or [])]
    print('Available chat models:', ', '.join(models))
    print('Available embedding models:', ', '.join(embeddings))
except Exception as exc:
    print('Connection check failed:', type(exc).__name__, 'HTTP code:', getattr(exc, 'code', None))
    sys.exit(1)
