"""One minimal structured generation probe with credential-redacted diagnostics."""
import json, os, re, sys, tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from langchain_google_genai import ChatGoogleGenerativeAI
from pipeline.answer_generator import GeneratedAnswer
from pipeline.configuration import resolve_api_key
from pipeline.gemini_client import DEFAULT_CHAT_MODEL
s=tomllib.loads((ROOT/'.streamlit/secrets.toml').read_text())
key,source=resolve_api_key(s)
model=os.getenv('GEMINI_CHAT_MODEL',DEFAULT_CHAT_MODEL)
result={'model':model,'credential_source':source}
try:
    llm=ChatGoogleGenerativeAI(model=model,google_api_key=key,vertexai=False,max_retries=0,timeout=30)
    response=llm.with_structured_output(GeneratedAnswer,method='json_schema').invoke('Return a JSON object containing only an empty claims list.')
    result.update(status='PASS',claims=response.model_dump())
except Exception as exc:
    safe=str(exc).replace(key,'[REDACTED]')
    safe=re.sub(r'AIza[\w-]+','[REDACTED]',safe)
    safe=re.sub(r'https?://\S+','[URL]',safe)
    result.update(status='BLOCKED',exception_type=type(exc).__name__,redacted_error=safe[:1800])
(ROOT/'audit/generation-diagnostic.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
