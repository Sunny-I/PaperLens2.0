"""LangChain prompt/model chains with Pydantic parsing and bounded repair."""
import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ollama import ResponseError
from pydantic import ValidationError


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", timeout=120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def structured(self, model, messages, schema, validator=None):
        """Return a Pydantic model; retry malformed or ungrounded output once."""
        llm = ChatOllama(model=model, base_url=self.base_url, temperature=0,
                         num_ctx=16384, num_predict=4096,
                         client_kwargs={"timeout": self.timeout})
        prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder("messages")])
        chain = prompt | llm.with_structured_output(schema, method="json_schema")
        conversation = list(messages)
        for attempt in range(2):
            try:
                result = schema.model_validate(chain.invoke({"messages": conversation}))
                if validator:
                    validator(result)
                return result
            except (ValidationError, OutputParserException, ValueError, TypeError) as exc:
                if attempt:
                    raise ValueError("The model did not return valid, source-backed structured output after two attempts.") from exc
                conversation.append({"role": "user", "content": (
                    "The previous output failed validation. Regenerate the complete JSON using only "
                    "the supplied evidence, required fields, and exact quotes. Missing information must "
                    "be null or an empty list. Validation error: " + str(exc)[:600])})
            except (httpx.HTTPError, ResponseError, ConnectionError, OSError) as exc:
                raise ValueError("Ollama request failed. Check the server URL, installed model, and structured-output support.") from exc

    def embed(self, texts, model):
        embedder = OllamaEmbeddings(model=model, base_url=self.base_url,
                                    client_kwargs={"timeout": self.timeout})
        vectors = []
        try:
            for offset in range(0, len(texts), 32):
                batch = texts[offset:offset + 32]
                result = embedder.embed_documents(batch)
                if len(result) != len(batch):
                    raise ValueError("Ollama returned an invalid embedding batch.")
                vectors.extend(result)
            return vectors
        except (httpx.HTTPError, ResponseError, ConnectionError, OSError) as exc:
            raise ValueError("Ollama embeddings failed. Check the server URL and embedding model.") from exc
