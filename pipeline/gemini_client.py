"""Gemini Developer API integration with validated, cited structured output."""
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import SecretStr, ValidationError


DEFAULT_CHAT_MODEL = "gemini-3.8-flash"


def validate_chat_model(model):
    model = model.strip().removeprefix("models/")
    if not model:
        raise ValueError("Enter a Gemini chat model, such as gemini-3.8-flash.")
    if "embedding" in model.lower():
        raise ValueError("An embedding model cannot generate answers or summaries. "
                         "Change Chat model to gemini-3.8-flash. "
                         "Embedding models belong in the separate Embedding model field.")
    return model


def provider_error(exc, operation="generation"):
    """Classify errors using private details, but return only fixed, safe messages."""
    errors = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        errors.append(exc)
        exc = exc.__cause__ or exc.__context__
    details = " ".join(str(e).lower() for e in errors)
    codes = {str(getattr(e, "code", "")) for e in errors}
    if "503" in codes or "high demand" in details or "503 unavailable" in details:
        return "Gemini is temporarily overloaded (503). Retry later or choose another available chat model; changing your API key will not resolve this service error."
    if "429" in codes or "resource_exhausted" in details or "quota" in details:
        return "Gemini quota or rate limit reached. Check your Google AI Studio quota/billing and retry later."
    if codes & {"401", "403"} or "api_key_invalid" in details or "api key not valid" in details:
        return "Gemini rejected the credentials or permissions. Check the API key and model access in Google AI Studio."
    if "404" in codes or "not_found" in details or "not found" in details:
        return "Gemini model was not found or is unavailable to this key. Check the selected model ID."
    if any(isinstance(e, TimeoutError) or "timeout" in type(e).__name__.lower() for e in errors):
        return "Gemini request timed out. Retry later; no alternative model was selected."
    if any(isinstance(e, (ConnectionError, OSError)) or "connect" in type(e).__name__.lower()
           or "timeout" in type(e).__name__.lower() for e in errors):
        return "Could not connect to Gemini. Check your internet connection and retry."
    if "400" in codes or "invalid_argument" in details:
        if operation == "embedding":
            return "Gemini rejected the embedding request (400). Check the embedding model, input size and task support."
        return "Gemini rejected the request. Check that the selected model supports structured JSON output."
    return f"Gemini {operation} failed unexpectedly. Check model access and retry."


class GeminiClient:
    def __init__(self, api_key, timeout=120):
        if not api_key or not api_key.strip():
            raise ValueError("Set GEMINI_API_KEY in your environment or Streamlit secrets before using Gemini.")
        self.api_key = SecretStr(api_key.strip())
        self.timeout = timeout

    def structured(self, model, messages, schema, validator=None):
        model = validate_chat_model(model)
        try:
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=self.api_key,
                vertexai=False, temperature=0, timeout=self.timeout, max_retries=1)
            prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder("messages")])
            chain = prompt | llm.with_structured_output(schema, method="json_schema")
            conversation = list(messages)
            for attempt in range(2):
                try:
                    result = schema.model_validate(chain.invoke({"messages": conversation}))
                    if validator:
                        validator(result)
                    return result
                except (ValidationError, OutputParserException, ValueError, TypeError):
                    if attempt:
                        raise StructuredOutputError("Gemini returned output that failed schema or citation validation twice. "
                                                    "Try a more specific question or another chat model.")
                    conversation.append({"role": "user", "content":
                        "Regenerate the complete JSON with all required fields, valid source IDs/pages "
                        "and exact supporting quotes. Use null or [] for unsupported information."})
        except StructuredOutputError:
            raise
        except Exception as exc:
            # Provider exceptions can contain request URLs/credentials: never display them.
            raise ValueError(provider_error(exc)) from None

    def _embeddings(self, model):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Enter a nonempty Gemini embedding model ID.")
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=self.api_key,
            vertexai=False, client_args={"http_options": {"timeout": self.timeout * 1000}})

    def embed(self, texts, model):
        batch_start, batch_end = 0, 0
        try:
            embedder = self._embeddings(model)
            result = []
            for offset in range(0, len(texts), 32):
                batch = texts[offset:offset + 32]
                batch_start, batch_end = offset + 1, offset + len(batch)
                vectors = embedder.embed_documents(batch)
                if len(vectors) != len(batch):
                    raise ValueError("Invalid embedding count")
                result.extend(vectors)
            return result
        except Exception as exc:
            position = f" Text chunks {batch_start}–{batch_end} of {len(texts)} failed; index was not updated." if batch_start else ""
            raise ValueError(provider_error(exc, "embedding") + position) from None

    def embed_query(self, query, model):
        try:
            return self._embeddings(model).embed_query(query)
        except Exception as exc:
            raise ValueError(provider_error(exc, "embedding")) from None


class StructuredOutputError(ValueError):
    """A schema/citation failure, distinct from authentication or transport errors."""


class ProviderEmbeddings:
    """Freeze the index's provider/model and distinguish documents from queries."""
    def __init__(self, client, model):
        self.client, self.model = client, model

    def __call__(self, texts):
        return self.client.embed(texts, self.model)

    def embed_query(self, query):
        if hasattr(self.client, "embed_query"):
            return self.client.embed_query(query, self.model)
        return self.client.embed([query], self.model)[0]
