"""OpenAI Responses API with typed output, citation checks, and bounded repair."""
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError
from pydantic import SecretStr, ValidationError

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


def resolve_openai_key(secrets=None, environ=None):
    import os
    env = os.environ if environ is None else environ
    for source, values in (("environment", env), ("secrets", secrets or {})):
        if "OPENAI_API_KEY" in values:
            key = values["OPENAI_API_KEY"]
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{source}:OPENAI_API_KEY is empty or invalid. Correct or remove it before using OpenAI.")
            return key.strip()
    raise ValueError("Set OPENAI_API_KEY in your environment or Streamlit secrets before using OpenAI.")


def safe_error(exc):
    if isinstance(exc, APITimeoutError):
        return "OpenAI request timed out. Retry later or select another available chat model."
    if isinstance(exc, APIConnectionError):
        return "Could not connect to OpenAI. Check your internet connection and retry."
    code = getattr(exc, "status_code", None)
    return {
        401: "OpenAI rejected the API key. Check OPENAI_API_KEY.",
        403: "OpenAI denied access. Check your project and model permissions.",
        404: "OpenAI model was not found or is unavailable to your project. Check the chat model ID.",
        429: "OpenAI quota or rate limit reached. Check API billing and limits, then retry later.",
        400: "OpenAI rejected the request. Check model support for structured output and input limits.",
    }.get(code, "OpenAI service request failed. Retry later.")


class OpenAIClient:
    def __init__(self, api_key, timeout=120):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Set OPENAI_API_KEY in your environment or Streamlit secrets before using OpenAI.")
        self.api_key = SecretStr(api_key.strip())
        self.timeout = timeout

    def structured(self, model, messages, schema, validator=None):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Enter a nonempty OpenAI chat model ID.")
        conversation = list(messages)
        try:
            with OpenAI(api_key=self.api_key.get_secret_value(), timeout=self.timeout,
                        max_retries=1) as client:
                for attempt in range(2):
                    try:
                        response = client.responses.parse(model=model.strip(), input=conversation,
                            text_format=schema, store=False)
                        if response.status != "completed":
                            raise ResponseUnavailable("OpenAI returned an incomplete response. Try a smaller request or another model.")
                        if any(getattr(part, "type", None) == "refusal"
                               for item in response.output for part in getattr(item, "content", [])):
                            raise ResponseUnavailable("OpenAI declined this request. Showing available source excerpts instead.")
                        if response.output_parsed is None:
                            raise ValueError("Missing structured output")
                        result = schema.model_validate(response.output_parsed)
                        if validator:
                            validator(result)
                        return result
                    except ResponseUnavailable:
                        raise
                    except (ValidationError, ValueError, TypeError):
                        if attempt:
                            raise ValueError("OpenAI output failed schema or citation validation twice. Showing source excerpts instead.") from None
                        conversation.append({"role": "user", "content":
                            "Regenerate the complete JSON with all required fields, valid source IDs/pages "
                            "and exact supporting quotes. Use null or [] for unsupported information."})
        except APIError as exc:
            raise ValueError(safe_error(exc)) from None
        except ValidationError:
            raise ValueError("OpenAI returned invalid structured output. Showing source excerpts instead.") from None


class ResponseUnavailable(ValueError):
    """Incomplete or refused requests should fall back without a repair call."""
