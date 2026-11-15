"""Explicit, testable configuration precedence. Never log returned credentials."""
import os


def resolve_api_key(secrets=None, environ=None):
    env = os.environ if environ is None else environ
    for source, values in (("environment", env), ("secrets", secrets or {})):
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if name in values:
                value = values[name]
                # An explicitly invalid higher-priority value must not select another key.
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{source}:{name} is empty or invalid. Correct or remove it before using Gemini.")
                return value.strip(), f"{source}:{name}"
    raise ValueError("Set GEMINI_API_KEY in your environment or Streamlit secrets before using Gemini.")


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value
