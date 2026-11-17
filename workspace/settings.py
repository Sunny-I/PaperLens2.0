"""Server-only configuration; secrets are never returned to the UI."""
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_environment():
    load_dotenv(ROOT / '.env', override=False)


def required(name):
    value = os.getenv(name, '').strip()
    if not value:
        raise ValueError(f'Configure {name} in the server .env file, then restart PaperLens.')
    return value


def integer(name, default):
    try:
        value = int(os.getenv(name, str(default)))
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        raise ValueError(f'{name} must be a positive integer.') from None


def database_url():
    value = required('DATABASE_URL')
    if value.startswith('postgres://'):
        value = 'postgresql://' + value[len('postgres://'):]
    if value.startswith('postgresql://'):
        value = 'postgresql+psycopg://' + value[len('postgresql://'):]
    if not value.startswith('postgresql+psycopg://'):
        raise ValueError('DATABASE_URL must be a PostgreSQL connection URL.')
    return value


def missing_configuration():
    names = ['DATABASE_URL', 'OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_HOST',
             'R2_ENDPOINT_URL', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_BUCKET_NAME',
             'GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET', 'AUTH_COOKIE_SECRET']
    return [name for name in names if not os.getenv(name, '').strip()
            or 'USER:PASSWORD@HOST' in os.getenv(name, '')
            or 'ACCOUNT_ID.' in os.getenv(name, '')]


def https_endpoint(value, service):
    if '://' not in value:
        value = 'https://' + value
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
        raise ValueError(f'{service} endpoint must be a valid HTTPS host URL.')
    return value.rstrip('/')
