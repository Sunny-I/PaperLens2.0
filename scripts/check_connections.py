"""Read-only external connectivity checks. Prints statuses, never credential values."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import text
from openai import OpenAI
from workspace.settings import load_environment, missing_configuration, required, integer
from workspace.database import Repository
from workspace.providers import R2Storage, PineconeVectors


def main():
    load_environment()
    missing = missing_configuration()
    if missing:
        print('Missing configuration: ' + ', '.join(missing))
        return 1
    failed = False
    checks = [
        ('PostgreSQL', lambda: database_check()),
        ('R2 bucket', lambda: R2Storage().check()),
        ('Pinecone dimensions', lambda: PineconeVectors().check(integer('OPENAI_EMBED_DIMENSIONS', 1536))),
        ('OpenAI model access', lambda: openai_check()),
    ]
    for name, action in checks:
        try:
            action()
            print(name + ': OK')
        except Exception:
            print(name + ': FAILED (check configuration, permissions and connectivity)')
            failed = True
    return int(failed)


def database_check():
    repo = Repository()
    try:
        with repo.engine.connect() as conn:
            conn.execute(text('SELECT 1'))
    finally:
        repo.engine.dispose()


def openai_check():
    with OpenAI(api_key=required('OPENAI_API_KEY'), timeout=20, max_retries=0) as client:
        client.models.retrieve(required('OPENAI_CHAT_MODEL'))
        client.models.retrieve(required('OPENAI_EMBED_MODEL'))


if __name__ == '__main__':
    raise SystemExit(main())
