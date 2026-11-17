"""Opt-in live end-to-end test. Uses provider quota and removes its own test data."""
from pathlib import Path
import os
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pymupdf
from workspace.settings import load_environment, integer
from workspace.database import Repository, users, projects, messages
from workspace.service import WorkspaceService


def main():
    if '--run' not in sys.argv:
        print('Use --run to test a synthetic PDF through the real services. This uses API quota.')
        return 1
    load_environment()
    repo = Repository()
    repo.initialize()
    service = WorkspaceService(repo)
    user = repo.user('live-smoke-' + uuid.uuid4().hex, 'smoke@example.invalid', 'Temporary connection test')
    project_id = repo.create_project(user['id'], 'Temporary live smoke test',
        os.getenv('OPENAI_EMBED_MODEL', 'text-embedding-3-small'), integer('OPENAI_EMBED_DIMENSIONS',1536))
    passed = False
    cleaned = False
    try:
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((60,60), 'Orion Study\nMethods: Researchers used attention layers.\n'
                'Results: The Orion method achieved 92 percent accuracy.\n'
                'Limitations: Only a small dataset was evaluated.')
            data = doc.tobytes()
        file_id = service.upload(user['id'], project_id, 'paperlens-live-test.pdf', data)
        print('Real PDF upload, R2 storage and Pinecone indexing: OK', flush=True)
        payload = None
        for attempt in range(5):
            payload = service.ask(user['id'], project_id, 'What accuracy did the Orion method achieve?', generate=True)
            if payload['hits']:
                break
            time.sleep(2)
        print('Pinecone retrieval: ' + ('OK' if payload['hits'] else 'FAILED'), flush=True)
        print('Answer mode: ' + payload['result']['mode'], flush=True)
        passed = bool(payload['hits']) and payload['result']['mode'] == 'generated' and bool(payload['result']['claims'])
        if passed:
            print('Generated claim: ' + payload['result']['claims'][0]['text'], flush=True)
        print('Saved messages: ' + str(len(repo.history(user['id'], project_id))), flush=True)
        print('Source link: ' + ('OK' if service.source_url(user['id'],project_id,file_id).startswith('https://') else 'FAILED'), flush=True)
        summary = service.summarize(user['id'], project_id, file_id)
        print('Summary mode: ' + summary['mode'], flush=True)
        passed = passed and summary['mode'] == 'generated'
    except ValueError as exc:
        print('Live flow failed: ' + str(exc), flush=True)
    except Exception:
        print('Live flow failed. Check service configuration.', flush=True)
    finally:
        cleaned = True
        for record in repo.list_files(user['id'], project_id):
            try:
                service.remove_file(user['id'], project_id, record['id'])
            except Exception:
                cleaned = False
        if cleaned:
            with repo.engine.begin() as conn:
                conn.execute(messages.delete().where(messages.c.project_id == project_id))
                conn.execute(projects.delete().where(projects.c.id == project_id, projects.c.user_id == user['id']))
                conn.execute(users.delete().where(users.c.id == user['id']))
        print('Temporary data cleanup: ' + ('OK' if cleaned else 'NEEDS RETRY'), flush=True)
        repo.engine.dispose()
    return 0 if passed and cleaned else 1


if __name__ == '__main__':
    raise SystemExit(main())
