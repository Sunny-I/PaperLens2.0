"""Persistent workspace regressions; external services are explicitly mocked."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pymupdf
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from workspace.database import Repository
from workspace.providers import PineconeVectors, OpenAIEmbeddings, R2Storage
from workspace.service import WorkspaceService, SearchQuestion
from pipeline.answer_generator import GeneratedAnswer


def pdf(text='Study Alpha\nMethods: We used attention layers.\nResults: Accuracy was 92 percent.'):
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((60, 60), text)
        return doc.tobytes()


class FakeEmbeddings:
    calls = []
    def __init__(self, model, dimensions):
        self.model = model
    def embed(self, texts):
        self.calls.append(texts)
        return [[0.4, 0.8] for _ in texts]


class FakeStorage:
    def __init__(self):
        self.objects = {}
    def put(self, key, data):
        self.objects[key] = data
    def delete(self, key):
        self.objects.pop(key, None)
    def url(self, key, page=None):
        return 'https://private.example/' + key + f'#page={page}'


class FakeVectors:
    def __init__(self):
        self.namespaces = {}
        self.calls = []
    def check(self, dimensions):
        pass
    def upsert(self, namespace, chunks, vectors, user_id, project_id, file_id):
        self.namespaces.setdefault(namespace, {}).update({c['chunk_id']: dict(c, file_id=file_id) for c in chunks})
    def query(self, namespace, vector, user_id, project_id, ready_ids, top_k=18):
        self.calls.append((namespace, user_id, project_id, ready_ids))
        return [{'id': c['chunk_id'], 'score': 0.9, 'metadata': {'text': 'Untrusted external metadata'}}
                for c in self.namespaces.get(namespace, {}).values() if c['file_id'] in ready_ids]
    def delete_file(self, namespace, file_id):
        rows = self.namespaces.get(namespace, {})
        for key in [k for k, c in rows.items() if c['file_id'] == file_id]:
            del rows[key]


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
        self.repo = Repository(self.engine)
        self.repo.initialize()
        self.alice = self.repo.user('alice', 'alice@example.test', 'Alice')['id']
        self.bob = self.repo.user('bob', 'bob@example.test', 'Bob')['id']
        self.project = self.repo.create_project(self.alice, 'Research', 'frozen-model', 2)
        self.storage, self.vectors = FakeStorage(), FakeVectors()
        FakeEmbeddings.calls = []
        self.service = WorkspaceService(self.repo, self.storage, self.vectors, FakeEmbeddings)
        self.addCleanup(self.engine.dispose)

    def upload(self):
        return self.service.upload(self.alice, self.project, 'study.pdf', pdf())

    def test_reload_uses_saved_library_and_only_embeds_new_question(self):
        file_id = self.upload()
        before = len(FakeEmbeddings.calls)
        repo2 = Repository(self.engine)
        service2 = WorkspaceService(repo2, self.storage, self.vectors, FakeEmbeddings)
        answer = service2.ask(self.alice, self.project, 'What accuracy?', generate=False)
        self.assertEqual(len(FakeEmbeddings.calls), before + 1)
        self.assertEqual(answer['hits'][0]['doc_id'], file_id)
        self.assertIn('92 percent', answer['hits'][0]['text'])
        self.assertNotIn('Untrusted external metadata', answer['hits'][0]['text'])
        self.assertEqual(len(repo2.history(self.alice, self.project)), 2)
        self.assertEqual(repo2.list_files(self.alice, self.project)[0]['status'], 'ready')

    def test_other_user_cannot_read_write_search_or_open_sources(self):
        file_id = self.upload()
        actions = [lambda: self.repo.project(self.bob, self.project),
                   lambda: self.repo.list_files(self.bob, self.project),
                   lambda: self.repo.history(self.bob, self.project),
                   lambda: self.repo.reserve_file(self.bob, self.project, 'evil.pdf', 'abc', 10),
                   lambda: self.service.ask(self.bob, self.project, 'Read it', generate=False),
                   lambda: self.service.source_url(self.bob, self.project, file_id),
                   lambda: self.service.remove_file(self.bob, self.project, file_id),
                   lambda: self.repo.update_file(self.bob, self.project, file_id, status='ready')]
        for action in actions:
            with self.assertRaisesRegex(ValueError, 'access denied'):
                action()
        self.assertEqual(self.vectors.calls, [])
        self.assertEqual(len(self.repo.list_projects(self.bob)), 0)

    def test_projects_and_selected_files_are_isolated(self):
        first = self.upload()
        second_project = self.repo.create_project(self.alice, 'Other project', 'frozen-model', 2)
        self.service.upload(self.alice, second_project, 'other.pdf', pdf('An entirely different topic.'))
        result = self.service.ask(self.alice, self.project, 'Methods?', generate=False)
        self.assertEqual({h['doc_id'] for h in result['hits']}, {first})
        with self.assertRaises(ValueError):
            self.service.ask(self.alice, second_project, 'Methods?', selected_ids=[first], generate=False)

    def test_duplicate_does_not_reembed_and_invalid_pdf_does_not_write(self):
        data = pdf()
        self.service.upload(self.alice, self.project, 'study.pdf', data)
        calls = len(FakeEmbeddings.calls)
        with self.assertRaisesRegex(ValueError, 'already'):
            self.service.upload(self.alice, self.project, 'renamed.pdf', data)
        self.assertEqual(len(FakeEmbeddings.calls), calls)
        with self.assertRaises(ValueError):
            self.service.upload(self.alice, self.project, 'bad.pdf', b'not pdf')
        self.assertEqual(len(self.repo.list_files(self.alice, self.project)), 1)

    def test_partial_upsert_is_excluded_and_can_be_removed(self):
        original = self.vectors.upsert
        def partial(*args):
            original(*args)
            raise ValueError('provider credential should never escape')
        self.vectors.upsert = partial
        with self.assertRaisesRegex(ValueError, 'marked failed') as caught:
            self.upload()
        self.assertNotIn('credential', str(caught.exception))
        record = self.repo.list_files(self.alice, self.project)[0]
        self.assertEqual(record['status'], 'failed')
        with self.assertRaisesRegex(ValueError, 'at least one'):
            self.service.ask(self.alice, self.project, 'Findings?', generate=False)
        self.service.remove_file(self.alice, self.project, record['id'])
        self.assertEqual(self.repo.list_files(self.alice, self.project), [])
        self.assertEqual(self.storage.objects, {})

    def test_failed_deletion_is_retryable_and_sources_unavailable(self):
        file_id = self.upload()
        original = self.storage.delete
        self.storage.delete = Mock(side_effect=ValueError('R2 cleanup failed'))
        with self.assertRaises(ValueError):
            self.service.remove_file(self.alice, self.project, file_id)
        self.assertEqual(self.repo.file(self.alice, self.project, file_id)['status'], 'deleting')
        with self.assertRaises(ValueError):
            self.service.source_url(self.alice, self.project, file_id)
        self.storage.delete = original
        self.service.remove_file(self.alice, self.project, file_id)
        self.assertEqual(self.repo.list_files(self.alice, self.project), [])

    def test_grounded_answer_saves_quotes_and_resolves_followups(self):
        self.upload()
        quote = 'Accuracy was 92 percent.'
        chat = Mock()
        def structured(model, messages, schema, validator=None):
            if schema is SearchQuestion:
                return SearchQuestion(question='Explain the accuracy in Study Alpha')
            answer = schema.model_validate({'claims': [{'text': 'Accuracy was 92 percent.',
                'citations': [{'source_id': 1, 'evidence_id': 1}]}]})
            if validator:
                validator(answer)
            return answer
        chat.structured.side_effect = structured
        self.service.chat = chat
        self.service.ask(self.alice, self.project, 'Accuracy?')
        answer = self.service.ask(self.alice, self.project, 'Explain that more')
        self.assertEqual(answer['retrieval_question'], 'Explain the accuracy in Study Alpha')
        history = self.repo.history(self.alice, self.project)
        self.assertEqual(len(history), 4)
        self.assertIn(quote, history[-1]['payload']['result']['claims'][0]['citations'][0]['quote'])

    def test_records_survive_engine_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            url = 'sqlite:///' + str(Path(folder) / 'test.db')
            engine = create_engine(url)
            repo = Repository(engine)
            repo.initialize()
            user = repo.user('persistent', 'p@example.test', 'P')['id']
            project = repo.create_project(user, 'Persistent', 'model', 2)
            engine.dispose()
            engine2 = create_engine(url)
            try:
                self.assertEqual(Repository(engine2).project(user, project)['name'], 'Persistent')
            finally:
                engine2.dispose()


class ProviderTests(unittest.TestCase):
    def setUp(self):
        setting = patch.dict('os.environ', {'PINECONE_EMBED_FIELD': ''})
        setting.start()
        self.addCleanup(setting.stop)

    def test_document_index_uses_integrated_embeddings_and_owner_filters(self):
        with patch.dict('os.environ', {'PINECONE_EMBED_FIELD': 'chunk_text',
                                      'PINECONE_EMBED_MODEL': 'text-embedding-3-small'}):
            adapter = PineconeVectors('index.example.test', 'fake')
        adapter._documents = Mock(return_value=SimpleNamespace(matches=[SimpleNamespace(id='chunk',score=0.9)]))
        chunk = {'chunk_id':'chunk','text':'Source passage','filename':'paper.pdf','page':1}
        adapter.upsert('ns',[chunk],None,'owner','project','file')
        document = adapter._documents.call_args.kwargs['documents'][0]
        self.assertEqual(document['chunk_text'], 'Source passage')
        self.assertEqual(document['user_id'], 'owner')
        result = adapter.query('ns',None,'owner','project',['file'],query_text='Question?')
        self.assertEqual(result,[{'id':'chunk','score':0.9}])
        filters = adapter._documents.call_args.kwargs['filter']['$and']
        self.assertIn({'file_id': {'$in': ['file']}}, filters)
        adapter.delete_file('ns','file')
        self.assertEqual(adapter._documents.call_args.args[0],'delete')

    def test_pinecone_request_has_namespace_and_all_ownership_filters(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={'matches': []})
        adapter = PineconeVectors('index.example.test', 'fake', httpx.MockTransport(handler))
        adapter.query('project-123', [0.1, 0.2], 'owner', '123', ['file'])
        data = requests[0]
        self.assertEqual(data['namespace'], 'project-123')
        self.assertEqual(data['filter']['$and'], [{'user_id': {'$eq': 'owner'}},
            {'project_id': {'$eq': '123'}}, {'file_id': {'$in': ['file']}}])

    def test_provider_error_does_not_expose_response_or_api_key(self):
        adapter = PineconeVectors('index.example.test', 'secret-value',
            httpx.MockTransport(lambda request: httpx.Response(401, text='secret-value')))
        with self.assertRaises(ValueError) as caught:
            adapter.query('ns', [1], 'owner', 'project', ['file'])
        self.assertNotIn('secret-value', str(caught.exception))

    def test_openai_embedding_batches_and_preserves_order(self):
        client = Mock()
        client.embeddings.create.side_effect = lambda **kw: SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[float(i+1), 0.5]) for i in reversed(range(len(kw['input'])))])
        adapter = OpenAIEmbeddings('model', 2, client=client)
        vectors = adapter.embed(['Text'] * 33)
        self.assertEqual(len(vectors), 33)
        self.assertEqual(vectors[0], [1.0, 0.5])
        self.assertEqual(client.embeddings.create.call_count, 2)
        self.assertEqual(client.embeddings.create.call_args.kwargs['dimensions'], 2)

    def test_r2_private_link_has_bounded_lifetime(self):
        client = Mock()
        client.generate_presigned_url.return_value = 'https://private.example/pdf?signature=temporary'
        storage = R2Storage(client=client, bucket='private')
        self.assertTrue(storage.url('owner/project/file.pdf', 5).endswith('#page=5'))
        self.assertEqual(client.generate_presigned_url.call_args.kwargs['ExpiresIn'], 300)


if __name__ == '__main__':
    unittest.main()
