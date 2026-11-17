"""Bounded provider adapters with sanitized failures and explicit configuration."""
import math
import os
import boto3
import httpx
import tiktoken
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from openai import OpenAI, APIError

from pipeline.openai_client import safe_error
from workspace.settings import required, https_endpoint


class OpenAIEmbeddings:
    def __init__(self, model, dimensions, client=None):
        self.model, self.dimensions = model, dimensions
        self.client = client
        self.encoding = tiktoken.get_encoding('cl100k_base')

    def embed(self, texts):
        if not texts:
            return []
        if any(not text.strip() or len(self.encoding.encode(text, disallowed_special=())) > 8000
               for text in texts):
            raise ValueError('A text chunk exceeds the embedding input limit. Try a simpler PDF.')
        vectors = []
        owned = self.client is None
        client = self.client or OpenAI(api_key=required('OPENAI_API_KEY'), timeout=90, max_retries=1)
        try:
            for start in range(0, len(texts), 32):
                batch = texts[start:start + 32]
                response = client.embeddings.create(model=self.model, dimensions=self.dimensions, input=batch)
                data = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in data] != list(range(len(batch))):
                    raise ValueError('Embedding provider returned an incomplete batch.')
                for item in data:
                    vector = item.embedding
                    if len(vector) != self.dimensions or not all(math.isfinite(x) for x in vector) or not any(vector):
                        raise ValueError('Embedding provider returned an invalid vector.')
                    vectors.append(vector)
            return vectors
        except APIError as exc:
            raise ValueError(safe_error(exc)) from None
        finally:
            if owned:
                client.close()


class R2Storage:
    def __init__(self, client=None, bucket=None):
        self.bucket = bucket or required('R2_BUCKET_NAME')
        self.client = client or boto3.client('s3',
            endpoint_url=https_endpoint(required('R2_ENDPOINT_URL'), 'R2'),
            aws_access_key_id=required('R2_ACCESS_KEY_ID'),
            aws_secret_access_key=required('R2_SECRET_ACCESS_KEY'), region_name='auto',
            config=Config(signature_version='s3v4', connect_timeout=10, read_timeout=60,
                          retries={'max_attempts': 2}, request_checksum_calculation='when_required',
                          response_checksum_validation='when_required'))

    def put(self, key, data):
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType='application/pdf')
        except (BotoCoreError, ClientError):
            raise ValueError('R2 upload failed. Check bucket access and server configuration.') from None

    def delete(self, key):
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except (BotoCoreError, ClientError):
            raise ValueError('R2 file cleanup failed. Retry removing the file.') from None

    def url(self, key, page=None):
        try:
            url = self.client.generate_presigned_url('get_object',
                Params={'Bucket': self.bucket, 'Key': key}, ExpiresIn=300)
            return url + (f'#page={int(page)}' if page else '')
        except (BotoCoreError, ClientError):
            raise ValueError('Could not prepare the PDF link. Check R2 configuration.') from None

    def check(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except (BotoCoreError, ClientError):
            raise ValueError('R2 bucket access failed.') from None


class PineconeVectors:
    """Pinecone vector REST API; namespaces always come from owned DB projects."""
    def __init__(self, host=None, api_key=None, transport=None):
        self.host = https_endpoint(host or required('PINECONE_INDEX_HOST'), 'Pinecone')
        self.api_key = api_key or required('PINECONE_API_KEY')
        self.transport = transport
        self.embedding_field = os.getenv('PINECONE_EMBED_FIELD', '').strip()
        self.embeds_documents = bool(self.embedding_field)

    def _documents(self, operation, **kwargs):
        from pinecone import Pinecone, NotFoundError
        client = Pinecone(api_key=self.api_key, timeout=60)
        try:
            index = client.index(host=self.host)
            return getattr(index.documents, operation)(**kwargs)
        except NotFoundError:
            if operation == 'delete':
                return None
            raise ValueError('Pinecone namespace is not available yet. Retry in a moment.') from None
        except Exception:
            raise ValueError('Pinecone document operation failed. Check embedding model access, index schema, and quota.') from None
        finally:
            client.close()

    def validate_model(self, model):
        if self.embeds_documents and model != required('PINECONE_EMBED_MODEL'):
            raise ValueError('This project uses a different embedding model from the configured Pinecone index.')

    def _request(self, path, payload=None):
        try:
            with httpx.Client(timeout=60, transport=self.transport) as client:
                headers = {'Api-Key': self.api_key,
                    'X-Pinecone-API-Version': os.getenv('PINECONE_API_VERSION', '2026-07')}
                response = (client.get(self.host + path, headers=headers) if payload is None
                    else client.post(self.host + path, headers=headers, json=payload))
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError):
            raise ValueError('Pinecone request failed. Check index host, API key, dimensions, and quota.') from None

    def check(self, dimensions):
        stats = self._request('/describe_index_stats', {})
        if int(stats.get('dimension', 0)) != dimensions:
            raise ValueError(f'Pinecone index must have {dimensions} dimensions for this project.')

    def upsert(self, namespace, chunks, vectors, user_id, project_id, file_id):
        if self.embeds_documents:
            for start in range(0, len(chunks), 16):
                documents = [{'_id': c['chunk_id'], self.embedding_field: c['text'],
                    'title': c['filename'], 'user_id': user_id, 'project_id': project_id,
                    'file_id': file_id, 'filename': c['filename'], 'page': c['page']}
                    for c in chunks[start:start+16]]
                self._documents('upsert', namespace=namespace, documents=documents)
            return
        if len(chunks) != len(vectors):
            raise ValueError('Chunk/vector counts do not match.')
        # Small batches stay below the request size limit, including Unicode text.
        for start in range(0, len(chunks), 16):
            records = []
            for chunk, vector in zip(chunks[start:start+16], vectors[start:start+16]):
                records.append({'id': chunk['chunk_id'], 'values': vector, 'metadata': {
                    'user_id': user_id, 'project_id': project_id, 'file_id': file_id,
                    'filename': chunk['filename'], 'page': chunk['page'], 'text': chunk['text']}})
            self._request('/vectors/upsert', {'namespace': namespace, 'vectors': records})

    def query(self, namespace, vector, user_id, project_id, ready_ids, top_k=18, query_text=None):
        if not ready_ids:
            return []
        if self.embeds_documents:
            from pinecone import EmbedQuery
            response = self._documents('search', namespace=namespace, top_k=top_k,
                score_by=[EmbedQuery(field=self.embedding_field, query=query_text)],
                filter={'$and': [{'user_id': {'$eq': user_id}},
                    {'project_id': {'$eq': project_id}}, {'file_id': {'$in': ready_ids}}]})
            return [{'id': match.id, 'score': match.score} for match in response.matches]
        data = self._request('/query', {'namespace': namespace, 'vector': vector,
            'topK': top_k, 'includeMetadata': True, 'includeValues': False,
            'filter': {'$and': [{'user_id': {'$eq': user_id}},
                               {'project_id': {'$eq': project_id}},
                               {'file_id': {'$in': ready_ids}}]}})
        return data.get('matches', [])

    def delete_file(self, namespace, file_id):
        if self.embeds_documents:
            self._documents('delete', namespace=namespace, filter={'file_id': {'$eq': file_id}})
            return
        self._request('/vectors/delete', {'namespace': namespace, 'filter': {'file_id': {'$eq': file_id}}})
