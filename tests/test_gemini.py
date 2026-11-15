import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pipeline.answer_generator import GeneratedAnswer, AnswerGenerator
from pipeline.gemini_client import GeminiClient, ProviderEmbeddings, validate_chat_model, provider_error
from pipeline.vector_store import FaissVectorStore
from test_pipeline import chunk


def response(data):
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(data)))])


class GeminiTests(unittest.TestCase):
    @patch("pipeline.gemini_client.ChatGoogleGenerativeAI")
    def test_embedding_model_rejected_before_any_request(self, chat):
        with self.assertRaisesRegex(ValueError, "embedding model cannot generate"):
            GeminiClient("fake-key").structured("text-embedding-004", [], GeneratedAnswer)
        chat.assert_not_called()
        self.assertEqual(validate_chat_model(" models/gemini-3.8-flash "), "gemini-3.8-flash")

    def test_error_categories_do_not_expose_provider_details(self):
        for code, expected in [(503, "overloaded"), (429, "quota"), (403, "credentials"), (404, "not found"), (400, "rejected")]:
            exc = RuntimeError("sensitive-key-in-request")
            exc.code = code
            message = provider_error(exc)
            self.assertIn(expected, message)
            self.assertNotIn("sensitive-key", message)

    def test_missing_key(self):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            GeminiClient("")

    @patch("langchain_google_genai.ChatGoogleGenerativeAI._generate")
    def test_actual_structured_chain_and_repair(self, generate):
        generate.side_effect = [response({"bad": "schema"}), response({"claims": []})]
        result = GeminiClient("fake-key").structured("test-model", [{"role": "user", "content": "Question?"}], GeneratedAnswer)
        self.assertEqual(result.claims, [])
        self.assertEqual(generate.call_count, 2)

    @patch("langchain_google_genai.ChatGoogleGenerativeAI._generate")
    def test_citations_and_provider_errors(self, generate):
        generate.return_value = response({"claims": [{"text": "Uses attention.", "citations": [
            {"source_id": 1, "quote": "The transformer uses attention layers."}]}]})
        hits = [chunk("The transformer uses attention layers.")]
        result = AnswerGenerator().answer("Which method?", hits, GeminiClient("fake-secret"), "test-model")
        self.assertEqual(result["mode"], "generated")
        generate.side_effect = RuntimeError("sensitive URL containing fake-secret")
        result = AnswerGenerator().answer("Which method?", hits, GeminiClient("fake-secret"), "test-model")
        self.assertEqual(result["mode"], "extractive")
        self.assertNotIn("fake-secret", result["warning"])

    @patch("langchain_google_genai.GoogleGenerativeAIEmbeddings.embed_query", return_value=[1.0, 0.0])
    @patch("langchain_google_genai.GoogleGenerativeAIEmbeddings.embed_documents")
    def test_documents_and_queries_use_separate_embedding_tasks(self, docs, query):
        docs.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
        client = GeminiClient("fake-key")
        store = FaissVectorStore(embedder=ProviderEmbeddings(client, "test-embedding"))
        store.build_index([chunk("Evidence about transformers.")])
        self.assertTrue(store.search("Architecture?")[0])
        docs.assert_called_once()
        query.assert_called_once_with("Architecture?")
        client.embed(["text"] * 33, "test-embedding")
        self.assertEqual(docs.call_count, 3)


if __name__ == "__main__":
    unittest.main()
