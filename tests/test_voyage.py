import os
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
from streamlit.testing.v1 import AppTest
from pipeline.voyage_client import VoyageClient
from pipeline.answer_generator import GeneratedAnswer
from test_pipeline import pdf


class VoyageTests(unittest.TestCase):
    @patch("pipeline.voyage_client.httpx.Client")
    def test_batches_tasks_order_and_credentials(self, client):
        def respond(url, headers, json):
            self.assertEqual(url, "https://api.voyageai.com/v1/embeddings")
            self.assertEqual(headers["Authorization"], "Bearer fake-voyage")
            self.assertEqual(json["model"], "voyage-4-lite")
            self.assertFalse(json["truncation"])
            return httpx.Response(200, json={"data": [
                {"index": i, "embedding": [float(i), 1.]} for i in reversed(range(len(json["input"])))
            ]}, request=httpx.Request("POST", url))
        post = client.return_value.__enter__.return_value.post
        post.side_effect = respond
        api = VoyageClient("fake-voyage")
        vectors = api.embed(["chunk"] * 33, "voyage-4-lite")
        self.assertEqual(len(vectors), 33)
        self.assertEqual(vectors[0], [0., 1.])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["json"]["input_type"], "document")
        self.assertEqual(api.embed_query("question", "voyage-4-lite"), [0., 1.])
        self.assertEqual(post.call_args.kwargs["json"]["input_type"], "query")

    @patch("pipeline.voyage_client.httpx.Client")
    def test_safe_quota_failure_no_retry(self, client):
        post = client.return_value.__enter__.return_value.post
        post.return_value = httpx.Response(429, text="fake-secret", request=httpx.Request("POST", "https://api.voyageai.com/v1/embeddings"))
        with self.assertRaisesRegex(ValueError, "quota") as caught:
            VoyageClient("fake-secret").embed(["chunk"], "voyage-4")
        self.assertNotIn("fake-secret", str(caught.exception))
        self.assertEqual(post.call_count, 1)
        with self.assertRaisesRegex(ValueError, "VOYAGE_API_KEY"):
            VoyageClient("")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "fake-gemini", "VOYAGE_API_KEY": "fake-voyage"})
    @patch("pipeline.demo_papers.load_demo_papers", return_value=[("audit.pdf", pdf("Orion achieved 85 percent accuracy."))])
    @patch("pipeline.gemini_client.GeminiClient.structured", return_value=GeneratedAnswer(claims=[]))
    @patch("pipeline.voyage_client.VoyageClient.embed_query", return_value=[1., 0.])
    @patch("pipeline.voyage_client.VoyageClient.embed", return_value=[[1., 0.]])
    def test_independent_providers_and_dropdown_memory(self, embed, query, chat, samples):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "legacy_app.py"), default_timeout=30).run()
        def ctl(kind, label):
            return next(x for x in getattr(app, kind) if x.label == label)
        ctl("selectbox", "Search method for new uploads").select("Semantic embeddings").run()
        self.assertEqual(ctl("selectbox", "Embedding provider").value, "Voyage AI")
        ctl("selectbox", "Embedding model").select("voyage-4").run()
        ctl("selectbox", "Embedding provider").select("Gemini").run()
        ctl("selectbox", "Embedding model").select("gemini-embedding-2").run()
        ctl("selectbox", "Embedding provider").select("Voyage AI").run()
        self.assertEqual(ctl("selectbox", "Embedding model").value, "voyage-4")
        ctl("button", "Try sample papers").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(embed.call_args.args[1], "voyage-4")
        ctl("text_input", "Research question").set_value("Orion accuracy?")
        ctl("button", "Ask").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(query.call_args.args[1], "voyage-4")
        chat.assert_called_once()
        self.assertEqual(app.session_state["last_answer"]["configuration"]["provider"], "Gemini")
        self.assertIn("Voyage AI", app.session_state["corpus"].retrieval_label)
