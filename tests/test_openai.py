"""OpenAI SDK transport and application tests; no real credentials or network."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx2
from openai import OpenAI
from streamlit.testing.v1 import AppTest

from pipeline.openai_client import OpenAIClient, resolve_openai_key
from pipeline.answer_generator import AnswerGenerator, GeneratedAnswer
from pipeline.summarizer import PaperSummarizer
from test_pipeline import chunk, pdf


class OpenAITests(unittest.TestCase):
    def transport(self, handler):
        sdk = OpenAI(api_key="fake-secret", max_retries=0,
                     http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
        factory = patch("pipeline.openai_client.OpenAI", return_value=sdk)
        factory.start()
        self.addCleanup(factory.stop)
        return sdk

    @staticmethod
    def response(payload=None, status="completed", refusal=False):
        content = ({"type": "refusal", "refusal": "Cannot comply"} if refusal else
                   {"type": "output_text", "text": json.dumps(payload), "annotations": []})
        return httpx2.Response(200, json={"id": "resp_test", "object": "response",
            "created_at": 0, "model": "test-model", "status": status,
            "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                        "status": "completed", "content": [content]}]})

    def test_sdk_parsing_and_citation_repair(self):
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertFalse(body["store"])
            self.assertTrue(body["text"]["format"]["strict"])
            citation = {"source_id": 99 if len(requests) == 1 else 1,
                        "quote": "The transformer uses attention layers."}
            return self.response({"claims": [{"text": "Uses attention.", "citations": [citation]}]})
        self.transport(handler)
        result = AnswerGenerator().answer("Which method?", [chunk("The transformer uses attention layers.")],
                                          OpenAIClient("fake-secret"), "test-model")
        self.assertEqual(result["mode"], "generated")
        self.assertEqual(len(requests), 2)
        self.assertIn("Regenerate", requests[1]["input"][-1]["content"])
        self.assertIn('Source ID 99 does not exist', requests[1]['input'][-1]['content'])
        self.assertEqual(requests[1]['input'][-2]['role'], 'assistant')

    def test_bad_quotes_report_citation_stage_and_one_fallback_notice(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return self.response({'claims': [{'text': 'Uses attention.', 'citations': [
                {'source_id': 1, 'quote': 'A fabricated quotation that is absent from the source.'}]}]})
        self.transport(handler)
        result = AnswerGenerator().answer('Which method?', [chunk('The transformer uses attention layers.')],
            OpenAIClient('fake-secret'), 'test-model')
        self.assertEqual(result['failure_stage'], 'citations')
        self.assertEqual(result['warning'].count('Showing source excerpts instead.'), 1)
        self.assertIn('one contiguous passage', requests[1]['input'][-1]['content'])

    def test_numbered_evidence_preserves_pdf_accented_text(self):
        source = 'The paper was written by Tolga ¸Sakar and Hakan Emekci.'
        self.transport(lambda request: self.response({'claims': [{'text': 'The authors are Tolga Şakar and Hakan Emekci.',
            'citations': [{'source_id': 1, 'evidence_id': 1}]}]}))
        result = AnswerGenerator().answer('Who wrote it?', [chunk(source)],
            OpenAIClient('fake'), 'test', evidence_ids=True)
        self.assertEqual(result['mode'], 'generated')
        self.assertEqual(result['claims'][0]['citations'][0]['quote'], source)

    def test_unknown_evidence_id_is_rejected_after_bounded_repair(self):
        calls = []
        def handler(request):
            calls.append(json.loads(request.content))
            return self.response({'claims': [{'text': 'An unsupported claim.',
                'citations': [{'source_id': 1, 'evidence_id': 99}]}]})
        self.transport(handler)
        result = AnswerGenerator().answer('Who wrote it?', [chunk('This is the actual source passage.')],
            OpenAIClient('fake'), 'test', evidence_ids=True)
        self.assertEqual(result['mode'], 'extractive')
        self.assertEqual(result['failure_stage'], 'citations')
        self.assertEqual(len(calls), 2)
        self.assertIn('source_id/evidence_id pairs', calls[1]['input'][-1]['content'])

    def test_schema_repair_is_bounded(self):
        calls = []
        def handler(request):
            calls.append(request)
            return self.response({"bad": "schema"})
        self.transport(handler)
        with self.assertRaisesRegex(ValueError, "twice"):
            OpenAIClient("fake").structured("test", [], GeneratedAnswer)
        self.assertEqual(len(calls), 2)

    def test_api_error_is_safe_and_falls_back(self):
        self.transport(lambda request: httpx2.Response(429,
            json={"error": {"message": "fake-secret sensitive URL", "type": "insufficient_quota"}}))
        result = AnswerGenerator().answer("Which method?", [chunk("The transformer uses attention layers.")],
                                          OpenAIClient("fake-secret"), "test")
        self.assertEqual(result["mode"], "extractive")
        self.assertIn("quota", result["warning"])
        self.assertNotIn("fake-secret", result["warning"])

    def test_refusal_and_incomplete_fall_back_without_repair(self):
        for status, refusal in [("completed", True), ("incomplete", False)]:
            with self.subTest(status=status):
                calls = []
                def handler(request):
                    calls.append(request)
                    return self.response({"claims": []}, status=status, refusal=refusal)
                self.transport(handler)
                result = AnswerGenerator().answer("q", [chunk("Source text for testing.")],
                                                  OpenAIClient("fake"), "test")
                self.assertEqual(result["mode"], "extractive")
                self.assertEqual(len(calls), 1)

    def test_summary_uses_same_client(self):
        quote = "The transformer uses attention layers."
        self.transport(lambda request: self.response({"research_problem": None,
            "methodology": {"text": "Uses attention.", "evidence": [{"page": 1, "quote": quote}]},
            "datasets_and_benchmarks": None, "key_findings": [], "limitations": []}))
        result = PaperSummarizer().summarize({"filename": "test.pdf", "raw_text": quote,
            "pages": [{"page": 1, "text": quote}]}, OpenAIClient("fake"), "test")
        self.assertEqual(result.mode, "generated")

    def test_summary_reference_schema_is_parsed_by_sdk(self):
        quote = 'The RAG method improves efﬁciency in the study by ¸Sakar.'
        self.transport(lambda request: self.response({'research_problem': None,
            'methodology': {'text': 'Improves RAG efficiency.', 'evidence': [{'page': 1, 'evidence_id': 1}]},
            'datasets_and_benchmarks': None, 'key_findings': [], 'limitations': []}))
        result = PaperSummarizer().summarize({'filename': 'test.pdf', 'raw_text': quote,
            'pages': [{'page': 1, 'text': quote}]}, OpenAIClient('fake'), 'test', evidence_ids=True)
        self.assertEqual(result.mode, 'generated')
        self.assertEqual(result.summary.evidence['methodology'][0].quote, quote)

    def test_key_precedence_and_validation(self):
        self.assertEqual(resolve_openai_key({"OPENAI_API_KEY": "secret"},
                                           {"OPENAI_API_KEY": " env "}), "env")
        for env in [{"OPENAI_API_KEY": ""}, {"OPENAI_API_KEY": None}]:
            with self.assertRaises(ValueError):
                resolve_openai_key({"OPENAI_API_KEY": "secret"}, env)
        with self.assertRaises(ValueError):
            resolve_openai_key({}, {})

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake"})
    @patch("pipeline.openai_client.OpenAIClient.structured", return_value=GeneratedAnswer(claims=[]))
    @patch("pipeline.demo_papers.load_demo_papers", return_value=[("test.pdf", pdf("The transformer uses attention layers."))])
    def test_app_provider_switch_and_generation(self, samples, generate):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "legacy_app.py"), default_timeout=30).run()
        def ctl(kind, label):
            return next(x for x in getattr(app, kind) if x.label == label)
        ctl("selectbox", "Model provider").select("OpenAI").run()
        ctl("text_input", "Chat model").set_value("custom-openai").run()
        ctl("selectbox", "Model provider").select("Gemini").run()
        ctl("selectbox", "Model provider").select("OpenAI").run()
        self.assertEqual(ctl("text_input", "Chat model").value, "custom-openai")
        ctl("button", "Try sample papers").click().run()
        ctl("text_input", "Research question").set_value("attention")
        ctl("button", "Ask").click().run()
        self.assertFalse(app.exception)
        generate.assert_called_once()
        self.assertEqual(app.session_state["last_answer"]["configuration"]["provider"], "OpenAI")
