import unittest
import json
from test_pipeline import pdf
from pipeline.corpus import ingest
from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    sample_count = 3

    def setUp(self):
        fixtures = [(f"paper-{i}.pdf", pdf(f"Paper {i}\nAbstract: The transformer uses attention layers in study {i}.")) for i in range(3)]
        patcher = patch("pipeline.demo_papers.load_demo_papers", return_value=fixtures)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_simple_defaults_and_empty_selected_scope(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        self.assertTrue(next(c for c in app.checkbox if c.label == "Generate AI answers").value)
        self.assertFalse(any(t.label == "Embedding model" for t in app.text_input))
        next(b for b in app.button if b.label == "Try sample papers").click().run()
        self.assertEqual([tab.label for tab in app.tabs], ["Ask your papers", "Paper summaries"])
        next(r for r in app.radio if r.label == "Search in").set_value("Selected papers").run()
        next(t for t in app.text_input if t.label == "Research question").set_value("Explain the methods")
        next(b for b in app.button if b.label == "Ask").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Choose at least one paper" in w.value for w in app.warning))

    def test_sample_button_removes_old_generated_examples_but_keeps_uploads(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        app.session_state["corpus"] = ingest([
            ("Sample_Study_A.pdf", pdf("FICTIONAL EXAMPLE - not a published research paper.\nMethods: Invented data.")),
            ("my-upload.pdf", pdf("My actual paper with independent research content.")),
        ])
        app.run()
        next(b for b in app.button if b.label == "Try sample papers").click().run()
        self.assertFalse(app.exception)
        names = {p["filename"] for p in app.session_state["corpus"].papers.values()}
        self.assertNotIn("Sample_Study_A.pdf", names)
        self.assertIn("my-upload.pdf", names)
        self.assertEqual(len(names), 4)

    @patch("ollama.Client.chat")
    def test_generate_summaries_in_dashboard_and_preserve_on_rerun(self, chat):
        def respond(**kwargs):
            data = json.loads(kwargs["messages"][1]["content"])
            page = data["pages"][0]
            payload = {"research_problem": {"text": "A generated research problem.", "evidence": [
                {"page": page["page"], "quote": page["text"][:80]}]},
                "methodology": None, "datasets_and_benchmarks": None, "key_findings": [], "limitations": []}
            return iter([{"message": {"role": "assistant", "content": json.dumps(payload)}, "done": True}])
        chat.side_effect = respond
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        next(s for s in app.selectbox if s.label == "Model provider").select("Ollama").run()
        next(t for t in app.text_input if t.label == "Chat model").set_value("test-model")
        next(c for c in app.checkbox if c.label == "Summarize on upload").check()
        next(b for b in app.button if b.label == "Try sample papers").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["corpus"].summary_results), self.sample_count)
        self.assertTrue(all(r.mode == "generated" for r in app.session_state["corpus"].summary_results.values()))
        self.assertEqual(chat.call_count, self.sample_count)
        app.run()
        self.assertEqual(chat.call_count, self.sample_count)
        next(b for b in app.button if b.label == "Summarize this paper").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(chat.call_count, self.sample_count + 1)

    def test_summary_generation_requires_model(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        next(b for b in app.button if b.label == "Try sample papers").click().run()
        next(t for t in app.text_input if t.label == "Chat model").set_value("")
        next(b for b in app.button if b.label == "Summarize this paper").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("chat model" in e.value for e in app.error))

    def test_load_search_rerun_clear(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        self.assertFalse(app.exception)
        next(c for c in app.checkbox if c.label == "Generate AI answers").uncheck()
        next(b for b in app.button if b.label == "Try sample papers").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["corpus"].papers), self.sample_count)
        next(t for t in app.text_input if t.label == "Research question").set_value("transformer")
        next(b for b in app.button if b.label == "Ask").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["last_answer"]["hits"])
        app.run()
        self.assertEqual(app.session_state["last_answer"]["question"], "transformer")
        next(t for t in app.text_input if t.label == "Research question").set_value("xyzzyqwerty")
        next(b for b in app.button if b.label == "Ask").click().run()
        self.assertEqual(app.session_state["last_answer"]["result"]["mode"], "no_evidence")
        next(b for b in app.button if b.label == "Clear library").click().run()
        self.assertEqual(len(app.session_state["corpus"].papers), self.sample_count)
        next(b for b in app.button if b.label == "Cancel").click().run()
        self.assertEqual(len(app.session_state["corpus"].papers), self.sample_count)
        next(b for b in app.button if b.label == "Clear library").click().run()
        next(b for b in app.button if b.label == "Yes, clear library").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["corpus"].papers), 0)


if __name__ == "__main__":
    unittest.main()
