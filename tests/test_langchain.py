import json
import unittest
from unittest.mock import patch

from pipeline.answer_generator import AnswerGenerator, GeneratedAnswer
from pipeline.corpus import ingest, summarize_corpus
from pipeline.ollama_client import OllamaClient
from pipeline.summarizer import PaperSummarizer, SummaryFields, page_batches
from test_pipeline import pdf, chunk


def stream(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return iter([{"message": {"role": "assistant", "content": content}, "done": True}])


def fields(**kwargs):
    return dict(research_problem=None, methodology=None, datasets_and_benchmarks=None,
                key_findings=[], limitations=[], **kwargs) if not kwargs else {**fields(), **kwargs}


def finding(text="The model uses attention.", page=1, quote="The transformer uses attention layers."):
    return {"text": text, "evidence": [{"page": page, "quote": quote}]}


class LangChainTests(unittest.TestCase):
    @patch("ollama.Client.chat")
    def test_real_chain_parses_pydantic_and_repairs_invalid_json(self, chat):
        chat.side_effect = [stream("not json"), stream({"claims": []})]
        result = OllamaClient().structured("test-model", [{"role": "user", "content": "Question {literal braces}"}], GeneratedAnswer)
        self.assertIsInstance(result, GeneratedAnswer)
        self.assertEqual(chat.call_count, 2)
        request = chat.call_args.kwargs
        self.assertEqual(request["format"]["title"], "GeneratedAnswer")
        self.assertIn("failed validation", request["messages"][-1]["content"])

    @patch("ollama.Client.chat")
    def test_invalid_schema_retries_only_once(self, chat):
        chat.side_effect = lambda **kwargs: stream({"unexpected": "field"})
        with self.assertRaises(ValueError):
            OllamaClient().structured("test-model", [], GeneratedAnswer)
        self.assertEqual(chat.call_count, 2)

    @patch("ollama.Client.chat")
    def test_bad_citation_repaired_through_real_chain(self, chat):
        bad = {"claims": [{"text": "Uses attention.", "citations": [{"source_id": 8, "quote": "uses attention layers"}]}]}
        good = {"claims": [{"text": "Uses attention.", "citations": [{"source_id": 1, "quote": "uses attention layers"}]}]}
        chat.side_effect = [stream(bad), stream(good)]
        answer = AnswerGenerator().answer("Which method?", [chunk("The transformer uses attention layers.")], OllamaClient(), "test")
        self.assertEqual(answer["mode"], "generated")
        self.assertEqual(chat.call_count, 2)

    @patch("ollama.Client.chat", side_effect=ConnectionError("offline"))
    def test_connection_error_falls_back_without_retry(self, chat):
        result = AnswerGenerator().answer("q", [chunk("The transformer uses attention layers.")], OllamaClient(), "test")
        self.assertEqual(result["mode"], "extractive")
        self.assertIn("Ollama request failed", result["warning"])
        self.assertEqual(chat.call_count, 1)

    @patch("ollama.Client.embed")
    def test_langchain_embeddings_batched(self, embed):
        embed.side_effect = lambda model, texts, **kwargs: {"embeddings": [[3, 4]] * len(texts)}
        self.assertEqual(len(OllamaClient().embed(["text"] * 33, "embedding-model")), 33)
        self.assertEqual(embed.call_count, 2)

    @patch("ollama.Client.chat")
    def test_summary_synthesis_and_quote_provenance(self, chat):
        chat.return_value = stream(fields(methodology=finding()))
        corpus = ingest([("paper.pdf", pdf("Test Paper\nMethods\nThe transformer uses attention layers."))])
        updated = summarize_corpus(corpus, OllamaClient(), "test")
        key = next(iter(updated.papers))
        self.assertEqual(updated.summary_results[key].mode, "generated")
        self.assertEqual(updated.summaries[key].methodology, "The model uses attention.")
        self.assertEqual(updated.summaries[key].evidence["methodology"][0].page, 1)
        self.assertEqual(corpus.summary_results, {})

    @patch("ollama.Client.chat")
    def test_hallucinated_quote_and_page_fall_back(self, chat):
        for evidence in [finding(page=99), finding(quote="Entirely fabricated source quote.")]:
            chat.side_effect = lambda **kwargs: stream(fields(methodology=evidence))
            corpus = ingest([("p.pdf", pdf("Test Paper\nMethods\nThe transformer uses attention layers."))])
            result = PaperSummarizer().summarize(next(iter(corpus.papers.values())), OllamaClient(), "test")
            self.assertEqual(result.mode, "extractive")
            self.assertIn("failed", result.warning)

    def test_long_pages_split_without_losing_text(self):
        pages = [{"page": 1, "text": "abc " * 10000}, {"page": 2, "text": "Final finding."}]
        batches = list(page_batches(pages))
        self.assertTrue(all(sum(len(p["text"]) for p in batch) <= 10000 for batch in batches))
        self.assertEqual("".join(p["text"] for batch in batches for p in batch), "".join(p["text"] for p in pages))

    @patch("ollama.Client.chat")
    def test_all_31_pages_reach_summary_chain(self, chat):
        seen = []
        def respond(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            seen.extend(p["page"] for p in payload["pages"])
            last = next((p for p in payload["pages"] if p["page"] == 31), None)
            return stream(fields(key_findings=[finding("Late finding retained.", 31, "Accuracy increased on the final evaluation.")]) if last else fields())
        chat.side_effect = respond
        pages = [{"page": i, "text": ("Background evidence. " * 80 if i < 31 else "Accuracy increased on the final evaluation.")} for i in range(1, 32)]
        paper = {"filename": "long.pdf", "raw_text": "", "pages": pages}
        result = PaperSummarizer().summarize(paper, OllamaClient(), "test")
        self.assertEqual(result.mode, "generated")
        self.assertEqual(set(seen), set(range(1, 32)))
        self.assertEqual(result.summary.key_findings, ["Late finding retained."])

    @patch("ollama.Client.chat")
    def test_six_paper_qa_and_filtered_retrieval(self, chat):
        corpus = ingest([(f"paper{i}.pdf", pdf(f"Paper {i}\nThe transformer uses attention layers. Study {i}.")) for i in range(6)])
        hits, _ = corpus.store.search("attention")
        self.assertEqual(len({h["doc_id"] for h in hits}), 6)
        chat.return_value = stream({"claims": [{"text": "All six papers discuss attention layers.", "citations": [
            {"source_id": i, "quote": "The transformer uses attention layers."} for i in range(1, 7)]}]})
        result = AnswerGenerator().answer("Compare the methods.", hits, OllamaClient(), "test")
        self.assertEqual(result["mode"], "generated")
        chosen = next(iter(corpus.papers))
        filtered, _ = corpus.store.search("attention", doc_ids=[chosen])
        self.assertEqual({h["doc_id"] for h in filtered}, {chosen})

    @patch("ollama.Client.chat")
    def test_failed_refresh_preserves_successful_summary(self, chat):
        corpus = ingest([("p.pdf", pdf("Test Paper\nMethods\nThe transformer uses attention layers."))])
        chat.return_value = stream(fields(methodology=finding()))
        good = summarize_corpus(corpus, OllamaClient(), "test")
        chat.side_effect = ConnectionError("offline")
        failed = summarize_corpus(good, OllamaClient(), "test")
        key = next(iter(corpus.papers))
        self.assertEqual(failed.summaries[key], good.summaries[key])
        self.assertEqual(failed.summary_results[key].mode, "generated")
        self.assertTrue(failed.summary_results[key].warning)


if __name__ == "__main__":
    unittest.main()
