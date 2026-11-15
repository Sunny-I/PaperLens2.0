import json
import unittest
from unittest.mock import Mock, patch

import pymupdf

from pipeline.answer_generator import AnswerGenerator, GeneratedAnswer
from pipeline.corpus import ingest
from pipeline.evaluate import evaluate
from pipeline.ollama_client import OllamaClient
from pipeline.pdf_parser import PDFParser
from pipeline.schema_extractor import StructuredExtractor
from pipeline.vector_store import FaissVectorStore


def pdf(*pages):
    with pymupdf.open() as doc:
        for text in pages:
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text, fontsize=11)
        return doc.tobytes()


def chunk(text, doc="a", page=1):
    return dict(text=text, filename=f"{doc}.pdf", doc_id=doc, page=page, chunk_id=f"{doc}-{page}")


class ParserTests(unittest.TestCase):
    def test_pages_and_overlap(self):
        parser = PDFParser(5, 2)
        result = parser.parse_pdf(pdf("one two three four five six seven eight", "Second page evidence."), "a.pdf")
        self.assertEqual(result["total_pages"], 2)
        self.assertEqual(result["chunks"][0]["text"], "one two three four five")
        self.assertEqual(result["chunks"][1]["text"], "four five six seven eight")
        self.assertEqual(result["chunks"][-1]["page"], 2)
        self.assertEqual(len(result["chunks"]), 3)

    def test_invalid_parameters(self):
        for size, overlap in [(0, 0), (10, 10), (10, -1)]:
            with self.assertRaises(ValueError):
                PDFParser(size, overlap)

    def test_empty_corrupt_scanned_and_encrypted(self):
        with pymupdf.open() as doc:
            doc.new_page()
            encrypted = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                                   owner_pw="owner", user_pw="secret")
        for data in [b"", b"not a pdf", pdf(""), encrypted]:
            with self.subTest(data_size=len(data)), self.assertRaises(ValueError):
                PDFParser().parse_pdf(data, "bad.pdf")


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.store = FaissVectorStore()
        self.store.build_index([chunk("The transformer uses attention layers."),
                                chunk("Residual networks classify images.", "b")])

    def test_relevant_query_and_abstention(self):
        self.assertEqual(self.store.search("attention")[0][0]["doc_id"], "a")
        for query in ["", "  ", "xyzzyqwertyunknown", "the and or"]:
            self.assertEqual(self.store.search(query)[0], [])

    def test_empty_rebuild_clears_index(self):
        self.store.build_index([])
        self.assertIsNone(self.store.index)
        self.assertFalse(self.store.is_fitted)
        self.assertEqual(self.store.search("attention")[0], [])

    def test_failed_rebuild_preserves_index_and_vocabulary(self):
        before = self.store.search("attention")[0]
        with self.assertRaises(ValueError):
            self.store.build_index([chunk("the and or")])
        self.assertEqual(self.store.search("attention")[0], before)

    def test_balanced_papers(self):
        self.store.build_index([chunk("attention attention transformer", "a", i) for i in range(1, 5)]
                               + [chunk("attention image classification", "b")])
        hits, _ = self.store.search("attention", top_k=2)
        self.assertEqual({h["doc_id"] for h in hits}, {"a", "b"})

    def test_threshold_and_arguments(self):
        self.assertEqual(self.store.search("attention residual", min_score=1)[0], [])
        for kwargs in [{"top_k": 0}, {"min_score": -1}, {"min_score": 2}]:
            with self.assertRaises(ValueError):
                self.store.search("attention", **kwargs)

    def test_neural_vectors_normalized_and_checked(self):
        store = FaissVectorStore(embedder=lambda texts: [[3, 4] for _ in texts])
        store.build_index([chunk("semantic evidence")])
        self.assertAlmostEqual(store.search("paraphrase")[0][0]["similarity_score"], 1)
        for bad in [[[float("nan"), 1]], [[0, 0]], []]:
            with self.assertRaises(ValueError):
                FaissVectorStore(embedder=lambda texts: bad).build_index([chunk("text")])


class ExtractionTests(unittest.TestCase):
    def test_unknown_fields_have_no_fabricated_defaults(self):
        result = StructuredExtractor().extract("Unstructured text with no recognized sections", "unknown.pdf")
        self.assertIsNone(result.research_problem)
        self.assertIsNone(result.methodology)
        self.assertIsNone(result.datasets_and_benchmarks)
        self.assertEqual(result.key_findings, [])
        self.assertEqual(result.limitations, [])
        self.assertEqual(result.evidence, {})

    def test_multiline_sections_and_page_citations(self):
        pages = [{"page": 1, "text": "Test Paper\nAbstract: We study retrieval.\n2 Methods\nWe propose\na retrieval model."},
                 {"page": 2, "text": "Test Paper\n3 Results\nAccuracy reached 80%.\n4 Limitations\nOnly one dataset.\nReferences\nUnrelated citation."}]
        result = StructuredExtractor().extract("", "test.pdf", pages)
        self.assertEqual(result.methodology, "We propose a retrieval model.")
        self.assertEqual(result.key_findings, ["Accuracy reached 80%."])
        self.assertEqual(result.limitations, ["Only one dataset."])
        self.assertEqual(result.evidence["key_findings"][0].page, 2)


class CorpusTests(unittest.TestCase):
    def test_add_replace_duplicates_and_filename_collisions(self):
        first = pdf("Attention architecture and transformer networks.")
        second = pdf("Residual image classification networks.")
        corpus = ingest([("paper.pdf", first)])
        added = ingest([("paper.pdf", second)], corpus)
        self.assertEqual(len(added.papers), 2)
        self.assertEqual({h["doc_id"] for h in added.store.chunks_metadata}, set(added.papers))
        self.assertEqual(len(corpus.papers), 1)
        duplicate = ingest([("renamed.pdf", first)], added)
        self.assertEqual(len(duplicate.papers), 2)
        replaced = ingest([("paper.pdf", second)], added, replace=True)
        self.assertEqual(len(replaced.papers), 1)
        self.assertEqual(len(replaced.summaries), 1)
        self.assertEqual(replaced.store.search("attention")[0], [])

    def test_failed_batch_is_atomic(self):
        old = ingest([("old.pdf", pdf("Attention transformer architecture."))])
        with self.assertRaises(ValueError):
            ingest([("new.pdf", pdf("Residual image classification.")), ("bad.pdf", b"invalid")], old)
        self.assertEqual(len(old.papers), 1)
        self.assertEqual(old.store.search("attention")[0][0]["filename"], "old.pdf")


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.hits = [chunk("The transformer uses self-attention layers.")]
        self.client = Mock()
        self.payload = {"claims": [{"text": "The model uses self-attention.", "citations": [
            {"source_id": 1, "quote": "uses self-attention layers"}]}]}

    def answer(self):
        self.client.structured.return_value = GeneratedAnswer.model_validate(self.payload)
        return AnswerGenerator().answer("Which architecture?", self.hits, self.client, "test-model")

    def test_grounded_generation(self):
        result = self.answer()
        self.assertEqual(result["mode"], "generated")
        self.assertEqual(result["claims"][0]["citations"][0]["source_id"], 1)
        self.assertIn("Which architecture?", self.client.structured.call_args.args[1][1]["content"])

    def test_bad_citation_or_quote_falls_back(self):
        for citation in [{"source_id": 99, "quote": "uses self-attention layers"},
                         {"source_id": 1, "quote": "Unsupported fabricated result"}]:
            self.payload["claims"][0]["citations"] = [citation]
            result = self.answer()
            self.assertEqual(result["mode"], "extractive")
            self.assertIsNotNone(result["warning"])

    def test_invalid_json_and_outage_fall_back(self):
        for response in ["not json", '{"claims": [{"text": "uncited", "citations": []}]}']:
            self.client.structured.side_effect = ValueError("Invalid structured output")
            self.assertEqual(AnswerGenerator().answer("q", self.hits, self.client, "m")["mode"], "extractive")
        self.client.structured.side_effect = ValueError("unavailable")
        self.assertEqual(AnswerGenerator().answer("q", self.hits, self.client, "m")["mode"], "extractive")

    def test_abstention_and_offline(self):
        result = AnswerGenerator().answer("q", [], self.client, "m")
        self.assertEqual(result["mode"], "no_evidence")
        self.client.structured.assert_not_called()
        self.assertEqual(AnswerGenerator().answer("q", self.hits)["mode"], "extractive")
        self.payload = {"claims": []}
        self.assertEqual(self.answer()["claims"], [])


class EvaluationTests(unittest.TestCase):
    def test_recall_and_abstention(self):
        store = FaissVectorStore()
        store.build_index([chunk("Attention transformer architecture.")])
        report = evaluate(store, [
            {"query": "attention", "relevant": [{"filename": "a.pdf", "page": 1}]},
            {"query": "xyzzy", "relevant": []},
            {"query": "absent", "relevant": [{"filename": "a.pdf"}]},
        ])
        self.assertEqual(report["mean_recall_at_k"], 0.5)
        self.assertEqual(report["mean_reciprocal_rank"], 0.5)
        self.assertEqual(report["unanswerable_abstention_rate"], 1)


if __name__ == "__main__":
    unittest.main()
