"""Transactional corpus ingestion shared by the UI and tests."""
from dataclasses import dataclass, field, replace
from pipeline.pdf_parser import PDFParser
from pipeline.schema_extractor import StructuredExtractor
from pipeline.vector_store import FaissVectorStore
from pipeline.summarizer import PaperSummarizer


@dataclass
class Corpus:
    papers: dict = field(default_factory=dict)
    summaries: dict = field(default_factory=dict)
    store: FaissVectorStore = field(default_factory=FaissVectorStore)
    stats: dict = field(default_factory=dict)
    retrieval_label: str = "TF-IDF keyword retrieval"
    summary_results: dict = field(default_factory=dict)


def summarize_corpus(corpus, client, model, doc_ids=None):
    if not model.strip():
        raise ValueError("Enter an installed chat model to generate summaries.")
    summaries = dict(corpus.summaries)
    results = dict(corpus.summary_results)
    for doc_id in (list(corpus.papers) if doc_ids is None else doc_ids):
        result = PaperSummarizer().summarize(corpus.papers[doc_id], client, model)
        # Failed refreshes preserve an earlier valid generated summary and show the failure.
        previous = results.get(doc_id)
        if result.mode != "generated" and previous and previous.mode == "generated":
            results[doc_id] = replace(previous, warning="Summary refresh failed. Keeping the previous generated summary.")
        else:
            results[doc_id] = result
            summaries[doc_id] = result.summary
    return replace(corpus, summaries=summaries, summary_results=results)


def ingest(files, existing=None, replace=False, embedder=None, retrieval_label="TF-IDF keyword retrieval"):
    papers = {} if replace or existing is None else dict(existing.papers)
    summaries = {} if replace or existing is None else dict(existing.summaries)
    parser, extractor = PDFParser(), StructuredExtractor()
    parse_seconds = 0.0
    if not files:
        raise ValueError("Select at least one PDF.")
    for filename, data in files:
        parsed = parser.parse_pdf(data, filename)
        doc_id = parsed["doc_id"]
        parse_seconds += parsed["parse_duration_sec"]
        if doc_id not in papers:
            papers[doc_id] = parsed
            summaries[doc_id] = extractor.extract(parsed["raw_text"], filename, parsed["pages"])
    chunks = [chunk for paper in papers.values() for chunk in paper["chunks"]]
    store = FaissVectorStore(embedder=embedder)
    stats = store.build_index(chunks)
    stats.update(parse_time_sec=round(parse_seconds, 4), papers_count=len(papers))
    results = {} if replace or existing is None else dict(existing.summary_results)
    return Corpus(papers, summaries, store, stats, retrieval_label, results)
