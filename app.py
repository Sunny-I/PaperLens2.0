"""PaperLens: research evidence search and optional local RAG."""
import os
import json
import time
from dataclasses import replace as copy_corpus

import pandas as pd
import streamlit as st

from pipeline.demo_papers import load_demo_papers
from pipeline.voyage_client import VoyageClient, EMBEDDING_MODELS
from pipeline.configuration import resolve_api_key
from pipeline.answer_generator import AnswerGenerator
from pipeline.corpus import Corpus, ingest, summarize_corpus
from pipeline.ollama_client import OllamaClient
from pipeline.openai_client import OpenAIClient, DEFAULT_OPENAI_MODEL, resolve_openai_key
from pipeline.gemini_client import GeminiClient, ProviderEmbeddings, DEFAULT_CHAT_MODEL, validate_chat_model

st.set_page_config(page_title="PaperLens", page_icon="📄", layout="wide")
st.title("📄 PaperLens")
st.caption("Search research PDFs, inspect source excerpts, and generate answers with citations.")
if "corpus" not in st.session_state:
    st.session_state.corpus = Corpus()


def persistent_text(label, name, default, **kwargs):
    """Store settings independently from conditional Streamlit widget lifetimes."""
    saved = st.session_state.setdefault("saved_settings", {})
    saved.setdefault(name, st.session_state.get(name, default))
    widget_key = f"_setting_{name}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = saved[name]
    def remember():
        saved[name] = st.session_state[widget_key]
    value = st.text_input(label, key=widget_key, on_change=remember, **kwargs)
    saved[name] = value
    return value

def persistent_choice(label, name, options):
    saved = st.session_state.setdefault("saved_settings", {})
    default = os.getenv(name.upper().replace("_EMBEDDING_MODEL", "_EMBED_MODEL"), options[0])
    saved.setdefault(name, st.session_state.get(name, default))
    choices = list(options)
    if saved[name] not in choices:
        choices.append(saved[name])  # Keep explicit legacy/env selections reviewable.
    key = f"_choice_{name}"
    if key not in st.session_state:
        st.session_state[key] = saved[name]
    def remember():
        saved[name] = st.session_state[key]
    value = st.selectbox(label, choices, key=key, on_change=remember)
    saved[name] = value
    return value


with st.sidebar:
    st.header("Your papers")
    uploads = st.file_uploader("Upload papers", type=["pdf"], accept_multiple_files=True,
                               key=f"uploads_{st.session_state.get('upload_version', 0)}")
    replace = st.checkbox("Start a new collection", value=False,
                          help="When you analyze new uploads, they replace the current collection.")
    generate_summaries = st.checkbox("Summarize on upload", value=False)
    # From above to her only received the files 
    with st.expander("Settings", expanded=False):
        st.caption("Choose embeddings independently from the provider used for answers and summaries.")
        provider = st.selectbox("Model provider", ["Gemini", "OpenAI", "Ollama"],
                                help="Used for AI answers and summaries. Embeddings have a separate provider below.")
        generate = st.checkbox("Generate AI answers", value=True,
                               help="Turn off to display matching source excerpts without generating an answer.")
        retrieval = st.selectbox("Search method for new uploads", ["TF-IDF keywords (offline)", "Semantic embeddings"])
        base_url = "http://localhost:11434"
        if provider == "Ollama":
            base_url = persistent_text("Ollama server URL", "OLLAMA_BASE_URL", os.getenv("OLLAMA_BASE_URL", base_url))
        elif provider == "OpenAI":
            st.caption("Configure OPENAI_API_KEY in the environment or .streamlit/secrets.toml. Keys are never displayed.")
        else:
            st.caption("Configure GEMINI_API_KEY in the environment or .streamlit/secrets.toml. Keys are never displayed.")
        embedding_provider = "Voyage AI"
        embedding_model = ""
        if retrieval == "Semantic embeddings":
            embedding_provider = persistent_choice("Embedding provider", "embedding_provider", list(EMBEDDING_MODELS))
            prefix = "Voyage" if embedding_provider == "Voyage AI" else embedding_provider
            embedding_model = persistent_choice("Embedding model", f"{prefix}_embedding_model", EMBEDDING_MODELS[embedding_provider])
            st.caption("Voyage AI uses VOYAGE_API_KEY; Gemini uses GEMINI_API_KEY. Configure keys in .streamlit/secrets.toml or your environment.")
            if embedding_provider == "Ollama" and provider != "Ollama":
                base_url = persistent_text("Ollama server URL", "OLLAMA_BASE_URL", os.getenv("OLLAMA_BASE_URL", base_url))
        chat_model = persistent_text("Chat model", f"{provider}_chat_model", os.getenv(f"{provider.upper()}_CHAT_MODEL", DEFAULT_OPENAI_MODEL if provider == "OpenAI" else DEFAULT_CHAT_MODEL if provider == "Gemini" else ""),
                                  help="Use a chat model for answers and summaries, not an embedding model.")
        if provider == "Gemini" and chat_model.strip():
            try:
                validate_chat_model(chat_model)
            except ValueError as exc:
                st.error(str(exc))
        threshold = st.slider("Minimum cosine similarity", 0.0, 1.0,
                              0.05, 0.01, key="retrieval_min_score")
        if st.session_state.corpus.papers:
            requested = "TF-IDF keyword retrieval" if retrieval == "TF-IDF keywords (offline)" else f"{embedding_provider} semantic embeddings: {embedding_model.strip()}"
            if requested != st.session_state.corpus.retrieval_label:
                st.info(f"Current papers still use {st.session_state.corpus.retrieval_label}. Analyze papers to apply the new search settings.")
        st.caption("Advanced relevance filter. Higher values can exclude useful evidence; this is not a confidence score.")
    st.caption("Paper text goes to the selected embedding provider. Answer and summary requests go to the chat provider. API quotas and charges may apply.")

    def get_client(selected_provider=None):
        selected_provider = selected_provider or provider
        if selected_provider == "OpenAI":
            if "OPENAI_API_KEY" in os.environ:
                return OpenAIClient(resolve_openai_key(environ=os.environ))
            try:
                return OpenAIClient(resolve_openai_key(st.secrets, environ={}))
            except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
                raise ValueError("Set OPENAI_API_KEY in your environment or Streamlit secrets before using OpenAI.") from None
        if selected_provider == "Voyage AI":
            if "VOYAGE_API_KEY" in os.environ:
                key = os.environ["VOYAGE_API_KEY"]
            else:
                try:
                    key = st.secrets.get("VOYAGE_API_KEY", "")
                except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
                    key = ""
            return VoyageClient(key)
        if selected_provider == "Ollama":
            return OllamaClient(base_url)
        if "GEMINI_API_KEY" in os.environ or "GOOGLE_API_KEY" in os.environ:
            api_key, key_source = resolve_api_key(environ=os.environ)
        else:
            try:
                secret_values = st.secrets
                api_key, key_source = resolve_api_key(secret_values, environ={})
            except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
                raise ValueError("Set GEMINI_API_KEY in your environment or Streamlit secrets before using Gemini.") from None
        st.session_state.credential_source = key_source
        return GeminiClient(api_key)

    process = st.button("Analyze papers", type="primary", disabled=not uploads)
    samples = st.button("Try sample papers")
    st.caption("Loads the PDFs from your sample_papers folder.")
    if st.session_state.corpus.papers and st.button("Clear library"):
        st.session_state.confirm_clear = True
    if st.session_state.get("confirm_clear"):
        st.warning("Remove all papers, summaries, and answers from this session?")
        confirm, cancel = st.columns(2)
        if confirm.button("Yes, clear library", type="primary"):
            st.session_state.corpus = Corpus()
            st.session_state.pop("last_answer", None)
            st.session_state.confirm_clear = False
            st.session_state.upload_version = st.session_state.get("upload_version", 0) + 1
            st.rerun()
        if cancel.button("Cancel"):
            st.session_state.confirm_clear = False
            st.rerun()
    if process or samples:
        try:
            if generate_summaries and not chat_model.strip():
                raise ValueError("Enter a chat model to generate summaries.")
            client = get_client() if generate_summaries else None
            embedder = None
            label = "TF-IDF keyword retrieval"
            if retrieval == "Semantic embeddings":
                if not embedding_model.strip():
                    raise ValueError("Enter an embedding model before indexing.")
                # Freeze settings for this index, including subsequent query embeddings.
                model_name = embedding_model.strip()
                embedder = ProviderEmbeddings(get_client(embedding_provider), model_name)
                label = f"{embedding_provider} semantic embeddings: {model_name}"
            files = load_demo_papers() if samples else [(f.name, f.getvalue()) for f in uploads]
            previous = st.session_state.corpus
            if samples:
                # Remove only the generated examples from the earlier demo implementation.
                legacy = {key for key, paper in previous.papers.items()
                          if paper["filename"] in {"Sample_Study_A.pdf", "Sample_Study_B.pdf"}
                          and "FICTIONAL EXAMPLE - not a published research paper." in paper["raw_text"]}
                previous = copy_corpus(previous,
                    papers={k: v for k, v in previous.papers.items() if k not in legacy},
                    summaries={k: v for k, v in previous.summaries.items() if k not in legacy},
                    summary_results={k: v for k, v in previous.summary_results.items() if k not in legacy})
            with st.spinner("Analyzing your papers…"):
                corpus = ingest(files, previous, replace, embedder, label)
            if generate_summaries:
                with st.spinner("Generating page-cited structured summaries…"):
                    corpus = summarize_corpus(corpus, client, chat_model.strip())
            st.session_state.corpus = corpus
            st.session_state.pop("last_answer", None)
            st.success(f"Your collection is ready: {len(corpus.papers)} papers.")
            st.session_state.confirm_clear = False
        except (ValueError, OSError) as exc:
            st.error(f"Library update failed: {exc} Your previous library is unchanged.")
    corpus = st.session_state.corpus
    st.divider()
    st.caption(f"{len(corpus.papers)} papers in your collection")
    for paper in corpus.papers.values():
        st.write(paper["filename"])
        st.caption(f"{paper['total_pages']} pages")

corpus = st.session_state.corpus
if not corpus.papers:
    st.info("Upload your research papers, then choose Analyze papers. Or try the sample collection. Scanned PDFs need text recognition before upload.")
    st.stop()

def paper_label(key):
    name = corpus.papers[key]["filename"]
    duplicate = sum(p["filename"] == name for p in corpus.papers.values()) > 1
    return f"{name} ({key[:8]})" if duplicate else name


qa, summaries = st.tabs(["Ask your papers", "Paper summaries"])
with qa:
    st.caption("Ask for an explanation, compare methods, or explore findings. Answers include supporting sources.")
    scope = st.radio("Search in", ["All papers", "Selected papers"], horizontal=True)
    selected_papers = []
    if scope == "Selected papers":
        selected_papers = st.multiselect("Choose papers", list(corpus.papers), format_func=paper_label)
    with st.form("research_question"):
        question = st.text_input("Research question", placeholder="What are the main findings, and how do the methods compare?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        st.session_state.pop("last_answer", None)
        if scope == "Selected papers" and not selected_papers:
            st.warning("Choose at least one paper, or switch to All papers.")
        elif not question.strip():
            st.warning("Enter a research question.")
        elif generate and not chat_model.strip():
            st.error("Enter a chat model, or turn off answer generation.")
        else:
            try:
                started = time.perf_counter()
                with st.spinner("Finding evidence and preparing the answer…"):
                    hits, latency = corpus.store.search(question, min_score=threshold,
                        doc_ids=selected_papers or None)
                    result = AnswerGenerator().answer(question, hits,
                        get_client() if generate else None, chat_model.strip() if generate else None)
                st.session_state.last_answer = {"question": question, "hits": hits, "result": result,
                    "doc_ids": selected_papers or list(corpus.papers),
                    "configuration": {"provider": provider if generate else None, "chat_model": chat_model.strip() if generate else None,
                                      "retrieval": corpus.retrieval_label, "min_score": threshold, "top_k": 6},
                    "retrieval_ms": latency, "total_seconds": round(time.perf_counter() - started, 3)}
            except ValueError as exc:
                st.error(str(exc))
    if "last_answer" in st.session_state:
        saved = st.session_state.last_answer
        result, hits = saved["result"], saved["hits"]
        st.download_button("Download answer and citations", json.dumps(saved, indent=2),
                           file_name="paperlens-answer.json", mime="application/json")
        st.subheader(saved["question"])
        st.caption("AI answer with sources" if result["mode"] == "generated" else "Source excerpts" if result["mode"] == "extractive" else "No supporting evidence found")
        if result["warning"]:
            st.warning(result["warning"])
        st.write(result["text"])
        for claim in result["claims"]:
            ids = sorted({c["source_id"] for c in claim["citations"]})
            st.write(claim["text"])
            st.caption("Sources: " + ", ".join(f"[{i}]" for i in ids))
            if result["mode"] == "generated":
                for citation in claim["citations"]:
                    st.text(f"[{citation['source_id']}] {citation['quote']}")
        if result["mode"] == "generated":
            st.caption("Source IDs and quoted text are checked automatically. Verify that each quote supports its claim.")
        if hits:
            st.subheader("Evidence by paper")
            rows = []
            for doc_id, paper in corpus.papers.items():
                if doc_id not in saved["doc_ids"]:
                    continue
                paper_hits = [(i, h) for i, h in enumerate(hits, 1) if h["doc_id"] == doc_id]
                rows.append({"Paper": paper_label(doc_id),
                             "Evidence for this question": "\n\n".join(f"[{i}] p. {h['page']}: {h['text']}" for i, h in paper_hits)
                             or "No retrieved evidence above the threshold."})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        for i, hit in enumerate(hits, 1):
            with st.expander(f"[{i}] {paper_label(hit['doc_id'])} · page {hit['page']}"):
                st.text(hit["text"])

with summaries:
    st.caption("Review each paper’s research problem, methods, findings, and limitations, with supporting page references.")
    selected = st.selectbox("Paper", list(corpus.papers), format_func=paper_label)
    summary_one = st.button("Summarize this paper")
    summary_all = st.button("Summarize all papers")
    if summary_one or summary_all:
        try:
            if not chat_model.strip():
                raise ValueError("Enter a chat model to generate summaries.")
            with st.spinner("Reading paper batches and validating summaries…"):
                corpus = summarize_corpus(corpus, get_client(), chat_model.strip(),
                                          None if summary_all else [selected])
            st.session_state.corpus = corpus
        except ValueError as exc:
            st.error(str(exc))
    summary = corpus.summaries[selected]
    summary_result = corpus.summary_results.get(selected)
    if summary_result:
        st.caption("AI-generated summary" if summary_result.mode == "generated" else "Source excerpts")
        if summary_result.warning:
            st.warning(summary_result.warning)
    else:
        st.info("These are source excerpts. Choose Summarize this paper for a concise AI summary.")
    st.subheader(summary.title)
    st.download_button("Download structured summary", summary.model_dump_json(indent=2),
                       file_name=f"summary-{selected[:8]}.json", mime="application/json")
    for field, label in [("research_problem", "Research problem"), ("methodology", "Methodology"),
                         ("datasets_and_benchmarks", "Datasets / experiments"), ("key_findings", "Results / conclusions"),
                         ("limitations", "Declared limitations")]:
        st.markdown(f"**{label}**")
        evidence = summary.evidence.get(field, [])
        if not evidence:
            st.write("Not found in the extracted text.")
        value = getattr(summary, field)
        if isinstance(value, list):
            for item in value:
                st.write(f"• {item}")
        elif value:
            st.write(value)
        if evidence:
            with st.expander(f"Supporting quotes: {label}"):
                for item in evidence:
                    st.text(item.quote)
                    st.caption(f"Source: {corpus.papers[selected]['filename']}, page {item.page}")
    st.subheader("Cross-paper summary comparison")
    st.dataframe(pd.DataFrame([{
        "Paper": paper_label(key),
        "Summary type": "AI summary" if key in corpus.summary_results and corpus.summary_results[key].mode == "generated" else "Source excerpts",
        "Problem": item.research_problem or "Not found",
        "Methodology": item.methodology or "Not found",
        "Findings": "\n".join(item.key_findings) or "Not found",
        "Limitations": "\n".join(item.limitations) or "Not found",
    } for key, item in corpus.summaries.items()]), width="stretch", hide_index=True)

with st.sidebar.expander("Developer diagnostics", expanded=False):
    requested_retrieval = "TF-IDF keyword retrieval" if retrieval == "TF-IDF keywords (offline)" else f"{embedding_provider} semantic embeddings: {embedding_model.strip()}"
    st.write(f"Active index: {corpus.retrieval_label}")
    st.write(f"Requested search configuration: {requested_retrieval}")
    if requested_retrieval != corpus.retrieval_label:
        st.info("Search settings have changed. Analyze papers to rebuild the index. Until then, queries use the active index's original model.")
    st.json({"current_chat_provider": provider, "current_chat_model": chat_model.strip(), "min_score": threshold,
             "credential_source": st.session_state.get("credential_source", "not loaded")})
    if "last_answer" in st.session_state:
        st.json(st.session_state.last_answer.get("configuration", {}))
    st.json(corpus.stats)
    st.json({"papers": len(corpus.papers), "vectors": corpus.store.index.ntotal,
             "dimensions": corpus.store.index.d, "index_type": "faiss.IndexFlatIP"})
    st.write("Normalized vectors use inner-product search for cosine similarity. TF-IDF matches vocabulary; Provider embeddings provide model-based semantic retrieval.")
    st.caption("Indexes are held in this session's memory. No benchmark or factual-accuracy guarantees are implied.")
