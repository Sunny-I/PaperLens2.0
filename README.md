# PaperLens

## Persistent hosted workspace

The default app now uses **Google login, PostgreSQL projects and chat, private
Cloudflare R2 PDFs, OpenAI embeddings/answers, and project-scoped Pinecone search**.
See **[SETUP.md](SETUP.md)** for `.env` configuration, startup, deployment, and limits.
Run `python scripts/run.py` after configuring the credentials.

The documentation below describes the original session-only dashboard, preserved
as `legacy_app.py`. Run `streamlit run legacy_app.py` to use that mode.

### OpenAI answers and summaries

Install the dependencies from `requirements.txt`, then configure `OPENAI_API_KEY`
in the server environment or `.streamlit/secrets.toml` (never paste it into chat).
An explicitly empty environment value is rejected rather than falling back to secrets.
In **Settings**, select **Model provider → OpenAI** and enable **Generate AI answers**.
The default chat model is `gpt-4.1-mini`; enter another model supported by your
account or set `OPENAI_CHAT_MODEL`. Chat settings are remembered per provider.
The existing Gemini default remains available; select OpenAI explicitly to switch.

OpenAI is used for answers and optional summaries, independently of the embedding
provider. Existing indexes do not need rebuilding when only the chat provider changes.
For example, Voyage can retrieve passages while OpenAI generates cited answers.
The official OpenAI SDK uses the Responses API with Pydantic Structured Outputs,
`store=False`, a 120-second timeout and one SDK retry for transient API errors.
Citation/schema failures get at most one repair attempt. Refusals, incomplete
responses and provider failures fall back to source excerpts. Error messages never
display raw provider exceptions or keys. Requests send retrieved passages (or paper
batches for summaries) to OpenAI and may incur API charges.

Reference: [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

### Voyage embeddings with Gemini answers

Embeddings and answer generation now have independent providers. Under Settings:

1. Keep **Model provider → Gemini** for answers and summaries.
2. Set **Search method for new uploads → Semantic embeddings**.
3. Choose **Embedding provider → Voyage AI** and **Embedding model → voyage-4-lite**.
4. Add your own key locally to the existing ignored file **.streamlit/secrets.toml**:

   VOYAGE_API_KEY = "your-voyage-api-key"

Keep the existing GEMINI_API_KEY entry for answers. You can also set VOYAGE_API_KEY in the server environment. An explicitly empty environment key is rejected instead of falling back to secrets. Never put real keys in secrets.toml.example.

The model dropdown offers Voyage4 lite/standard/large/code, Gemini embedding001/2, and existing Ollama options. Changing embedding providers leaves the chat provider unchanged. Each provider remembers its model within the session. Existing explicit environment/legacy model selections remain in the dropdown rather than being silently overwritten. Provider model access and quotas still apply.

Click **Analyze papers** with uploaded files, or **Try sample papers** for the sample collection, to rebuild using your selection. Until the rebuild succeeds, the old index remains active and queries continue using its original provider/model. Voyage document and query requests use their respective input types; failures never switch to Gemini or TF-IDF. The adapter uses the hosted REST endpoint through httpx, batches32chunks, and disables silent input truncation.

Verification:67 automated tests passed, including mocked Voyage transport, batching, query/document task separation, safe quota errors, provider independence and dropdown persistence. Live Voyage verification is pending a configured VOYAGE_API_KEY.

A research-paper summarization and cross-paper Q&A system built with LangChain, FAISS, PyMuPDF, Pydantic, and Streamlit. Gemini supports API-key-based chat and embeddings; Ollama remains available for local models. Offline evidence search also remains available.

## What it does

- Extracts text with PyMuPDF and splits it into overlapping **word** chunks with file/page citations.
- Indexes normalized vectors in FAISS. Offline mode uses TF-IDF keyword matching; optional Ollama embeddings enable semantic retrieval.
- Retrieves passages above a configurable similarity threshold, with balanced representation across relevant papers.
- Displays question-specific evidence by paper. Offline mode returns verbatim evidence excerpts, not synthesized answers.
- Generates cross-paper answers through a LangChain prompt/model chain using `ChatOllama.with_structured_output(GeneratedAnswer)`. Each claim requires source IDs and exact supporting quotes. Schema or citation failures trigger one repair attempt, then an explicit excerpt fallback.
- Generates structured problem, methodology, datasets, findings, and limitations using LangChain with a Pydantic `SummaryFields` schema. Every statement requires page-specific quotes. All pages are processed in bounded batches; synthesized observations are deduplicated and combined without dropping late-page findings.
- Supports selected-paper comparisons, a cross-paper summary table, and JSON downloads for summaries and answers with citations.
- Provides offline section excerpts when generation is disabled or fails. Missing information stays unknown.

LangChain Core composes prompts and structured-output chains; `langchain-google-genai` and `langchain-ollama` supply provider integrations. Pydantic validates structure; it does not establish factual accuracy. Quote matching verifies provenance, not whether a generated claim is logically supported.

## Setup

Use Python 3.12+ (the included environment was tested on Python 3.14). From the project directory:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit. Upload text-based PDFs, or select **Try sample papers**.

The Try sample papers button loads the actual PDFs currently in the project’s sample_papers folder. It does not generate substitute documents. Add or replace PDFs in that folder to change the sample collection.

## Everyday use

Upload PDFs and choose **Analyze papers**, then use **Ask your papers** or **Paper summaries**. Questions search **All papers** by default; choose **Selected papers** to narrow the scope. AI answers are enabled by default. **Summarize on upload** is optional; summaries can also be requested later. **Clear library** requires explicit confirmation and clears the upload selection as well as session results.

Provider names, chat/embedding model IDs, credentials instructions, the retrieval method and similarity threshold are inside the collapsed **Settings** area. The embedding-model field appears only for semantic retrieval. FAISS metadata and timings are inside **Developer diagnostics**. These are configuration areas, not access-controlled admin pages.

## Gemini API setup (no Ollama required)

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the placeholder with your Gemini API key locally. The real secrets file is gitignored. Alternatively, set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment used to start Streamlit. Environment values take precedence over Streamlit secrets; `GEMINI_API_KEY` takes precedence over `GOOGLE_API_KEY` within each source. `.env` files are not loaded automatically.
2. Start or restart the app and open **Settings** and select **Gemini** (the default provider).
3. Enter a chat model ID available in your Google AI Studio account. The current default is `gemini-3.8-flash`. Never enter `text-embedding-004` or another embedding model in the Chat model field. You can also set `GEMINI_CHAT_MODEL` in the environment.
4. Enable **Generate AI answers** and/or **Summarize on upload**. Existing papers can be summarized from the Paper summaries tab.
5. Keep **TF-IDF keywords (offline)** for local retrieval, or select **Semantic embeddings** and enter an embedding model ID (or set `GEMINI_EMBED_MODEL`). Reprocess papers when changing embedding providers/models.

Gemini uses LangChain's `ChatGoogleGenerativeAI` and `GoogleGenerativeAIEmbeddings` with the Gemini Developer API, not Vertex AI. It preserves Pydantic output validation, citation checks, repair, and excerpt fallbacks. Document and query embeddings use their respective retrieval tasks. Gemini operations send text to Google and may consume your API quota; model IDs and access depend on your account. Credentials are not included in downloads or displayed provider errors. The normal regression suite mocks model responses. Separate opt-in live audit scripts use the configured key; see the audit report for actual results.

See the [LangChain Gemini integration documentation](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai) for provider details.

## Optional semantic retrieval and generated answers

Run an Ollama server and install a compatible embedding model and/or chat model of your choice. Enter the server URL and installed model names in the sidebar. Nothing is downloaded by PaperLens.

- Select **Ollama** as the model provider and **Semantic embeddings** before processing PDFs to build a neural vector index.
- Enable **Generate AI answers** and enter a chat model that supports structured JSON output to synthesize answers from retrieved evidence. Generation also works with the TF-IDF index.
- Enable **Summarize on upload**, or use **Summarize this paper** or **Summarize all papers** in the Paper summaries tab. This also works for already indexed papers without rebuilding the vector index.
- Generated summaries are retained across dashboard reruns. A failed refresh preserves a previous successful summary and displays a warning.
- Leaving generation disabled keeps the app offline when using TF-IDF.

Optional defaults:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_EMBED_MODEL=your-installed-embedding-model
export OLLAMA_CHAT_MODEL=your-installed-chat-model
streamlit run app.py
```

The [LangChain Ollama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama) uses `ChatPromptTemplate`, `ChatOllama.with_structured_output` with Pydantic classes, and `OllamaEmbeddings`. Malformed output and invalid source quotes get at most one repair attempt; connection errors fall back immediately. Model calls have a 120-second request timeout, a 16,384-token context setting, and a 4,096-token output limit. Paper text is processed in batches of at most 10,000 characters; the full paper is covered rather than silently truncated. Summary generation can require several calls per paper. The configured chat model must support the requested context and structured output. PDF excerpts and queries are sent to the configured server when its features are enabled. Indexes retain their embedding configuration; changing the sidebar settings takes effect on the next indexing run.

## Library behavior

Uploads add to the current library by default; **Start a new collection** replaces it. Content hashes deduplicate identical files and distinguish different PDFs sharing a filename. All papers, excerpts, and searchable chunks are committed together only after a successful batch. A failed batch leaves the previous library intact. **Clear library** removes the session's corpus and previous answer.

Indexes live in Streamlit session memory and are not persisted across server restarts. Password-protected, corrupt, and textless PDFs produce actionable errors; OCR is not included. Section detection is heuristic and complex layouts or unconventional headings can cause missed sections. Review the original paper when interpreting excerpts.

## Tests and evaluation

```bash
python -m unittest discover -s tests -v
python scripts/make_audit_fixtures.py
python -m pipeline.evaluate audit/fixtures/valid audit/fixtures/queries.json
```

Tests cover ingestion consistency, duplicate uploads and filenames, PDF errors, retrieval abstention and paper filters, real LangChain/Pydantic parsing, bounded repair, citation rejection, full 31-page text coverage, six-paper Q&A, embedding batching, summary refresh behavior, and Streamlit generation/search workflows. Ollama transport responses are mocked while the actual LangChain chains execute: these tests do not establish real model quality or validate a live Ollama deployment. See `audit/FUNCTIONAL_AUDIT.md` for the separately executed browser and live Gemini checks and their limitations.

Explicit model fields persist across provider/search-mode changes within a session. Changing an embedding model applies on the next analysis: existing queries retain the index's original embedding provider/model. The UI reports pending search changes. API keys resolve from environment before Streamlit secrets, with `GEMINI_API_KEY` before `GOOGLE_API_KEY` within each source. An explicitly empty higher-priority key is an error; it does not silently select another key. Clearing the library preserves settings; a full browser reload starts a new in-memory session.

The generated audit fixtures are synthetic smoke tests, not a representative benchmark. Gold filenames and pages must exist in the indexed corpus; invalid labels raise an error. To evaluate your own corpus, create a JSON array:

```json
[
  {"query": "What method does this paper propose?", "relevant": [{"filename": "paper.pdf", "page": 3}]},
  {"query": "A question unsupported by this corpus", "relevant": []}
]
```

Then run:

```bash
python -m pipeline.evaluate /path/to/papers /path/to/questions.json --top-k 6 --min-score 0.05
```

The CLI evaluates the offline TF-IDF pipeline and reports recall@k, reciprocal rank, abstention on unsupported queries, timing, and retrieved pages. Page numbers are one-based; omit `page` for paper-level relevance. The Python `evaluate` function also accepts a store configured with neural embeddings.

Calibrate similarity thresholds against labeled queries: scores are not confidence probabilities. Evaluate generated claim support separately with human review. No fixed latency, accuracy, schema-consistency percentage, or time-saving claims are made.

## Files

- `app.py`: Streamlit controls, evidence display, and per-session state.
- `pipeline/pdf_parser.py`: PDF extraction and word chunking.
- `pipeline/vector_store.py`: transactional FAISS indexing and retrieval.
- `pipeline/schema_extractor.py`: source-backed section excerpts.
- `pipeline/corpus.py`: atomic library ingestion.
- `pipeline/gemini_client.py`: Gemini chat/embedding integration and provider embedding adapter.
- `pipeline/ollama_client.py`: LangChain chat/embedding integrations, Pydantic parsing, and bounded repair.
- `pipeline/summarizer.py`: full-paper batched synthesis and page/quote validation.
- `pipeline/answer_generator.py`: cited generation, validation, and offline fallback.
- `pipeline/evaluate.py`: labeled retrieval evaluation.
- `tests/`: regression checks and demo queries.
