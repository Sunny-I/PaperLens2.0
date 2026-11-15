# PaperLens functional audit inventory

Audit date: 2026-11-15 (Asia/Kolkata). Scope: all application modules, tests, configuration, CLI and provider integrations; repository is not a Git checkout. No credentials are recorded.

| Feature | Purpose / expected result | Implementation | Inputs | Dependencies | Failure risks | Execution plan |
|---|---|---|---|---|---|---|
| Startup/settings | Start dashboard; preserve explicit choices | app.py | provider, model IDs, search mode, generation flags, threshold | Streamlit, imports, secrets | missing imports, hidden-widget cleanup, changed slider defaults | AppTest state sequences + browser |
| Credentials | Supply key only to selected Gemini API | app.py get_client; scripts/check_gemini.py | env and Streamlit secrets | Google SDK | whitespace/empty precedence, malformed TOML | synthetic precedence tests + safe live key check |
| Upload | Read selected bytes; analyze only on action | app.py | file selection, replace flag | Streamlit uploader | file pointer, partial batch errors | actual browser upload + ingest tests |
| Samples | Read actual folder PDFs | demo_papers.py | directory | filesystem | empty directory, stale fixtures | exact filename/byte tests; actual folder parse |
| PDF parse | Text/page metadata and content hash | pdf_parser.py | bytes/name | PyMuPDF | corrupt, empty, encrypted, scanned, unusual chars | reproducible fixture corpus |
| Chunking | Overlapping per-page word windows | PDFParser._create_chunks | size=500, overlap=80 (backend only) | none | invalid numeric types, boundary loss | nondefault 7/2, exact/short/long/empty boundaries |
| Corpus | Atomic append/replace/dedup | corpus.py ingest | files, existing, replace, embedder | parser, schema, FAISS | failed rebuild, same filename, duplicates | multi-batch scenarios, failure preservation |
| Offline extraction | Heading-based excerpts, unknown absent fields | schema_extractor.py | raw text/pages/name | Pydantic | unknown headings/layout, false section attribution | complete/missing headings and page evidence |
| TF-IDF | Local vocabulary vectors | vector_store.py | max_features=8192, chunks | sklearn, NumPy | empty vocabulary, lexical mismatch, dense RAM | exact/unknown queries, explicit feature cap |
| Neural embeddings | Provider-selected document/query vectors | gemini_client.py; ollama_client.py | model, endpoint, key, timeout | Gemini/Ollama | API limits, 404, auth, mismatched dimensions, zero/NaN | adapter spies; real small sentence/PDF corpus |
| FAISS | Normalize and index; cosine retrieval | vector_store.py | top_k=6, threshold, doc IDs, balance flag | faiss-cpu | invalid vectors, invalid arguments, stale index config | nondefault k/threshold/filter, failure and rebuild tests |
| Q&A | Retrieve evidence then synthesize cited claims | app.py; answer_generator.py | question, matches, provider/model, generate flag | LangChain, provider | irrelevant question, invalid citations, schema errors, conflict, injection | ground-truth corpus; mocked error cases; small real requests |
| Summarization | Cover every page, typed cited fields | summarizer.py; corpus.py | paper(s), client/model; batches=10000 chars | LangChain, provider | missing text, false facts despite valid quotes, failure mid-batch | short/31-page corpus; duplicate facts; invalid response; real summary |
| State/reset | Preserve library across reruns; clear with confirmation | app.py session_state | clear/cancel, provider/mode changes | Streamlit | stale results, widget values lost, active vs requested index | repeated interactions and app reload |
| Export | JSON answer/citations and summary | app.py download_button | current answer/summary | Streamlit | misleading old model results, missing provenance | inspect payload + browser download |
| Evaluation | Recall@k, MRR, abstention, timings | evaluate.py | PDF dir, labeled queries, k, threshold | corpus, retrieval | stale labels, invalid gold pages, mixed scopes | manual small example + CLI subprocess |
| Diagnostics | Reveal active vector/index metadata without secrets | app.py | active corpus | none | settings differ from active index invisibly | nondefault runtime trace assertions |
| Unsupported features | No OCR, individual paper removal, disk persistence, auth/admin access control, factual-entailment verifier | implementation absence | n/a | n/a | resume overclaims | explicitly mark N/A / limitations, not PASS |

## Initial execution checklist

1. Save baseline unit/integration results and dependency check.
2. Reproduce suspected widget/model/default loss without API calls; record received backend parameters.
3. Create deterministic synthetic PDFs: two studies with known facts, contradictory result, 31-page text, blank, corrupt, encrypted, scanned and Unicode.
4. Exercise selected/non-default parameters and failure atomicity.
5. Inspect active runtime key presence/source without emitting values. Controlled Gemini model listing, document/query embeddings, FAISS retrieval, short QA and summary. Stop retrying auth failures. Probe local Ollama readiness; do not download models.
6. Actual browser on separate localhost tab: upload synthetic PDFs, offline analysis/Q&A, selected/all papers, settings, summary selection, downloads, reset/cancel, reload. Do not replace the user's collection.
7. Prioritize reproduced bugs, implement targeted fixes, run failing cases and full regressions.
8. Write final report with evidence locations, counts, PASS/FAIL/PARTIAL/BLOCKED/NOT TESTED, exact runtime configurations and remaining limitations.
