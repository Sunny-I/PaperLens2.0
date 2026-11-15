# Verified initial defects and targeted fix order

Baseline: existing 42 tests pass, but additional state sequences fail. Evidence: initial-reproductions.json. Inventory and traces were reviewed before changes.

1. HIGH — Chat model resets to default after Gemini → Ollama → Gemini; embedding model becomes empty after hiding its widget. Streamlit removes widget-owned state when the widget is absent. Preserve provider-specific values in separate non-widget state, including explicit empty strings; regression: set nondefault values, switch away/back, inspect actual provider calls.
2. HIGH — User threshold 0.17 becomes 0.30 after replacing a lexical corpus with a neural corpus. Widget identity/default changes. Use stable key/default initialized once; preserve explicit zero and arbitrary user values. Show active index vs requested settings rather than silently using a different-looking configuration.
3. MEDIUM — Gemini embedding exceptions erase error class, HTTP status and failed-batch position. Generation categorizer exists but isn't used for embeddings. Preserve safe status/category/action and batch position, without keys or arbitrary provider text; do not swap embedding modes/models.
4. MEDIUM — Empty paper summary reports generated with zero batches. Require nonempty source text, otherwise return labeled fallback with warning and no provider request.
5. MEDIUM — Chunk size 1.5 fails with TypeError and True is accepted; top_k has analogous type gaps. Validate positive integer units and finite scalar thresholds explicitly.
6. MEDIUM — Neural embedding matrix with a zero row and another valid row is indexed. Reject individual zero neural document vectors atomically; retain lexical zero-row behavior for stop-word-only chunks and no-evidence query semantics.
7. MEDIUM — CLI demo labels reference PDFs no longer in the actual sample folder. Evaluation reports meaningless zero retrieval metrics for missing gold sources. Validate reference filenames/pages and ship a deterministic fixture-generation/evaluation command.
8. Investigate before changing — live embedding batch failures, credentials precedence, stale generation provenance, SDK retry behavior, ambiguous/irrelevant QA and quote-vs-claim factual support. Do not label speculative causes fixed.

No credential edits, provider/model substitutions, deployment changes or removal of features. UI remains simple; diagnostics expose nonsecret active settings.
