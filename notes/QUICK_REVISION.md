# PaperLens: quick revision before an interview

## My 45-second explanation

> PaperLens helps users understand research PDFs through private projects. It uses
> Google OIDC login, PostgreSQL for projects and conversations, R2 for original
> PDFs, and Pinecone for semantic search. In the current setup, Pinecone creates
> embeddings using text-embedding-3-small. OpenAI generates answers and summaries
> from numbered evidence. The server attaches original quotes, filenames, and
> pages. Saved data survives refreshes because it lives in external storage.

## Remember the jobs

| Component | Job |
| --- | --- |
| Google/Streamlit OIDC | Identity |
| Streamlit session state | Temporary UI state |
| PostgreSQL | Durable records, source text, chat, summaries |
| R2 | Original PDF bytes |
| Pinecone | Embeddings/search in the configured integrated mode |
| OpenAI | Question rewriting, answers, summaries |
| Pydantic | Output structure validation |
| Our evidence lookup | Reference validation and original quote attachment |

## Five flows to memorize

```text
LOGIN
Google -> validated identity -> framework cookie -> Google sub -> internal user ID

UPLOAD
Check owner -> parse/chunk/hash -> processing record -> R2 -> saved text
-> Pinecone indexing -> ready

QUESTION
Check owner/ready files -> optional history rewrite -> project-scoped retrieval
-> allowed source text -> numbered evidence -> OpenAI -> checked IDs -> saved chat

SUMMARY
All saved pages -> bounded batches -> numbered evidence -> OpenAI findings
-> original page quotes -> combine/deduplicate -> save

REOPEN
Cookie restores identity -> PostgreSQL finds saved projects/chat
-> R2 and Pinecone already retain files/search data
```

Author/title questions use front matter instead of ordinary semantic retrieval.

## Numbers from the current code

| Setting | Value |
| --- | --- |
| PDF chunk size / overlap | 500 words / 80 words |
| Configured embedding dimensions | 1536 |
| Ordinary search candidates / final passages | 18 / up to 6 |
| Ordinary similarity cutoff | 0.20; not a confidence probability |
| Evidence passage size | Around 650 characters |
| Summary batch size | Up to 10,000 text characters |
| History rewrite context | Latest 6 messages, up to 3000 characters each |
| App-level validation repair | At most 1 |
| Private PDF link lifetime | 300 seconds |
| Default PDF/project limits | 20 MB/file; 30 files/project |
| Default project/message limits | 20 projects/user; 1000 messages/project |
| Displayed/exported chat | Latest 100 messages |
| Last complete test run after summary fix | 103 passed |

## Things I must not mix up

- RAG is retrieval plus generation; it is not fine-tuning.
- An embedding model creates vectors; an answer model writes prose.
- Current integrated Pinecone mode receives text. Raw-vector mode receives vectors.
- Authentication identifies me; authorization controls my projects.
- Google subject, our user ID, project ID, and file ID are different IDs.
- Login cookie, temporary session state, and saved data are different kinds of memory.
- Signing detects changes; encryption hides contents.
- A valid citation proves its source exists, not that every interpretation is true.
- A hash detects identical bytes; it does not encrypt the file.
- A local server can still use cloud APIs. Local does not mean offline.
- A database transaction does not roll back external R2/Pinecone writes.

## My strongest debugging example

**Problem:** author queries returned bibliography names; accent transcription caused
quote validation failures.

**Fix:** select front-matter evidence for identity questions; use numbered evidence
IDs and server-attached quotes. Apply the same reference approach to summaries.

**Proof:** the live author question succeeded, the affected 25-page summary was
saved as generated with five evidence-backed sections, and regression tests passed.

## Honest limitations

No OCR, no learned reranker, no background queue, no complete privacy-erasure
workflow, no per-user spending budgets, and no universal accuracy/scale claim.

## A five-minute demo

1. Sign in with Google and create a project.
2. Upload a small text-based PDF with a known fact.
3. Ask about that fact; open its filename/page/quote citation.
4. Ask a follow-up, then ask who wrote the paper.
5. Generate a summary and inspect its evidence.
6. Refresh to show saved data returns.
7. If available, use another account to demonstrate project separation.

Use synthetic or public demo papers. Never expose .env, private URLs, cookies, or
API keys during screen sharing.

## If I forget everything else

**Find evidence first. Answer second. Attach the original source. Check ownership
at every project boundary. Save durable data outside temporary sessions.**
