# PaperLens interview practice

Read the [guide](PAPERLENS_GUIDE.md) first. These are sample answers, not a script
to recite. Explain the idea, then point to the implementation if asked.

Questions marked **Next improvement** describe future work, not current features.

## A. Explain your project

### 1. What is PaperLens?

A private research-paper workspace. Users create projects, upload PDFs, ask
questions with citations, and generate saved paper summaries.

### 2. What problem does it solve?

Finding evidence across long research papers takes time. PaperLens helps readers
locate relevant passages, understand them, and open the original sources.

### 3. What makes it more than a chatbot wrapper?

It has project ownership, persistent files and chat, PDF extraction, evidence
retrieval, source validation, and failure handling. The model is one component.

### 4. Explain it in 30 seconds.

“PaperLens is a RAG-based research assistant. Google identifies users, PostgreSQL
stores their projects and conversations, R2 stores PDFs, and Pinecone searches
paper passages. OpenAI generates answers and summaries. The server attaches
filename/page citations from stored evidence.”

### 5. What is the main architectural decision?

Separate identity, durable records, original files, semantic retrieval, and
generation. Each component has a clear job and communicates through IDs.

### 6. What was a difficult bug?

PDF extraction represented an accented name differently from the model's spelling.
Exact quote validation rejected the result. We changed generation to choose
numbered evidence IDs and attach the original quote on the server.

### 7. What other practical bug did you solve?

An author question matched bibliography entries. We added a front-matter path for
recognized author/title questions, preserving ordinary semantic search for content.

### 8. What did you test?

The suite had 103 passing tests after the summary fix. It covers workflows,
ownership, PDF errors, references, provider failures, and UI behavior. We also ran
real service checks and refreshed the affected paper's summary successfully.

### 9. What should you avoid claiming?

That you trained an LLM, built Pinecone's internal search algorithm, guaranteed
factual correctness, or demonstrated production scale. Those claims are not true
of this implementation.

## B. RAG and language models

### 10. What does RAG stand for?

Retrieval-Augmented Generation: retrieve useful evidence, add it to the model's
input, and generate an answer.

### 11. Is RAG model training?

No. It supplies context at request time. Fine-tuning changes model weights;
uploading a paper into this app does not.

### 12. Why use RAG instead of asking the model directly?

The question concerns the user's specific papers. Retrieval gives the model those
papers' relevant evidence and gives us sources to display.

### 13. Why not send all PDFs every time?

It uses more tokens, increases latency, introduces unrelated material, and can
exceed context limits. Retrieval narrows the input for ordinary questions.

### 14. Does the model automatically know what is in Pinecone?

No. Our server searches Pinecone and supplies the selected text to the model.
The model does not log in to the database itself.

### 15. Can RAG hallucinate?

Yes. Retrieval may miss evidence, or the model may misinterpret it. We restrict
sources and validate references, but a valid quote does not prove every inference.

### 16. What is grounding?

Connecting an answer to supplied evidence. Our grounding mechanism is source
selection, evidence IDs, original quote attachment, and visible citations.

### 17. What is a token?

A unit of text handled by a model, often a word piece. Tokens, words, and
characters are different measurements.

### 18. What is the context window?

The amount of input and output material a model can handle in a request, subject
to its limits. Larger windows do not remove cost or retrieval-quality concerns.

### 19. Which model answers questions?

The default configured answer model is OpenAI `gpt-4.1-mini`. The server setting
can change it. It is separate from the embedding model.

### 20. Why structured output?

The app needs predictable fields such as claims and citation IDs. A schema makes
responses easier to validate, save, and display than arbitrary prose.

### 21. Does Pydantic prove an answer is true?

No. It checks data structure and types. Our additional validation checks source
references; interpreting whether the evidence truly supports a claim remains separate.

### 22. What does `store=False` mean here?

We ask OpenAI not to store the Responses resource through that option. It is not
a promise of zero provider retention, and our application still saves chat in PostgreSQL.

### 23. How do you handle prompt injection inside a PDF?

We label document text as untrusted evidence, instruct the model not to follow it,
and validate output references. These reduce risk; they are not a complete proof
of resistance to every malicious document.

## C. PDFs and embeddings

### 24. What happens when I upload a PDF?

The server checks ownership, extracts page text, chunks it, reserves a processing
record, uploads the PDF to R2, saves parsed text in PostgreSQL, indexes it in
Pinecone, and marks it ready.

### 25. Is selecting a file enough?

No. **Upload and analyze** starts the processing flow.

### 26. Why PyMuPDF?

It provides page-aware text extraction and PDF error handling. Keeping physical
page numbers makes citations possible. It does not solve OCR or every layout issue.

### 27. Why chunk text?

To search specific passages instead of treating an entire paper as one result.
It also bounds the evidence sent to the answer model.

### 28. What chunk size did you choose?

500 words with 80 words of overlap, within each page. These are defaults that
should be tuned against a realistic evaluation set.

### 29. Why overlap?

A relevant sentence may be near a boundary. Shared text helps preserve context,
but also creates additional storage and embedding work.

### 30. What if chunks are too small or too large?

Small chunks may lose context. Large chunks may mix topics and waste model input.
The best size depends on the documents, queries, and model.

### 31. What is an embedding?

A numeric representation of text that lets a search system compare semantic
similarity. It is not the final answer.

### 32. What does 1536 dimensions mean?

Each vector has 1536 numeric coordinates. It does not mean 1536 sentences or
1536 named human-readable facts.

### 33. Who creates embeddings in your current setup?

Pinecone's integrated embedding field uses `text-embedding-3-small`. Our server
sends text documents and receives an acknowledgment, not a separate vector list.

### 34. Can you create embeddings directly with OpenAI?

Yes. The supported raw-vector path calls OpenAI, receives vectors, and sends them
to a compatible Pinecone vector index. That is not the currently configured path.

### 35. Can you use one model for document vectors and another for questions?

Not arbitrarily. They must be compatible. Same-length vectors from different
models can still be in different embedding spaces.

### 36. Does changing the chat model require rebuilding embeddings?

No. Changing the embedding model usually does; changing only answer generation does not.

### 37. Do embeddings happen again on every refresh?

No. Ready document embeddings are saved in Pinecone. A new semantic question
still needs query embedding/search. A recognized metadata question skips that path.

### 38. How do you detect duplicate PDFs?

SHA-256 of file bytes plus a unique project/hash constraint. Renaming identical
bytes does not bypass duplicate detection in the same project.

### 39. Is hashing encryption?

No. A hash is a one-way fingerprint used here for equality checking. It does not
encrypt the original file.

### 40. What about scanned PDFs?

The app rejects PDFs with no extractable text. OCR is a future improvement.

### 41. What about tables or two-column papers?

Extracted reading order can be imperfect. Page-aware text extraction is useful,
but we should not claim robust table or layout understanding without evaluation.

## D. Pinecone and retrieval

### 42. Why Pinecone?

It provides managed persistent vector/search infrastructure. It avoids maintaining
a local search service in the hosted prototype, but adds network dependency and cost.

### 43. Why not FAISS?

FAISS is a useful local vector-search library. Our original dashboard used it.
The hosted workspace uses Pinecone for persistent managed search across sessions.

### 44. What is a Pinecone index?

The configured search collection and its schema/model settings. Its requirements
must match the write/search API and embedding configuration.

### 45. What is a namespace?

A partition inside an index. We create a namespace name from the project ID so
each project's searches target its own partition.

### 46. What metadata do you attach?

User ID, project ID, file ID, filename, and physical page. Documents also carry
their chunk ID and text in the configured embedding field.

### 47. Why both namespace and filters?

The namespace selects a project partition. Metadata filters and the ready-file
allowlist further narrow allowed records. Server ownership checks come before both.

### 48. Can an attacker just change the project ID?

The server checks project ID together with the authenticated user's ID. An
unowned project is rejected before search or source access.

### 49. How does selected-file search work?

The server checks that every selected ID is a ready file in that project and
uses those IDs in the search allowlist.

### 50. What comes back from search?

In our integrated path, the adapter returns matching IDs and similarity scores.
The server resolves allowed IDs to original saved text in PostgreSQL.

### 51. Why not trust returned metadata text directly?

Resolving IDs against the saved allowed chunks adds a provenance/authorization
check and makes our source text authoritative.

### 52. What is cosine similarity?

A comparison of vector direction. Similar directions generally indicate related
representations. It is not a probability that the model's answer is correct.

### 53. What are your retrieval numbers?

Ordinary questions request 18 candidates, reject scores below 0.20, and provide
at most 6 passages to generation with round-robin paper balancing.

### 54. Why those numbers?

They bound context and allow several papers to contribute. They are heuristic
defaults; I would calibrate them with labeled questions instead of claiming optimality.

### 55. Is paper balancing reranking?

It changes selection order across paper groups, but it is not a separate learned
reranker. We do not currently implement cross-encoder reranking or MMR.

### 56. Why do author questions use a different path?

References also contain author names. For recognized identity questions we use
the first two physical pages, where the actual title/byline usually appears.

### 57. What happens when no evidence is found?

The answer generator returns a no-evidence result instead of inventing an answer.
An earlier history-rewrite call may already have happened.

### 58. Why can a newly uploaded file briefly not appear in search?

Index updates are eventually consistent. A successful write acknowledgment does
not always mean an immediate search already sees the new record.

### 59. What is hybrid search? Is it implemented?

It combines lexical matching, such as BM25, with semantic retrieval. It is a
possible improvement, not the current workspace search strategy.

### 60. What is reranking? Is it implemented?

A second model scores candidate relevance more carefully after initial retrieval.
It can improve selection at additional cost/latency. We have not added it here.

## E. Answers, citations, summaries, and chat

### 61. What is sent to the answer model?

The standalone question, selected evidence passages with source information,
instructions, and a structured-output schema. We do not send Pinecone credentials.

### 62. What comes back?

Claim text with source IDs and evidence IDs. Our server attaches the actual quotes.

### 63. Why numbered evidence rather than generated quotes?

It avoids quote transcription changes involving accents, ligatures, punctuation,
or PDF artifacts. IDs can be checked and mapped to exact server-held passages.

### 64. Can a valid reference still support a bad answer?

Yes. The claim might misinterpret its passage. Reference validity and factual
entailment are different checks.

### 65. How many repair attempts do you allow?

One application-level repair after schema/reference failure. SDK retries for
eligible API/transport failures are separate. Refusals/incomplete responses fall back.

### 66. Why not retry forever?

It wastes time and quota and may repeat the same error. Bounded repair followed
by an honest fallback gives predictable behavior.

### 67. Is the displayed fallback an AI summary?

No. It is explicitly labeled source text/excerpts. We should not present it as a
successful synthesized answer or summary.

### 68. How is a full-paper summary different from question answering?

A question retrieves selected evidence. A summary reads all saved page text in
batches, generates structured findings, and combines them with page-specific evidence.

### 69. Why summary batches?

To bound model input while covering the whole paper. The batch limit is 10,000
text characters, not 10,000 tokens or a fixed number of pages.

### 70. Does your summary skip late pages?

The batching loop covers all nonempty page text, splitting long pages as needed.
There are regression tests for findings on page 31.

### 71. What if one summary batch fails?

The current implementation returns a whole-summary excerpt fallback. If refreshing
a previously successful generated summary fails, the old successful one is retained.

### 72. How do follow-up questions work?

When generation is on, a rewrite call can turn the latest question and bounded
recent history into a standalone search question. We retrieve fresh paper evidence.

### 73. Why not use previous AI answers as evidence?

They might contain mistakes. We use conversation to resolve references, while
source evidence comes from the papers.

### 74. Where is chat saved?

In PostgreSQL `pl_messages`, linked to a project. Assistant payloads include
claims, citation/source snapshots, retrieval question, and execution settings.

### 75. Is there a separate chat per paper?

Not by default. There is one conversation per project, with optional file scope.
The UI displays/exports the latest 100 messages; the default storage cap is 1000.

### 76. What happens to old answers when a PDF is removed?

Their source snapshots remain in history. Opening the removed original is no
longer available. Removal is not a complete historical-content erasure workflow.

## F. Google authentication and browser memory

### 77. OAuth or OIDC: which one are you using?

Google OIDC through Streamlit's native login. OIDC adds identity to the OAuth 2.0
framework. Our app delegates the protocol flow to the integration.

### 78. Explain authentication versus authorization.

Authentication identifies the user. Authorization determines which project/file
that user may access. Google login does not replace our database ownership checks.

### 79. Explain the login flow simply.

Browser goes to Google, Google signs the user in, browser returns a temporary code
to our callback, the server integration exchanges it and validates identity, and
Streamlit establishes its identity cookie. We then find/create the database user.

### 80. Do you store Google passwords?

No. The user authenticates with Google. Our app stores a subject identity and
profile fields, not their Google password.

### 81. Why use Google's `sub` instead of email as the identity?

`sub` is Google's stable account subject. Email/profile details can change. We
map `google:<sub>` to our own permanent internal user ID.

### 82. What are client ID and client secret?

They identify and authenticate our registered OAuth application in the protocol.
The secret is server configuration, not a user's password and not browser UI data.

### 83. What is a redirect URI?

The registered callback address Google is allowed to return to. A mismatch can
break login. Localhost and the hosted domain need the correct registered URI.

### 84. What is an ID token versus an access token?

An ID token asserts identity for the application. An access token authorizes
requests to permitted APIs. They are not interchangeable credentials.

### 85. What is a JWT?

A token format with a header, payload, and cryptographic protection, commonly a
signature. Google's OIDC ID token is a JWT. Our app does not build custom JWT middleware.

### 86. What are issuer, audience, and expiry?

Who issued the identity token, which application it targets, and when it expires.
The OIDC integration handles relevant validation; we do not just trust decoded JSON.

### 87. What do state and nonce do?

They help bind responses to an initiated login flow and protect against replay
or flow substitution. Recognize the concepts; do not claim you wrote that library code.

### 88. How does my browser remember me?

It stores Streamlit's signed identity cookie and sends it on matching requests.
Streamlit recognizes the identity; PostgreSQL maps it to saved projects.

### 89. Are projects stored in the cookie?

No. The cookie represents login identity. Project records and chat are in
PostgreSQL; PDFs are in R2; searchable vectors/documents are in Pinecone.

### 90. Does a signature encrypt the cookie?

No. Signing detects tampering. Encryption hides content. They are different.
Never assume a signed value is safe to share or that it cannot be stolen.

### 91. How long is login remembered?

Streamlit documents a 30-day identity-cookie lifetime. Browser behavior and
deployment/framework configuration can affect the observed result. We did not
implement a custom session-expiry or refresh-token system.

### 92. What happens on logout?

The framework clears the browser identity cookie and starts a new app session.
Saved project data remains. It is not a global Google logout or all-device revocation.

### 93. Why can Google sign me back in quickly after logout?

Google has its own separate browser session. Clearing the app's cookie does not
necessarily clear Google's cookie.

### 94. What happens in another tab or device?

A new tab has its own UI session and can share the site's cookie. A different
device needs authentication again, then can load the same account's saved projects.

### 95. What if someone steals a valid cookie?

It may enable impersonation while accepted. Use HTTPS, appropriate framework
cookie protections, careful secret handling, and stronger session controls for a
larger deployment. A signature prevents modification, not copying.

### 96. What do HttpOnly, Secure, and SameSite mean?

HttpOnly limits JavaScript access; Secure limits sending to HTTPS; SameSite
controls cross-site cookie sending. The framework manages these attributes;
verify the actual deployment rather than guessing them.

### 97. Does Google login grant Gmail/Drive access?

Not in this application. We request identity login, not a Drive/Gmail integration.

## G. Sessions, SQL, files, and safety

### 98. Is `st.session_state` the same as login?

No. It is per-UI-session Python state for selected project and widgets. Login is
the framework's identity cookie. Durable records are another separate layer.

### 99. What happens when the page fully refreshes?

Temporary UI state may reset. The cookie can restore identity, and saved data
reloads from PostgreSQL/R2/Pinecone. PDFs are not re-embedded automatically.

### 100. What does `st.cache_resource` cache?

The repository/engine resource, shared in the process. It is not a global list of
one user's private documents. Each relevant query checks the user/project scope.

### 101. Why PostgreSQL instead of NoSQL?

Users, projects, files, and messages have clear relationships and benefit from
constraints and transactions. JSONB handles flexible nested evidence data.

### 102. What is SQLAlchemy Core versus psycopg?

SQLAlchemy builds queries and manages connections/transactions. psycopg is the
driver that communicates with PostgreSQL.

### 103. How do you prevent SQL injection?

We use SQLAlchemy expressions and bound values instead of building SQL strings
from user input. Ownership checks address a different problem: unauthorized access.

### 104. What is a transaction?

A group of database operations committed together or rolled back. We save a
question and its answer in one database transaction.

### 105. Why use row locks?

They serialize relevant decisions such as checking a project's file count and
reserving a file. They do not lock all three external services into one transaction.

### 106. Why R2 rather than storing PDF bytes in PostgreSQL?

R2 is designed for objects and direct file delivery. PostgreSQL keeps file
metadata and relationships. Storing small PDFs in a DB is possible, but it changes
backup, database-size, and delivery tradeoffs.

### 107. Is an R2 object key a public URL?

No. It is the object's storage address inside a bucket. Our bucket is private;
an authorized user gets a short-lived signed download URL.

### 108. What is a presigned URL?

A temporary signed capability for a specific operation/object. Our GET links
expire after 300 seconds. Someone with the valid link may use it until expiry.

### 109. Does the browser receive the R2 secret key?

No. The server signs the URL. The browser receives that temporary URL, not the
credential used to create it.

### 110. Can one database rollback undo a Pinecone write?

No. These services do not share one transaction. We use processing/ready/failed/
deleting states, ready-file filtering, and cleanup retries.

### 111. What if embedding/indexing fails halfway?

The file becomes failed and is excluded from searches. Partial external data can
be removed through the cleanup flow before retrying.

### 112. What if the server crashes during upload?

A processing record can remain. Durable background jobs and stale-work recovery
are future improvements, not implemented guarantees.

### 113. What if deletion partly fails?

The file stays deleting and out of new searches. Its record remains so external
cleanup can be retried rather than pretending deletion fully completed.

### 114. Is PostgreSQL row-level security enabled?

No. Ownership is enforced by the application repository methods. Database RLS
could provide another layer later.

### 115. Where are the secrets?

Local .env or host environment variables. The launcher generates ignored
Streamlit authentication configuration. Real secrets are excluded from Git and
the Docker build context; .env.example contains placeholders only.

## H. Performance, evaluation, deployment, and future work

### 116. What costs API quota?

Indexing/embeddings, semantic queries, question rewriting, answer generation,
summary batches, and eligible repairs/retries. Ordinary browser refresh does not
regenerate saved document embeddings.

### 117. How would you measure retrieval quality?

Use labeled question/source pairs and metrics such as recall@k and reciprocal
rank. Also test unsupported questions and inspect irrelevant retrieved passages.

### 118. How would you measure answer quality?

Review whether claims answer the question, preserve numbers/conditions, and are
actually supported by citations. A correct JSON schema is not an accuracy metric.

### 119. What is recall@k?

How much expected relevant material appears within the first k results. If two
of four gold sources appear, recall for that case is 0.5.

### 120. What is reciprocal rank?

One divided by the position of the first relevant result. First place gives 1;
fourth place gives 0.25. Mean reciprocal rank averages across questions.

### 121. Do 103 tests mean 100% real-world accuracy?

No. Automated tests cover selected behavior, often with mocked services. Real
model quality needs representative live evaluation and human source review.

### 122. How would you speed uploads? — Next improvement

Use background workers with durable jobs, progress updates, and safe retries.
Tune bounded batching and concurrency without exceeding provider quotas.

### 123. How would you scale retrieval? — Next improvement

Optimize candidate retrieval and database reads, evaluate index configuration,
and test realistic concurrent workloads. Keep project isolation throughout.

### 124. How would you control costs? — Next improvement

Track per-user usage, enforce budgets/rate limits, avoid duplicate indexing,
bound context, and evaluate the benefit of additional model calls.

### 125. Why not PostgreSQL pgvector instead of Pinecone?

It could combine relational and vector storage in one system. That reduces
services but requires selecting and operating suitable vector indexes and capacity.
Pinecone was chosen for managed search; pgvector is a reasonable alternative.

### 126. Why Streamlit instead of React and FastAPI?

It lets us build and validate a Python-based prototype quickly. A larger product
could benefit from a dedicated frontend/API, streaming UX, and worker architecture.
Those would be future changes, not the current stack.

### 127. What does Docker provide?

A repeatable server package with dependencies. It does not automatically create
the database, configure OAuth, provide API keys, or make the app publicly hosted.

### 128. What does hosting require?

A long-running server with WebSocket support, environment secrets, the assigned
port, HTTPS, and a correctly registered Google callback. External services retain
research data across app restarts.

### 129. What are your main production gaps?

Background jobs/recovery, migrations, backups, full privacy erasure, spending
controls, representative quality evaluation, and load/security testing.

### 130. What would you say if you do not know an implementation detail?

“That is handled by the framework/service. I understand its role, but I would
check its documentation before claiming the internal algorithm.” That is better
than guessing or presenting a managed service's internals as your own work.

## A mock interview you can practise

Have a friend ask these, without showing you the answers:

1. Explain PaperLens in one minute.
2. Where does my PDF go after upload?
3. What exactly do you send to Pinecone in the configured mode?
4. How is a question answered, and how are projects filtered?
5. Why can you not claim citations guarantee correctness?
6. How does Google login become your internal database user?
7. What survives a page refresh, app restart, or logout?
8. Describe the accent-related failure and the fix.
9. Explain a partially failed upload or deletion.
10. What would you improve before a large public launch?

Aim for clear 30–60 second answers. Use one example. Stop when the question is
answered; let the interviewer choose the next level of detail.
