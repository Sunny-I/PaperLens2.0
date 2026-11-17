# PaperLens, explained simply

## 1. What did I build?

PaperLens is a research-paper workspace. A user signs in with Google, creates a
project, uploads PDFs, and asks questions about those papers. Answers show the
supporting filename, page, and quote. The user can also generate paper summaries.

Example: Maya creates **Cancer Research**, uploads three papers, and asks:

> Which method achieved the highest accuracy, and what were its limitations?

PaperLens searches those papers and asks an AI model to explain the relevant
passages. Maya can open each source to check the answer.

It solves a practical problem: finding and comparing information across long PDFs
takes time. It helps the reader locate evidence and understand it.

It does not guarantee that every AI interpretation is correct.

### A short answer you can say in an interview

> I built PaperLens, a project-based research assistant using RAG. Users sign in
> with Google and organize papers into private projects. PostgreSQL stores users,
> projects, extracted text, and conversations. R2 stores the PDFs, and Pinecone
> retrieves relevant passages. OpenAI generates answers and summaries. The server
> attaches citations from stored evidence, so users can verify the original pages.

## 2. The whole system in one picture

```text
                          GOOGLE
                        confirms identity
                              |
USER'S BROWSER <----> STREAMLIT APP SERVER
                              |
             +----------------+------------------+
             |                |                  |
         POSTGRESQL           R2              PINECONE
       users/projects      original PDFs    searchable passages
       text/chat/summary                    and embedding vectors
                              |
                    Server calls OPENAI
                  for answers and summaries
```

The browser shows the interface. The Python server performs database operations,
PDF processing, and provider calls. Users do not install Python on their computers
when the app is hosted.

The server keeps API keys. We do not send those keys to the browser.

## 3. What each technology does

| Tool | Simple meaning | Its job in PaperLens |
| --- | --- | --- |
| Python | Programming language | Implements application behavior |
| Streamlit | Python web-app framework | Pages, widgets, chat display, session state, Google login integration |
| PyMuPDF | PDF-reading library | Extracts page text and page numbers |
| PostgreSQL | Relational database | Stores durable records and their relationships |
| SQLAlchemy Core | Python database toolkit | Builds queries, manages connections and transactions |
| psycopg | PostgreSQL driver | Carries database requests between Python and PostgreSQL |
| Cloudflare R2 | S3-compatible object storage | Keeps the original PDF bytes |
| boto3 | S3-compatible Python SDK | Uploads, deletes, and signs access links for R2 objects |
| Pinecone | Managed search/vector service | Stores indexed passages and finds similar evidence |
| text-embedding-3-small | Embedding model | Represents text as vectors; configured dimension is 1536 |
| OpenAI gpt-4.1-mini | Default answer model | Generates structured answers and summaries |
| Pydantic | Typed data validation | Checks the shape and field types of model responses |
| Authlib | Authentication protocol library | Used by Streamlit's OIDC login integration |
| python-dotenv | Configuration loader | Loads local .env settings into the server environment |
| tiktoken | Token-counting library | Checks embedding input size |
| Docker | Application packaging | Packages the server and dependencies for hosting |

The original dashboard also uses FAISS, TF-IDF, LangChain, and other model
providers. Those remain in `legacy_app.py`. Do not describe that older path as the
current PostgreSQL/R2/Pinecone workspace.

## 4. What is RAG?

**RAG = Retrieval-Augmented Generation.**

In normal language: **find useful source material, then let AI answer using it.**

An open-book exam is a useful comparison. Retrieval finds the right book pages.
Generation writes an explanation based on those pages.

```text
Retrieve: find relevant passages
Augment:  add those passages to the AI's input
Generate: ask the AI to write a grounded answer
```

Uploading PDFs does not train or fine-tune the model. The model's weights do not
change. We provide relevant material when we make a request.

Why not send every paper with every question? It increases cost, latency, and
irrelevant context. It can also exceed model input limits. Retrieval narrows the
material before answering. A whole-paper summary is a different flow: we process
all pages in bounded batches.

## 5. Follow one PDF from upload to storage

Suppose Maya uploads `study.pdf` to project `p1`.

```text
Browser sends PDF bytes to the app server
  -> check the signed-in user's ownership of p1
  -> validate and extract the PDF
  -> split page text into chunks
  -> calculate the file's content hash
  -> reserve a PostgreSQL file record: processing
  -> upload original PDF bytes to private R2
  -> save extracted pages/chunks in PostgreSQL
  -> index passages in the project's Pinecone namespace
  -> mark the PostgreSQL file record: ready
```

The user must click **Upload and analyze**. Selecting a file alone does not
complete processing.

### PDF extraction

PyMuPDF opens the PDF and extracts text from each page. We keep physical PDF page
numbers starting at 1. These can differ from printed page labels: a cover sheet
counts as physical page 1.

A corrupt, password-protected, empty, or textless PDF produces an error. There is
no OCR implementation. A scanned picture of a page needs text recognition first.
Complex columns, tables, and fonts can also reduce extraction quality.

### Chunks: why divide the text?

A chunk is a smaller piece of a paper. Searching smaller pieces helps locate the
specific paragraph relevant to a question.

Our defaults are **500 words per chunk, with 80 words of overlap**. These are
words, not tokens. Each chunk stays on one page.

```text
Chunk 1: words    1–500
Chunk 2: words  421–920
Chunk 3: words  841–1340
```

The overlap preserves context near chunk boundaries. It increases the number of
stored words and vectors, so it is a tradeoff, not free accuracy.

We keep a chunk's ID, file ID, filename, page, and text. In the workspace, chunk
IDs use the reserved file ID and a chunk number.

### Duplicate detection

SHA-256 computes a fingerprint of the PDF bytes. The combination of project ID
and content hash is unique in PostgreSQL.

Uploading the same bytes under a different filename in the same project is still
a duplicate. Different PDFs with the same filename can coexist. The same file can
belong to separate projects. This is not global cross-user deduplication.

The hash is a fingerprint, not encryption. It does not hide or encrypt the PDF.

## 6. Embeddings and vectors

An embedding is a list of numbers representing aspects of text's meaning.

For illustration only:

```text
"A dog runs"       -> [0.2, 0.8, -0.1]
"A puppy is fast"  -> [0.3, 0.7, -0.2]
"Database backup"  -> [-0.6, 0.1, 0.9]
```

Related text tends to have related vectors. The real configured vectors have
**1536 numbers**, not three. A dimension is one position in that list. The
coordinates are learned features; they are not named facts such as age or height.

An embedding model produces vectors. A chat model writes an answer. They do
different jobs.

### Our CURRENT configured embedding path

Your Pinecone index has an integrated embedding field named `chunk_text`, using
`text-embedding-3-small`. The app sends text documents to Pinecone. Pinecone runs
the configured embedding model and indexes the results.

```text
App -> Pinecone: chunk text + IDs + metadata
Pinecone: create embeddings and index documents
Pinecone -> app: operation acknowledgment
```

The application does not receive a separate vector list in this mode. Pinecone
manages that step. This embedding usage is handled through the Pinecone service;
we do not also call OpenAI's embedding endpoint for those same chunks.

The separate OpenAI API key is used for answer and summary generation.

### The other supported embedding path

If `PINECONE_EMBED_FIELD` is empty and the index accepts raw vectors:

```text
App -> OpenAI embeddings API: list of chunk texts
OpenAI -> app: list of 1536-number vectors
App -> Pinecone: vectors + metadata
```

The code supports both paths. Explain which one is configured before explaining
what is sent. Integrated document indexes and raw-vector indexes require different
Pinecone operations; using the wrong write API was a real issue we diagnosed.

### Why keep the same embedding model?

Questions and document chunks must use compatible embeddings: the same model and
dimensions. Two different models can produce vectors of the same length but with
different meanings. Matching dimensions alone is insufficient.

Each project stores its embedding model and dimensions. Changing the answer model
does not require re-embedding PDFs. Changing the embedding model requires a
compatible index and rebuilding the affected document embeddings.

## 7. Why three storage services?

| Place | Stored data | Example |
| --- | --- | --- |
| R2 | Original PDF bytes | `users/u1/projects/p1/f1.pdf` |
| PostgreSQL | Relationships, source text, summaries, chat | f1 belongs to p1; p1 belongs to u1 |
| Pinecone | Searchable documents/vectors and metadata | A passage from f1, physical page 5 |

R2 is like a file cupboard. PostgreSQL is the organized register. Pinecone is the
meaning-based search system.

We deliberately also keep extracted pages and chunks in PostgreSQL. When
Pinecone returns a chunk ID, the app resolves it against our saved source text.
It does not blindly display whatever text an external search response contains.

This duplicates some text across services. It simplifies provenance in this
prototype, at the cost of storage and larger database reads.

### PostgreSQL tables

```text
pl_users
  id, identity, email, name, created_at

pl_projects
  id, user_id, name, namespace,
  embedding_model, embedding_dimensions, created_at

pl_files
  id, project_id, filename, content_hash, r2_key,
  status, size_bytes, total_pages, chunk_count,
  paper, summary, error, created_at

pl_messages
  id, project_id, role, content, payload, created_at
```

```text
One user -> many projects
One project -> many files
One project -> many messages
```

`paper`, `summary`, and `payload` use PostgreSQL JSONB for nested data. We can use
relational tables for ownership and flexible JSON for citations and parsed pages.
We do not need a separate NoSQL database just because a user has a list of projects.

A primary key identifies a row. A foreign key connects it to a parent row. A
unique constraint prevents duplicate values or combinations. An index helps
queries locate rows efficiently; it is different from a Pinecone search index.

SQLAlchemy Core constructs queries and transactions. psycopg communicates with
PostgreSQL. We are not using SQLAlchemy ORM entity classes in this implementation.

## 8. Follow one ordinary question

Maya asks: **What accuracy did the method achieve?**

1. The server checks that Maya owns the project.
2. It loads ready files, applying any selected-file restriction.
3. If AI generation is on and history exists, it may rewrite the question using
   the latest six messages, taking at most 3000 characters from each message.
4. In the current integrated mode, it sends the standalone question, namespace,
   and filters to Pinecone. Pinecone embeds the query and searches.
5. Pinecone returns candidate IDs and similarity scores.
6. The server resolves IDs against allowed saved chunks in PostgreSQL.
7. It drops candidates below 0.20 and balances passages across papers, choosing
   at most six passages from the top 18 search candidates.
8. It divides the chosen text into numbered evidence passages, around 650
   characters each, ending at word boundaries when possible.
9. It sends the question, evidence, filenames/pages, instructions, and output
   schema to OpenAI.
10. OpenAI returns structured claims with source and evidence IDs.
11. The server checks IDs and attaches the corresponding original quotes.
12. PostgreSQL saves the question, answer, source snapshots, and execution details.
13. The browser displays the answer and source cards.

### An example Pinecone filter

This is an illustrative request body, not your real IDs:

```json
{
  "namespace": "project-p1",
  "filter": {
    "$and": [
      {"user_id": {"$eq": "u1"}},
      {"project_id": {"$eq": "p1"}},
      {"file_id": {"$in": ["f1", "f2"]}}
    ]
  }
}
```

The project ID comes from an owner-checked PostgreSQL record. The browser cannot
choose an arbitrary namespace and gain access to it.

### Index, namespace, metadata, and top-k

- **Index:** the configured Pinecone search collection/service endpoint.
- **Namespace:** a partition inside that index. We use one per project.
- **Metadata:** descriptive fields such as user, project, file, filename, and page.
- **top-k:** how many search results we request. Our ordinary search requests 18
  candidates and later supplies up to 6 passages to the answer model.

Namespaces help separate projects. They are not a replacement for authorization.
Pinecone is accessed using a server API key, so the application must choose the
correct namespace and check ownership first.

### Similarity is not confidence

Cosine similarity compares the direction of two vectors. Its formula is:

```text
cosine(a,b) = dot_product(a,b) / (length(a) * length(b))
```

For example, a score of 0.8 does not mean an answer is 80% correct. The 0.20 cutoff
is an implementation choice that needs evaluation with representative questions.

Our balancing step gives multiple relevant papers a chance to contribute. It is
not a learned reranker, hybrid search, or MMR implementation.

### Special case: authors and titles

A bibliography contains many author names. Semantic search can therefore return
references when the user wants the uploaded paper's own authors.

For recognized author/title questions, we use stored physical pages 1 and 2,
bounded to 10,000 characters per page, from the allowed ready files. This path
skips Pinecone search and document/query embedding calls. A history rewrite can
still happen first. It uses the same answer and citation checks afterward.

The intent detector is a heuristic, including the typo `authers`. It does not
understand every possible phrasing, language, or unusual cover-page layout.

## 9. What exactly is sent and returned?

| Operation | Sent from our server | Returned |
| --- | --- | --- |
| R2 upload | Original PDF bytes and object key | Upload acknowledgment |
| PostgreSQL write | Records, extracted text, chat, or summary | Row/operation result |
| Pinecone integrated indexing | Text, chunk ID, user/project/file metadata | Indexing acknowledgment |
| Pinecone integrated search | Question text, namespace, ownership/file filters, top-k | Matching IDs and scores |
| Direct OpenAI embedding mode | Chunk texts or one question; embedding model/dimensions | Numerical vectors |
| Follow-up rewrite | Latest question and bounded recent conversation | Standalone question |
| AI answer | Question and selected numbered evidence with source information | Claims and reference IDs |
| AI summary | Bounded batches of numbered page evidence | Structured findings and reference IDs |
| PDF open | Browser requests a temporary signed R2 URL | Original PDF bytes |

We send extracted text to model/search services, not just vectors in all modes.
Do not say the app is completely offline or that uploaded content never leaves
the server. Google login does not automatically give the model access to Gmail,
Drive, other projects, or the browser's files.

## 10. Citations, summaries, and follow-up questions

### How citations work now

The model might return:

```json
{
  "claims": [{
    "text": "The method achieved 92% accuracy.",
    "citations": [{"source_id": 1, "evidence_id": 2}]
  }]
}
```

The server looks up evidence 2 in source 1. It attaches the original quote,
filename, and page. Unknown source/evidence IDs are rejected.

This avoids asking the AI to retype unusual PDF characters perfectly. We
previously encountered a real failure where a PDF-extracted accent differed from
the AI's spelling. Numbered evidence fixed the transcription problem.

But a real quote can still be misinterpreted. These checks establish provenance,
not logical entailment or guaranteed factual correctness.

### What if model output is invalid?

The Responses API requests a Pydantic schema. We then validate source references.
Schema/reference failures get at most one application-level repair attempt.
The repair includes the rejected structured answer when available and safe
validation feedback. Refusals or incomplete responses do not enter this repair
loop. If generation fails, the app clearly shows source excerpts instead.

The OpenAI SDK also has one configured retry for eligible transport/API failures.
That retry is separate from the application's validation repair. Do not promise
that every question always costs exactly one API request.

We set `store=False` on Responses requests. This does not prove there is zero
provider logging or retention under every policy. It also does not stop our own
application from saving chat history in PostgreSQL.

### How a full-paper summary works

```text
Load the saved paper pages from PostgreSQL
  -> divide all page text into batches of up to 10,000 text characters
  -> divide each batch into numbered evidence passages
  -> ask OpenAI for structured findings with page/evidence references
  -> validate references and attach original quotes on the server
  -> deduplicate identical finding text and combine findings across batches
  -> save the result in PostgreSQL
```

The five sections are problem, methodology, datasets/experiments, findings, and
limitations. Missing evidence stays unknown. Long pages are split; later pages
are not silently dropped. This path does not retrieve only six Pinecone chunks.

Several batches mean several model requests. Findings are combined with exact-text
deduplication, not a separate final LLM consolidation pass. Distinct findings can
therefore make a long summary. If any batch fails, the current implementation
falls back for the whole summary. A failed refresh preserves a prior successful
generated summary instead of overwriting it.

### How follow-ups work

Question 1: **Which method performed best?**

Question 2: **What are its limitations?**

The rewrite step uses recent conversation to resolve “its,” then searches the
papers again. Previous AI answers help interpret the question but are not passed
as source evidence for the new answer. We do not send the entire saved history
on every request. When generation is off, that AI rewrite is also skipped.

## 11. Google login: authentication versus authorization

**Authentication:** Who are you?

**Authorization:** Are you allowed to access this project?

Google answers the first question. Our ownership checks answer the second.

We use Streamlit's native **OpenID Connect**, or OIDC, integration. OIDC is built
on OAuth 2.0 and adds identity information. Saying “Google OAuth login” informally
is common; saying “Google OIDC through Streamlit” is more precise.

### Login flow

```text
Click Continue with Google
  -> browser goes to Google's login/consent page
  -> Google authenticates the user
  -> browser returns to our registered /oauth2callback with a temporary code
  -> the server-side integration exchanges the code with Google
  -> the integration validates identity information
  -> Streamlit establishes its signed identity cookie
  -> our app reads the authenticated claims through st.user
  -> our database finds or creates the corresponding user record
```

The OIDC library handles protocol details such as checking identity-token
signatures and relevant claims. We delegate the flow rather than building our own
token-verification middleware. Concepts to recognize are issuer (who issued it),
audience (which app it is for), expiry, state (login-request correlation), and
nonce (replay protection). Do not claim a custom PKCE implementation in this app.

These are the general protocol concepts described in [Google's OIDC guide](https://developers.google.com/identity/openid-connect/openid-connect).

### Credentials and identity fields

| Item | Meaning |
| --- | --- |
| Google client ID | Identifies our registered application |
| Google client secret | Server credential used by the login integration |
| Redirect URI | Allowed address Google returns the browser to |
| ID token | Google's identity assertion; in OIDC it is a JWT |
| Access token | A credential for calling permitted APIs; not the same thing as an ID token |
| `sub` | Google's stable subject identifier for the user |
| Our user ID | Internal random ID for database relationships |
| AUTH_COOKIE_SECRET | Server secret used by Streamlit to protect its cookie |

We store `google:<sub>` as a unique identity, rather than using changeable email
as the permanent identity. An existing PostgreSQL row is updated for profile
changes, while its internal user ID is retained.

We do not collect Google passwords, implement password reset, or intentionally
expose provider tokens to the app UI. We do not use this login to request Drive
or Gmail access. Google sign-in alone is not authorization to a private project.

## 12. How does the browser remember me?

A **cookie** is a small value the browser stores for a website and sends back on
matching requests. Think of the authentication cookie as a signed entry pass.

Streamlit verifies its signature using the server cookie secret. Signing helps
detect tampering; it is not the same as encrypting a value. The browser does not
need the server secret to send its cookie back.

Streamlit documents an identity-cookie lifetime of 30 days. Browser privacy
settings, cookie clearing, private browsing, deployment settings, and framework
version behavior can change the observed experience. We did not implement our
own 30-day session database or rotating refresh-token system. See [Streamlit authentication](https://docs.streamlit.io/develop/concepts/connections/authentication).

On a later visit, the cookie lets Streamlit recognize the user. The Google `sub`
maps to the same PostgreSQL user, which leads to the saved projects. Papers are
not stored inside the cookie.

`st.logout()` clears the browser's identity cookie and starts a new app session.
It does not delete papers, sign the user out of Google everywhere, or implement
global revocation of every already-open session/device. New sessions may also
sign in quickly because the separate Google website still remembers the user.

Use HTTPS in production. HttpOnly restricts JavaScript access to a cookie,
Secure restricts transmission to HTTPS, and SameSite affects cross-site sending.
The framework manages cookie attributes; inspect the deployed browser/server
configuration before making blanket claims about every environment.

Never print, share, or paste real authentication cookies. A stolen valid cookie
can be dangerous even when its contents are signed.

## 13. There are THREE different kinds of memory

| Kind | Where it lives | What it remembers |
| --- | --- | --- |
| Login identity | Browser cookie, checked by Streamlit | Who is signed in |
| UI session state | Streamlit server session | Selected project, widgets, temporary upload state |
| Durable application data | PostgreSQL, R2, Pinecone | Projects, files, vectors, conversations, summaries |

`st.session_state` survives normal Streamlit script reruns. It is tied to the
WebSocket session, not permanent browser storage. A full reload or new tab can
start a new session. [Streamlit session-state documentation](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)

Saved project data survives because the app reloads it from the external services,
not because a temporary Python dictionary survived.

| Action | Expected result |
| --- | --- |
| Click an ordinary widget | Script reruns; UI session values generally remain |
| Fully refresh | Temporary UI state can reset; saved library/chat can reload |
| Open another tab | A separate UI session; it can share the site's login cookie |
| Restart the app | RAM state disappears; external saved data remains |
| Clear browser cookies | Login must be re-established; saved project data remains |
| Use another device | Sign in again; the same Google identity can load the same projects |
| Delete a file in the app | Its search records and PDF are cleaned up; old answer snapshots can remain |

The chosen project is currently temporary UI state, not a saved cross-device
preference. After a fresh session, the app can choose the first available project.

### What is cached?

`st.cache_resource` shares the Repository/SQLAlchemy engine resource in the server
process. This avoids creating a new connection pool on every UI rerun. It is not
a shared cache of one user's papers. User/project queries still check ownership.

## 14. How we keep projects separate

Imagine two users: Maya (`u1`) and Ravi (`u2`). Maya owns `p1`.

Before listing files or history, searching, summarizing, or opening a PDF, the
repository checks the combination:

```sql
SELECT * FROM pl_projects
WHERE id = :project_id AND user_id = :signed_in_user_id;
```

If no row matches, access is denied. Guessing `p1` is insufficient. We also check
that a file belongs to the requested project.

Search then uses the owned project's namespace, user/project metadata filters,
and an allowlist of ready file IDs. Selected-file search narrows that allowlist.
Returned chunk IDs must also exist among the allowed saved chunks.

These checks are in the application. PostgreSQL row-level security is not
configured here. We also do not have sharing roles or organization membership.
Never describe UI hiding or random IDs as the primary security boundary.

SQLAlchemy parameter binding keeps values separate from SQL syntax. It helps
prevent SQL injection when used correctly; it does not replace ownership checks.

## 15. Opening and removing a PDF

### Private download

1. Check user, project, file ownership, and ready status.
2. Generate an R2 presigned GET URL valid for 300 seconds.
3. Let the browser download/open the PDF using that URL.

The link contains a temporary signature, not the R2 secret key. It is still a
temporary bearer capability: someone with the valid link may use it before it
expires. It is not an extra Google-login check enforced by R2. We do not save
these signed URLs in chat history. `#page=5` asks the PDF viewer to show page 5;
viewer support can vary. [Cloudflare's presigned URL documentation](https://developers.cloudflare.com/r2/api/s3/presigned-urls/)

### Removal

```text
Check owner/file
 -> mark deleting in PostgreSQL
 -> delete this file's Pinecone records in its project namespace
 -> delete its R2 object
 -> remove its PostgreSQL file record
```

If external cleanup fails, keep the deleting record so removal can be retried.
It is excluded from ready-file retrieval immediately. A currently processing
file cannot be removed through this flow.

Old messages retain their historical quote snapshots. A link to a removed source
becomes unavailable. Removing a PDF is not the same as purging every historical
mention or backup of its text.

## 16. Failures, transactions, and honest limitations

A database transaction commits related SQL writes together or rolls them back.
The question and answer are saved in one PostgreSQL transaction. Row locks help
serialize project/file-limit decisions for concurrent operations.

However, PostgreSQL, R2, and Pinecone do not share one transaction. We cannot
pretend one SQL rollback undoes an R2 upload. Our prototype uses processing states
and retryable cleanup instead of a distributed transaction system.

```text
processing -> ready
processing -> failed
ready/failed -> deleting -> record removed
```

Only ready records enter search. A failed upload can leave partial external data,
but it is excluded from retrieval and can be removed before retrying. A server
crash can leave a processing record requiring operator recovery. Processing is
synchronous; there is no durable background-job queue or automatic recovery worker.

Other limitations you should admit:

- No OCR, robust table understanding, or guaranteed extraction of complex layouts.
- No automatic factual-entailment checker for every generated claim.
- No published accuracy or load-test benchmark for the hosted workspace.
- No production billing, per-user spending budget, or comprehensive rate limiter.
- No full-text/vector hybrid fusion or learned reranker in the main workspace.
- No organization sharing, project-deletion workflow, or full privacy-erasure workflow.
- No migration framework yet: initial tables are created with `metadata.create_all`.
- Some JSON source data is loaded on reruns; this should be optimized for larger libraries.
- Pinecone indexing is eventually consistent; a just-written document may need a
  short wait before search sees it.

## 17. Local, hosted, and offline are different

**Local:** the app server runs on your laptop. It can still call cloud APIs.

**Hosted:** the same server runs on an online host. Other people use a browser.

**Offline:** the feature makes no internet requests. The current Google/Pinecone/
OpenAI workspace is not an offline app. Turning off AI answers still leaves
semantic-search API calls for ordinary questions.

The legacy dashboard supports offline TF-IDF retrieval and optional local Ollama.
FAISS stores/searches vectors locally in that old dashboard; it is not the current
Pinecone workspace's vector store.

For hosting, the Dockerfile packages Python, code, and dependencies. It does not
contain .env, .git, the local virtual environment, or real Streamlit secrets.
Supply credentials through host environment variables. The launcher generates
the ignored OIDC secrets file and starts Streamlit. The host needs WebSocket
support, the correct port, and the registered HTTPS callback URL.

External research data does not depend on the app container's disk. The runtime
still needs to be able to write its generated authentication configuration. This
project includes hosting preparation; do not claim a public production deployment
has happened unless you have actually deployed and verified one.

## 18. Cost, performance, testing, and next steps

### What costs money or quota?

Document embedding/indexing, query embedding/search, answer generation, optional
question rewriting, summaries, and storage. Repairs/retries can add requests.
Refreshing a page reloads saved records; it does not re-embed ready PDFs. Rebuilding
an index, removing/re-uploading a PDF, or refreshing a summary performs work again.

Avoid quoting fixed prices: provider pricing and account plans change. Check the
provider dashboards when estimating actual costs.

### What we tested

The current suite has **103 passing automated tests** at the summary fix. It uses
mocked external responses and isolated relational test databases. Those tests
cover parsing, retrieval, ownership, state, citations, summaries, provider errors,
and UI behavior. SQLite is a test double in relevant tests, not the runtime database.

We also performed live service checks and synthetic upload/search/answer/summary
checks. The actual affected 25-page paper completed a saved generated summary
without a validation warning. Tests establish the behavior they exercise, not
universal accuracy, unlimited scalability, or a security certification.

The older `pipeline.evaluate` CLI evaluates labeled retrieval with recall@k,
reciprocal rank, abstention, and latency. It is not a benchmark of the current
Pinecone workspace unless adapted to that path and run on representative data.

### How I would improve it next

1. Add background processing, progress persistence, retry jobs, and stale-upload recovery.
2. Create a representative question set; measure retrieval and answer support separately.
3. Calibrate chunk size, candidate count, and score cutoff against that set.
4. Try hybrid search and reranking, comparing quality, cost, and latency.
5. Optimize database reads, add migrations, and plan backups/deletion policies.
6. Add spending limits, rate limiting, monitoring, and realistic load testing.
7. Add OCR and better table/column parsing only after evaluating their benefits.

These are proposed improvements, not features to claim already exist.

## 19. Where to look in the code

| Question | File |
| --- | --- |
| How does the UI work? | `app.py`, `workspace/ui.py` |
| How is Google configured? | `workspace/auth.py`, `scripts/run.py` |
| How do we load .env? | `workspace/settings.py` |
| Where are tables and ownership checks? | `workspace/database.py` |
| Where is the upload/question/summary flow? | `workspace/service.py` |
| Where are R2/Pinecone/embedding requests? | `workspace/providers.py` |
| How is PDF text/chunking done? | `pipeline/pdf_parser.py` |
| How are answer citations resolved? | `pipeline/answer_generator.py` |
| How are summary batches/citations handled? | `pipeline/summarizer.py` |
| How are OpenAI schemas, repairs, and safe errors handled? | `pipeline/openai_client.py`, `pipeline/output_validation.py` |
| How do I check configured services? | `scripts/check_connections.py` |
| How do I run an opt-in live smoke test? | `scripts/smoke_workspace.py --run` |
| Where is the old FAISS dashboard? | `legacy_app.py` |

## 20. Practise these explanations without looking

1. Draw one upload and name what each service receives.
2. Explain an ordinary question and an author question. Why are they different?
3. Explain why refreshing does not erase saved projects.
4. Explain how a different Google user is blocked from your project.
5. Explain what a citation proves and what it does not prove.
6. Explain the accent-related bug and how numbered evidence solved it.
7. Explain one honest limitation and a practical next improvement.

If you can explain these calmly, you understand the core project.
