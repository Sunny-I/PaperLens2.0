# PaperLens hosted workspace

The default `app.py` is now the persistent, Google-authenticated workspace.
The original session-only dashboard remains available as `legacy_app.py`.

## 1. Fill `.env`

Copy `.env.example` to `.env` if needed. The local `.env` has already been created;
do not overwrite populated values. It and `.streamlit/secrets.toml` are ignored by Git
and excluded from the Docker build context. Never commit credentials.

- **PostgreSQL:** `DATABASE_URL`, e.g. `postgresql://user:encoded-password@host:5432/db?sslmode=require`.
  URL-encode reserved characters in usernames/passwords. Use the database provider's
  recommended TLS configuration. The database role needs permission to create the
  `pl_users`, `pl_projects`, `pl_files`, and `pl_messages` tables on first startup.
- **R2:** S3 endpoint, bucket name, and an access key/secret with object read/write
  permissions for that private bucket. This is the R2 S3 key, not a Cloudflare global API key.
- **OpenAI:** API key; defaults are `gpt-4.1-mini` and `text-embedding-3-small`.
  Embedding dimensions default to **1536**. Model access and billing must be enabled.
- **Pinecone:** an existing **dense-vector index, cosine metric, 1536 dimensions**;
  copy its API key and data-plane index host, not its console URL. This app supplies
  vectors itself; do not select an integrated-embedding-only index. The REST adapter
  targets API version `2026-07`. Changing dimensions requires a compatible index.
- **Google:** OAuth client of type **Web application**, client ID and client secret.
  Add `http://localhost:8501/oauth2callback` to authorized redirect URIs. Add your
  Google account to OAuth test users if the consent app is in testing. For hosting,
  set `AUTH_REDIRECT_URI=https://your-domain/oauth2callback` and register that URI too.
  Access the local app at `http://localhost:8501` to match the configured callback.
- Keep the generated `AUTH_COOKIE_SECRET` stable, private, and at least 32 characters.

For an existing Pinecone **document index with integrated embedding**, set
`PINECONE_EMBED_FIELD` to its embedding text field (e.g. `chunk_text`) and
`PINECONE_EMBED_MODEL` to its exact model. Keep `OPENAI_EMBED_MODEL` equal to that
model so projects record the correct search configuration. In this mode Pinecone
creates document and query embeddings and bills its model usage; the app does not
also call OpenAI's embedding endpoint. OpenAI still generates answers/summaries.
Document-index operations use Pinecone's current Documents API. Leave the field
empty to use direct OpenAI embeddings and the vector API.

## 2. Start

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/check_connections.py
.venv\Scripts\python scripts/run.py
```

The launcher loads `.env`, writes the native Streamlit Google OIDC configuration to
the ignored secrets file, and starts the app. Restart after editing credentials.
Open http://localhost:8501. Sign in, create a project, and upload a text-based PDF.

Google sign-in is mandatory. There is no development login bypass in the app.
Model-list checks establish access only; they do not establish generation quality.

## 3. What is stored

| Service | Data |
| --- | --- |
| PostgreSQL | Google subject identities, project ownership, embedding configuration, file records, extracted pages/chunks, summaries, chat and citation snapshots |
| Private R2 bucket | Original PDFs under opaque user/project/file keys |
| Pinecone | Vectors and chunk metadata in a distinct namespace for each project |

Each operation verifies project ownership on the server. Pinecone search uses the
owned namespace plus user/project filters and a ready-file allowlist. Returned
chunk IDs resolve to authoritative text in PostgreSQL. Partial uploads are marked
failed and excluded from retrieval. Remove a failed record to clean up partial
R2/Pinecone writes, then upload again. Removing files is confirmed in the UI and
retryable if external cleanup fails. Old answers retain citation text; links to
removed files stop working. A PDF link is private and expires in five minutes.

Reopening or refreshing loads saved records without regenerating document
embeddings. Each new question calls the embedding API. Follow-up questions can
also use a bounded context-rewrite call before fresh retrieval. Prior AI answers
are not supplied as source evidence. One project has one saved conversation;
the latest 100 messages are displayed/exported, up to the configured storage limit.

## 4. Hosting

Deploy the Dockerfile to a host supporting a long-running Streamlit server and
WebSockets. Set the environment variables in the host's secret manager, including
`BIND_ADDRESS=0.0.0.0`, the assigned `PORT`, and the HTTPS OAuth callback. If the host
uses a port other than 8501, set it explicitly. The reverse proxy must forward
WebSocket requests. Keep the same cookie secret on every instance.

Application data lives in PostgreSQL, R2, and Pinecone, so the hosted app does not
need a persistent local disk for research data. The runtime must permit writing
`.streamlit/secrets.toml`; the provided container user owns that directory.

## 5. Validation and limits

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```

The workspace tests use an isolated SQLite database solely as a relational test
double and mocked provider transports. Runtime accepts only PostgreSQL.
Legacy UI regressions explicitly target `legacy_app.py`.

Default limits: 20 MB/PDF, 30 files/project, 1,500 chunks/file, 20 projects/user,
and 1,000 messages/project. Processing is synchronous; keep the page open until
uploads finish. An interrupted server can leave a `processing` record that needs
operator recovery. This version does not include OCR, background jobs, billing,
organization sharing, or per-user API spending budgets. Avoid an unrestricted
public launch until usage and provider quotas are monitored.

Exact quote checks verify provenance, not whether a claim logically follows from
the quote. Validate important interpretations against the original paper.

The project chat uses numbered evidence passages: the model selects source and
passage IDs, and the server attaches the original source quote. This avoids PDF
accent/ligature transcription errors while rejecting unknown references. Author
and title questions use the saved first two physical pages rather than semantic
bibliography matches. Older saved answers are retained; ask again to use the new
retrieval and citation flow. Light and dark appearances use native Streamlit theme
palettes; choose your appearance in the app menu's Settings. Restart the server
after changing `.streamlit/config.toml`.

Provider references: [Streamlit OIDC](https://docs.streamlit.io/develop/concepts/connections/authentication),
[R2 with boto3](https://developers.cloudflare.com/r2/examples/aws/boto3/),
[Pinecone vectors](https://docs.pinecone.io/reference/api/latest/data-plane/upsert),
[OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings).
