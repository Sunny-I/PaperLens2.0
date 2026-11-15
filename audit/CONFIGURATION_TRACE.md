# Configuration trace — 2026-11-15

No credential values recorded. Runtime verification means assertions on backend arguments/output, not merely displayed widgets. Models containing explicit/selected are synthetic test values with mocked transports, not claims those models exist.

| Input/source | Trace and validation | Nondefault → actual runtime / evidence | Default/lifetime |
|---|---|---|---|
| Upload PDFs | UploadedFile → getvalue() → ingest → PDFParser → SHA256 ID/page chunks | Browser Orion+Lyra →2 papers/4 chunks; downloaded Orion citation p2 | Analyze disabled without files; atomic commit |
| Start a new collection | Checkbox → replace → copied or empty dictionaries → rebuilt store | True replaces, False appends; duplicate/failed-batch tests | False; applies only on analysis |
| Summarize on upload | Checkbox → generate_summaries → client → summarize_corpus after indexing | AppTest true creates all summaries with mocked transport; false creates excerpts | False; independent of Generate AI answers |
| Provider | Selectbox → get_client → GeminiClient/OllamaClient → SDK | Gemini explicit model IDs asserted at provider methods; Ollama endpoint/model at embedding constructor | Gemini initially; no automatic substitution |
| Chat model | persistent_text → provider-specific saved_settings/widget → trim → AnswerGenerator/PaperSummarizer → structured → LangChain | gemini-selected-chat reaches structured; custom Gemini/Ollama IDs survive switches; explicit empty remains empty/errors | Env GEMINI_CHAT_MODEL/OLLAMA_CHAT_MODEL seeds first value; absent Gemini env uses gemini-3.8-flash; absent Ollama env empty. Gemini strips models/ prefix, rejects embedding IDs |
| Search method | Selectbox → analysis branch → None/ProviderEmbeddings → FaissVectorStore | TF-IDF vocabulary; semantic A used for docs/query after editing to B; next analysis builds B | TF-IDF initially; analysis required to change active index |
| Embedding model | Persistent field → trim/nonempty check → frozen ProviderEmbeddings(client,model) → docs/query | embedding-selected-A used for both after B selected; live gemini-embedding-001 produced3072 dimensions | Env GEMINI_EMBED_MODEL/OLLAMA_EMBED_MODEL seeds, otherwise empty. Hidden values survive mode switches |
| Ollama URL | Persistent field seeded by OLLAMA_BASE_URL → client → rstrip('/') → SDK | http://localhost:11555/ → http://localhost:11555; timeout19/model asserted at SDK constructor | localhost:11434. Index retains endpoint; pending-change label detects provider/model/mode, not endpoint changes |
| API key | Environment before Streamlit secrets; GEMINI_API_KEY before GOOGLE_API_KEY within source → resolver → strip → SecretStr → SDK google_api_key,vertexai=False | Synthetic env GOOGLE overrides secret GEMINI; empty higher-priority env errors. Real project key authenticated listing/embeddings | Required for remote operations. No .env loader. Global/project secrets merging delegated to Streamlit |
| Similarity | Stable slider key → threshold → search(min_score) → cosine filter | 0.23 in execution config;0.17/exact0 persist across index changes;CLI0.07 | Initial0.05,step0.01; zero valid; bool/NaN/infinity/outside0..1/non-numeric rejected |
| Generate AI answers | Checkbox → client/model supplied iff true | False no structured call, browser Source excerpts; true nondefault model reaches mock | True; does not disable embedding APIs/manual summaries |
| Scope/chosen papers | All/Selected → IDs → search(doc_ids) | Browser Orion-only excludes Lyra; empty Selected warns/removes old answer | All; no silent all-papers fallback |
| Question | Form → nonempty check → search → JSON question in prompt | Orion→p2; xyzzyquux→no evidence; ambiguous/conflict/80k-character/instruction-like inputs preserved in captured prompts | Empty rejected; no max length, provider limits apply |
| Summary selection/buttons | ID → [selected] or None/all → summarize_corpus → selected chat | AppTest one/all/refresh; browser excerpts/export | Previous successful summary retained with warning if refresh fails; model edits do not regenerate |
| Clear | Confirm → new Corpus/pop answer/increment upload version | Browser cancel retained2; confirm removed2/reset uploads | Settings survive clear; full reload loses session |
| Samples | Button → folder bytes → ingest | Browser5 original files/pages in browser-samples.txt; no fictional A/B | sample_papers; missing/empty errors, no substitute generation |
| Export | Saved answer/config → JSON; current schema → summary JSON | Browser files parsed: Orion85%,p2,offline provider=None | Answer provenance retained; summary lacks model/provider metadata |

## Backend/CLI parameters (not UI fields)

| Parameter | Trace/default | Nondefault/boundary execution |
|---|---|---|
| chunk_size/overlap | PDFParser per-page word windows500/80 | 7/2 with lengths0/6/7/8/17→0/1/1/2/3 chunks; coverage/page3 retained; bool,float,string,nonpositive size,overlap<0 or>=size rejected |
| max_features | FaissVectorStore→TfidfVectorizer8192→dense float32 | Explicit3 caps dimension3; positive integer check |
| top_k | UI default6; Python/CLI→capped results | 1/2/4/9 with6 candidates→1/2/4/6;CLI2;invalid types rejected |
| balance_papers | True round-robin relevant papers;False score order | Existing balance test; explicitFalse with top_k; differs from global score rank |
| doc_ids | None all/list filters candidates | Selected-paper browser and six-paper tests |
| summary max_chars | page_batches10000characters; helper only | Explicit200 covers802chars losslessly; malformed types/<200 rejected;31-page full coverage test |
| timeout | Gemini/Ollama120sec→HTTP milliseconds/Ollama seconds | Constructor17→17000ms;Ollama19→19;live direct30sec diagnostic timed out |
| retries/options | Gemini temperature0,max_retries1;both schemas at most2 attempts;Ollama context16384/output4096 | Real LangChain/mocked transport bounded repair and connection failure; fixed code options |
| evaluation paths | argparse→PDF bytes→offline ingest→gold labels | 2PDFs/3cases,top_k2/threshold0.07;missing filename/page errors;no answer correctness metric |

## Migration and limits

Conditional fields use separate saved_settings; old same-name keys migrate if available. Already lost values cannot be recovered. Invalid explicit selections stay visible for correction.

Indexes freeze provider/model/client. Chat edits apply immediately to next generation; embedding/endpoint edits require analysis. Key rotation does not mutate existing clients. No global application cache or disk index reuse found.

Generation failure produces labeled excerpts and warning, never another LLM. Embedding failure preserves the old corpus, never switches to TF-IDF. Exact citation matching does not establish semantic entailment.
