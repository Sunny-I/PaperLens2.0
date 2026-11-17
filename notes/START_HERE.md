# Learn PaperLens for your interview

You do not need to memorize the code. Learn what happens to one PDF and one question.

Read these in order:

1. [Simple project guide](PAPERLENS_GUIDE.md) — understand the system and its flows.
2. [Interview questions](INTERVIEW_QUESTIONS.md) — practise explaining your decisions.
3. [Quick revision](QUICK_REVISION.md) — revise just before the interview.

## A small learning plan

| Session | What to do |
| --- | --- |
| 1: 30 minutes | Read guide sections 1–5. Draw the upload flow without looking. |
| 2: 30 minutes | Read sections 6–10. Explain one question, filtering, citations, and summaries aloud. |
| 3: 30 minutes | Read sections 11–14. Explain Google login, the browser cookie, and refresh behavior. |
| 4: 30 minutes | Read the remaining sections. Practise failures, tradeoffs, and improvements. |
| 5: 30 minutes | Answer the interview questions aloud. Mark the questions you cannot explain. |
| Before the interview | Read the quick revision sheet and perform the demo. |

If a word is unfamiliar, stop and explain it using the example in the guide.
Do not learn an answer that you cannot explain in your own words.

## The one sentence to remember

**Google identifies the user; PostgreSQL remembers the workspace; R2 keeps PDFs;
Pinecone finds evidence; OpenAI explains it; our server attaches the citations.**

## What these notes describe

These notes follow the actual project code as of 17 November 2026. The current
configured Pinecone index uses integrated embeddings. The guide also explains the
supported direct-OpenAI embedding path and the older FAISS dashboard separately.
Values described as defaults can change through server configuration.

Examples use made-up users, IDs, filenames, and numbers. There are no real keys,
account IDs, database URLs, or private document contents in these notes.
