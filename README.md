# tech-doc-rag

A retrieval-augmented generation (RAG) assistant that answers support
questions about the public IONOS Cloud documentation
([docs.ionos.com](https://docs.ionos.com)), with citations to the
pages it used. Personal portfolio project, not affiliated with IONOS.

Every change to the system is measured against a fixed evaluation set
of 50 questions, so each improvement can be shown with numbers.

## Phase 1 results: naive RAG (vector search)

Setup: 439 chunks from 109 documentation pages, OpenAI
`text-embedding-3-small` embeddings in Chroma, top 5 chunks by vector
similarity, answers by `gpt-5.4-mini` (temperature 0, prompt v1).

Scores are the mean of two runs of all 50 questions, graded by the
same judge (Claude Sonnet 5.5) and answer key.

### Overall

| Metric | What it measures | Score |
| --- | --- | --- |
| hit@1 | the right chunk is the top search result | 67% |
| hit@5 | the right chunk is in the top 5 | 95% |
| MRR | how high the right chunk ranks on average (1.0 = always first) | 0.767 |
| Answer recall | share of the answer key's facts found in the answer | 0.85 |
| Faithfulness | share of the answer's claims supported by the retrieved chunks | 0.98 |
| Context recall | share of the answer key's facts present in the retrieved chunks | 0.95 |
| Required terms | answers that contain the exact identifiers asked for | 91% |
| Refusal accuracy | unanswerable questions correctly declined (8 questions) | 88% |
| False refusal rate | answerable questions wrongly declined (42 questions) | 5% |
| Cost per answer | answer LLM only | $0.0023 |
| Latency per answer | median | 1.2 s |

### By question type

| Type | Questions | hit@1 | hit@5 | Answer recall |
| --- | --- | --- | --- | --- |
| api_fact | 8 | 62% | 75% | 0.70 |
| exact_identifier | 7 | 29% | 100% | 0.95 |
| how_to | 7 | 100% | 100% | 0.84 |
| faq | 6 | 67% | 100% | 0.93 |
| paraphrased | 6 | 83% | 100% | 0.77 |
| multi_chunk | 5 | 40% | 100% | 0.86 |
| definition | 3 | 100% | 100% | 0.94 |
| unanswerable | 8 | – | – | 88% declined |

### What phase 1 gets wrong

- **Ranking of exact identifiers:** for error codes, parameter names
  and similar identifiers, the right chunk is first only 29% of the
  time (it is almost always in the top 5).
- **Missed evidence:** two answerable questions (q34, q41) never
  retrieve the right chunk, so the bot says the docs don't cover them.
- **Vague questions:** "What are the two defaults of that?" gets a
  guessed answer instead of a request for clarification.

Phase 2 targets these with keyword search (BM25), hybrid search, a
reranker and stricter prompts, each measured against this baseline.

### How the scores were checked

- **Run-to-run noise:** search and refusal scores were identical in
  both runs. Answer recall and faithfulness moved by 0.01–0.02, so
  later changes smaller than about 0.03 overall (0.10 per question
  type) are treated as noise.
- **Validating the judge:** reading the judge's decisions found two
  problems with the off-the-shelf RAGAS factual correctness metric.
  It scored correct answers 0.00 when they added extra correct details
  (its "recall" counts only the answer's claims that the answer key
  supports), and it never sees the question. The project replaces it
  with answer recall: each fact of the answer key, split one fact per
  claim, is checked against the question and answer together. Six
  answer key entries were corrected at the same time (wrong
  cause-and-effect links, details the questions did not ask for). For the same run 1 answers, the score
  went from 0.77 (RAGAS metric, old answer key) to 0.84 (answer recall).
- **Known limitations:** the judge still misjudges a few answers, for
  example when the answer key names a product the question does not
  mention. These are left as they are, because every phase is graded by
  the same judge and answer key.
