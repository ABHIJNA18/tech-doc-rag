# Evaluation summary: phase1_vector_v1_run1

Graded 50/50 questions (42 answerable, 8 unanswerable). Judge: claude-sonnet-5-5.

| Metric | Score |
| --- | --- |
| hit@1 | 67% |
| hit@3 | 88% |
| hit@5 | 95% |
| MRR | 0.767 |
| Evidence recall | 0.940 |
| All evidence found (multi-fact questions) | 83% |
| Faithfulness (RAGAS) | 0.985 |
| Answer recall (golden facts found) | 0.839 |
| Context recall (RAGAS) | 0.946 |
| Required terms present | 91% |
| Cites an evidence chunk | 95% |
| Refusal accuracy (unanswerable) | 88% |
| False refusal rate (answerable) | 5% |
| Answers with invalid citation numbers | 0 |

## Cost and latency

- Answer LLM: $0.00228 per answer, latency p50 1.264s, p95 6.073s
- Judge: claude-sonnet-5-5 (effort low), $2.0678 for 276 calls (91.0s); calls per model: {'claude-sonnet-5-5': 276}

## By question type

| Type | n | hit@1 | hit@5 | MRR | Faithfulness | Answer recall | Declined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| api_fact | 8 | 62% | 75% | 0.688 | 1.000 | 0.705 | 25% |
| exact_identifier | 7 | 29% | 100% | 0.524 | 1.000 | 0.936 | 0% |
| how_to | 7 | 100% | 100% | 1.000 | 0.964 | 0.834 | 0% |
| faq | 6 | 67% | 100% | 0.756 | 1.000 | 0.917 | 0% |
| paraphrased | 6 | 83% | 100% | 0.889 | 1.000 | 0.805 | 0% |
| multi_chunk | 5 | 40% | 100% | 0.633 | 0.975 | 0.808 | 0% |
| definition | 3 | 100% | 100% | 1.000 | 0.917 | 0.943 | 0% |
| unanswerable | 8 | – | – | – | – | – | 88% |

Declined: share of answers that said the docs do not cover it (good for unanswerable questions, a false refusal otherwise).

## Questions that lost marks

- q01 (exact_identifier): evidence at rank 3; answer recall 0.80; missing terms ['RATE_LIMIT_EXCEEDED']
- q02 (api_fact): evidence at rank 2
- q03 (exact_identifier): evidence at rank 2
- q08 (how_to): answer recall 0.67; faithfulness 0.83
- q09 (api_fact): answer recall 0.50
- q12 (how_to): faithfulness 0.92
- q13 (multi_chunk): evidence at rank 3; answer recall 0.88
- q15 (definition): faithfulness 0.75
- q19 (unanswerable): answered instead of declining; faithfulness 0.40
- q23 (exact_identifier): evidence at rank 4
- q26 (faq): evidence at rank 5
- q27 (paraphrased): evidence at rank 3
- q28 (multi_chunk): evidence at rank 3; evidence 1/2; answer recall 0.56
- q31 (exact_identifier): evidence at rank 3; answer recall 0.75
- q32 (exact_identifier): evidence at rank 4
- q34 (api_fact): evidence not retrieved in top 5; evidence 0/1; false refusal; answer recall 0.00
- q36 (how_to): answer recall 0.50
- q37 (faq): evidence at rank 3
- q38 (paraphrased): answer recall 0.33
- q39 (multi_chunk): answer recall 0.60
- q41 (api_fact): evidence not retrieved in top 5; evidence 0/1; false refusal; answer recall 0.14; missing terms ['1024x1024', '1024x1792', '1792x1024']
- q43 (how_to): answer recall 0.67
- q44 (faq): answer recall 0.50
- q45 (paraphrased): answer recall 0.50
- q46 (multi_chunk): evidence at rank 2; faithfulness 0.88
- q47 (definition): answer recall 0.83
- q50 (unanswerable): faithfulness 0.80
