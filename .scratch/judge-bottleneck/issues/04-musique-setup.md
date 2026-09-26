# How is MuSiQue set up, and how do others search it?

Type: research
Status: resolved
Blocked by: none

## Question

Are MuSiQue's unanswerable questions only in MuSiQue-Full, so that MuSiQue-Answerable has none? Where is it downloaded, what are its fields (paragraphs per question, `is_supporting` labels, decomposition sub-questions, hop counts), and is the dev split's gold data public? How do iterative-retrieval papers (for example IRCoT, Adaptive-RAG) search MuSiQue: one pooled corpus of all paragraphs, or each question's own candidate paragraphs? How big is the pooled corpus?

Feeds the map's fog on the MuSiQue search setting and backlog [T14](../../../backlog/T14-musique-data-and-corpus.md).

## Answer

- **Use MuSiQue-Ans.** Unanswerable questions exist only in MuSiQue-Full, so the answerable version has none.
- **Sample from dev.** Test-split labels aren't public (leaderboard only), so the Pilot set and Test set both come from dev: 2,417 answerable questions (1,252 two-hop, 760 three-hop, 405 four-hop). Each has 20 paragraphs, with one `is_supporting` paragraph per hop.
- **Search setting: one pooled corpus** of all paragraphs from all six files (139,416 paragraphs, ~64 MB), as IRCoT and Adaptive-RAG do. With only each question's 20 paragraphs, 3 Rounds retrieve most of them and the Judge has little to decide.
- **Titles aren't unique.** 34% of dev Gold paragraphs share their title with another paragraph, and 116 questions have two Gold paragraphs with the same title. So the Oracle judge must check coverage by **paragraph identity** (a hash of title + text), not title. The Steer signal can still name the title.
- **3 Rounds is tight for 4-hop.** A rough oracle-loop BM25 test found all Gold paragraphs within 3 Rounds for 76% of 2-hop, 47% of 3-hop and 17% of 4-hop questions. Keep 3 Rounds (more is out of scope) and report every MuSiQue result by hop count.
