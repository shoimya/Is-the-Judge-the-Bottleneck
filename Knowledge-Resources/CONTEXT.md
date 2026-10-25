# Is the Judge the Bottleneck?

A study of self-correcting retrieval loops for multi-hop question answering. It measures how much accuracy and cost an LLM sufficiency judge loses compared with an oracle judge, and whether the loss comes from stopping or from steering.

## Loop

**Round**:
One retrieve-then-judge cycle. A question gets at most 3 Rounds.
_Avoid_: iteration, hop (a hop is a reasoning step in the question, not a loop cycle)

**Evidence**:
The paragraphs retrieved for a question across all Rounds so far.
_Avoid_: context (alone), documents

**Gold paragraph**:
A paragraph the dataset marks as needed to answer a question.
_Avoid_: supporting fact (HotpotQA's term for sentence-level labels), ground truth

## Roles

**Judge**:
The role that reads the question and the Evidence and emits a Stop decision and a Steer signal.
_Avoid_: critic, evaluator, verifier

**Stop decision**:
The Judge's yes/no verdict that the Evidence is enough to answer now.
_Avoid_: sufficiency label, halt

**Steer signal**:
The Judge's free-text statement of what information is missing. Only the Rewriter reads it.
_Avoid_: feedback, critique

**Rewriter**:
The role that turns the question and the Steer signal into the next retrieval query.
_Avoid_: query generator, reformulator

**Answerer**:
The role that produces the final answer from the question and the Evidence.
_Avoid_: generator, reader

**Oracle judge**:
A Judge that sees the Gold paragraphs. Its Stop decision is "yes" once every Gold paragraph is in the Evidence; its Steer signal is the title of one missing Gold paragraph.
_Avoid_: gold judge, perfect judge

**LLM judge**:
A Judge implemented by prompting a language model, with no access to Gold paragraphs.
_Avoid_: deployed judge, model judge

## Conditions

**Condition A / B / C / D**:
The four cells of the 2x2 that cross who makes the Stop decision (LLM or oracle) with who produces the Steer signal (LLM or oracle). A = LLM both, B = oracle stops, C = oracle steers, D = oracle both.

**Single-turn**:
Reference point that retrieves once and answers immediately, with no Judge.

**Always-loop**:
Reference point that always runs 3 Rounds, using the LLM judge's Steer signal and ignoring its Stop decision.

**Answer-oracle**:
Reference point where the Answerer answers after every Round and a question counts as correct if any Round's answer is correct.

## Judge evaluation

**Coverage key**:
Answer key that marks a Stop decision correct when every Gold paragraph is in the Evidence.

**Answerability key**:
Answer key that marks a Stop decision correct when the Answerer's answer on the current Evidence is correct.

**Pilot set**:
The 100 questions used for tuning prompts and settings, never reported as results.
_Avoid_: dev set (ambiguous with the dataset's own dev split)

**Test set**:
The 1,000 questions used for reported results, run only after prompts and settings are frozen.
