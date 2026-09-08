# Working agreement

## What this project is
AI Support Ticket Intelligence — support ticket triage, classification,
severity prediction, summarization, and RCA assistance. Python/FastAPI.

Current state: a working classical ML priority classifier (TF-IDF +
logistic regression, trained on 12k tickets) plus two local Hugging Face
pipelines (sentiment, summarization). There is no LLM anywhere in the
codebase yet.

## Goal of this branch
Add an LLM path to a working classical app, then hold the result to a
production bar: measured results against a labeled evaluation set,
benchmarked against the existing classical baseline, validated structured
outputs, tested guardrails, and an explicit human-escalation path.

The bar: someone technical browsing this repo should see real
engineering judgment, not a tutorial. Numbers, not claims. I should be
able to defend every decision in it under interview questioning.

## How I want you to work with me — this overrides your defaults

Speed matters — I have two to three days. But I must be able to explain
every decision in an interview.

1. Work in chunks of one module at a time, not one whole feature.
2. Before each chunk, give me 3-5 sentences: what it does, why this
   approach, what the alternative was and why you rejected it.
3. After each chunk, list the 3 decisions in it I'd most likely be asked
   about in an interview, with the answers. Append these to
   INTERVIEW_NOTES.md as we go.
4. Don't wait for me to confirm understanding — keep moving unless I
   stop you. If I say "slow down", drop to one function per step.
5. No placeholder code, no TODOs, no stubbed functions. Each chunk must
   run and be testable before we move on.
6. If I ask you to do something that's a bad idea, say so directly.
7. Prefer boring, readable code over clever code. I need to be able to
   explain every line of this to an interviewer.

## Priorities, in order
1. Labeled evaluation set — nothing else is measurable without it
2. An LLM path for the same priority-prediction task, benchmarked
   against the existing TF-IDF + logistic regression baseline on
   identical held-out data
3. Pydantic-validated LLM outputs with retry on parse failure
4. Guardrail tests: adversarial cases asserting no invented case
   details, plus a confidence threshold that escalates to a human
5. Everything else is optional

## Things I don't want
- Do not refactor files I didn't ask about.
- Do not add dependencies without telling me what they do and why a
  stdlib approach won't work.
- Do not write the README — I write that myself, at the end.
- Do not build deployment infrastructure (Kubernetes, cloud hosting,
  frontends). Out of scope.
