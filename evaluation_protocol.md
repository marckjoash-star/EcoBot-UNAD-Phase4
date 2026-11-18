# EcoBot: evaluation protocol — Exercise 3

**Purpose.** Evaluate environmental factuality, clarity, usefulness, retrieval grounding, conversational context and abstention behavior. Perform **at least 10** queries in the running web app and record the observed answers.

**Scoring.** Give 1–5 points separately for relevance, clarity, usefulness and source support. Mark `pass=1` only when (i) the response addresses the request or appropriately abstains, (ii) factual statements are supported by the linked source material, and (iii) it contains no material false claims. Otherwise, mark `pass=0` and explain why. **Success rate = 100 × sum(pass) / number of evaluated tests.** This is a proposed operational metric, not a validated scientific benchmark.

**Human feedback.** Invite peers or volunteers to test the app. Ask them to evaluate clarity, usefulness and confidence using the separate blank template. Participation should be voluntary and should avoid personally identifying data.

**Protocol.**
1. Record the deployed or local URL, execution date, runtime type, model version and dataset revision.
2. Start a fresh session for independent questions. For the follow-up scenario, keep the previous turn.
3. Copy the chatbot's **actual** response into the CSV. Record retrieved source titles and execution time if shown.
4. Inspect linked sources manually for each substantive assertion before scoring.
5. Calculate descriptive statistics for ratings, pass rate and refusal/uncertainty responses.
6. From actual failures, identify at least three recurring issues, if observed. If fewer than three occur, report that limitation and run additional targeted edge-case tests rather than inventing failures.
7. Propose a change to the prompt, retrieval strategy, source collection or runtime for every observed issue.
8. Report results faithfully in the final paper; do not present unrun tests as completed.

**Potential failure modes to investigate, not assumed findings:** irrelevant retrieval, unsupported extrapolations, weak cross-language retrieval, poor multi-turn co-reference, latency and inconsistent citations.
