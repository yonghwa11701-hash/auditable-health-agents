# Build patterns for evaluated multi-agent health apps

These are lessons from two hackathon builds: a clinical-conversation assistant (Accel AI Innovate London, Oct 2026) and LungPair (AgentVerse 2026).

## 1. Write the contract before the code
Start with one `CLAUDE.md` / `AGENTS.md` file and treat it as the single source of truth. It holds the one-liner, the safety rules (numbered and non-negotiable), an MVP table (P0 / P1 / Out), the user journey, an architecture diagram and folder owners. The most serious failure we saw was contract drift: the contract, schema, prompts and UI disagreeing with each other.

## 2. Pipeline shape
- Code owns state, stages and resume. Agents only make judgements, and rules live in code rather than in prompts.
- Use producer → evaluator → retry with feedback (max 2 retries). Save every attempt and verdict. After the last retry, keep the best attempt and mark it **flagged**.
- The evaluator decides pass or fail and explains what to fix. It never rewrites the output itself.
- Use a cheap model for producers and a stronger one for review. Make each model an environment variable.
- Make one short call per unit of work, not one long-running request. Use a lease so only one worker handles a session.
- Put deterministic guards after the model: caps, dedup, and dropping anything that has no source ID.
- Parse JSON robustly (strip code fences, fail loudly, retry).
- Every claim cites its source. Build the citation check on day one.

## 3. Proof the agents ran
- Have every agent step write an audit event (agent, attempt, pass, score, payload).
- Show a live agent timeline, and include at least one rejected-then-regenerated event in the demo.
- Use a planted-error grader and a kill-and-resume test.
- Keep a per-run token and cost counter.

## 4. Tests that catch real bugs
- Build a fixture of synthetic or public inputs with planted errors and expected verdicts.
- Add a regression test for every prompt bug found.
- Read the actual output text of the demo run before showing it.

## 5. Security
- Clients get only publishable keys. Model and service keys stay server-side or in the OS keychain, never in a committed `.env`.
- Third-party APIs get short-lived, single-use tokens.
- Use synthetic or public data only.

## 6. Team workflow (humans plus several AI coding agents)
- Use branches and PRs, small commits, and named folder owners.
- Give each AI terminal its own git worktree and a claimed task on a shared board, with locks for shared files.
- Use three status words: *present in source* / *verified run* (a dated run with retained output) / *proposed*.
- Make every agent write its output to a file.
