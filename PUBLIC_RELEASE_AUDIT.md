# Public release audit

Audit date: 2026-09-04 UTC

## Verdict

`PUBLIC_RELEASE_READY`

| Release gate | Status | Evidence |
|---|---|---|
| Scientific claims | PASS | Architecture, opponent-shift, collapse, context-format, history, and OpenSpiel statements were checked against authoritative valid artifacts; invalid runs appear only as examples of protocol rejection. |
| Code provenance | PASS | Every substantive module is classified in `docs/licensing_and_provenance.md`; no copied or derived third-party implementation was identified. |
| Licensing | PASS | The source project has no top-level grant, so the repository uses an all-rights-reserved notice and makes no unsupported open-source claim. Public viewing does not imply reuse permission. |
| Citation metadata | PASS | `CITATION.cff` parses as YAML, names Henry He, points to the intended repository, and contains no private contact information or unsupported license field. |
| Privacy and secrets | PASS | Current tracked/untracked files and reachable Git history were scanned for credential patterns, private keys, tokens, private network addresses, private host details, absolute source paths, and internal communications; none were found. |
| Dataset redistribution | PASS | No raw CraigslistBargain or Deal-or-No-Deal records are included. Acquisition and upstream MIT / CC BY-NC 4.0 terms are documented. |
| Large files | PASS | No tracked file exceeds 5 MB; no checkpoint, adapter, model snapshot, tokenizer bundle, raw generation log, or dataset is tracked. |
| Installability | PASS | A non-editable package install into an isolated target directory succeeded without access to the source research repository. |
| CPU demos | PASS | Pricing, replay versus closed loop, representation equivalence, and representation reward parity all pass. |
| Optional OpenSpiel demo | PASS | Kuhn CFR+ completed 200 iterations, found 12 decision nodes / six information states, handled all actions legally, and reported NashConv `0.0005899209670298666`. |
| Tests | PASS | 20 tests pass from both the working checkout and isolated installed package. |
| README links | PASS | All local Markdown targets exist; external dataset links use their canonical upstream repositories. |
| Figures | PASS | Four valid SVGs use validated numbers, explicit axes or non-axis cards, readable labels, and scope captions. |
| Placeholders | PASS | No release placeholder remains. |
| GitHub readiness | PASS | GitHub CLI is authenticated as `henry550xz`; the intended repository name was unoccupied at audit time. |

## Release scope

CPU commands verify executable environments, evaluators, representation checks, and logged-support logic. They do not reproduce trained-model experiments. GPU/API artifacts, credentials, raw datasets, private checkpoints, and internal experiment trees are intentionally absent.
