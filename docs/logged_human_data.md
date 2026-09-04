# Logged human interaction data

No raw human dialogue is included in this repository.

## CraigslistBargain

Obtain CraigslistBargain from the original [Stanford CoCoA repository](https://github.com/stanfordnlp/cocoa/tree/master/craigslistbargain). The upstream CoCoA repository currently carries an MIT license. Users remain responsible for reviewing the upstream terms and dataset documentation before use.

The local adapter extracts conservative dialogue acts and price mentions. It is project code informed by the dataset schema; it does not contain upstream records or copied upstream implementation.

## Deal-or-No-Deal

Obtain the negotiation corpus from the original [End-to-End Negotiator repository](https://github.com/facebookresearch/end-to-end-negotiator/tree/master/data/negotiate). That upstream repository currently identifies its material as CC BY-NC 4.0. The corpus is not vendored because redistribution and downstream use must follow those non-commercial terms.

The local module contains only project-authored allocation enumeration and utility helpers. It does not contain upstream dialogue records or copied upstream implementation.

These adapters support logged next-action and legality analyses. They do not turn either corpus into a counterfactual human simulator; see [`closed_loop_evaluation.md`](closed_loop_evaluation.md).
