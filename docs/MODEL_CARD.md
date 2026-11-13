# Model and simulator card — Research Thunderdome

## Intended use

A bounded experiment runner linking selected model configurations and benchmark claims to real training records.

## Implemented methods

- Installed sklearn breast-cancer and wine benchmarks
- Fixed stratified train/validation/test partitions
- Allowlisted logistic, random-forest and histogram-boosting models
- Budgeted random search and saved winning pipeline
- Audit fixtures for leakage, test tuning and arbitrary code requests

Some projects use deterministic simulators and rules rather than trained neural models. Those distinctions are retained in this card and the source.

## Evaluation

See `docs/EXPERIMENTS.md` and the machine-readable reports. No pretrained foundation-model weights or real-world efficacy certification are included. Parameters, splits and synthetic generators are visible in source.

## Limitations

Allowlisted sklearn configurations with budgeted random search. No arbitrary code execution or self-certified scientific discovery.

## Extensions not implemented in this release

- Nested or cross-dataset model-selection evaluation
- Bayesian experiment planning and isolated external worker processes
