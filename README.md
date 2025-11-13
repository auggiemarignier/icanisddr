# Comparing Inner Core Anisotropy Models with the Savage-Dickey Density Ratio

[![tests](https://github.com/auggiemarignier/icanisddr/actions/workflows/tests.yml/badge.svg)](https://github.com/auggiemarignier/icanisddr/actions/workflows/tests.yml) [![codecov](https://codecov.io/github/auggiemarignier/icanisddr/graph/badge.svg?token=WY92GUYFC7)](https://codecov.io/github/auggiemarignier/icanisddr)

## Setup

The code is managed using `uv`.  The core package `sdicani` is used by the various experiments.

## Experiments

Experiments can be run simply by

```bash
uv run <experiment_name>
```

This will install all the dependencies for the `sdicani` core package and any additional dependencies for the experiment.

### Available experiments

- `bulkic` - Synthetic case with good coverage where the IC is represented by a single anisotropic crystal.
