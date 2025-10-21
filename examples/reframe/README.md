# ReFrame Tests

## Organization

The ReFrame configuration is in `sites.py`. Currently, the `sites.py` only contains
configuration for ALCF Aurora. We define two partitions for Aurora: `aurora:compute`
and `aurora:login`. You can pass this configuration file into reframe using the `-C`
flag and use the `--system` flag to select the partition you like.

The `core.py` contains thin wrappers over `reframe.CompileOnlyRegressionTest` and
`reframe.RunOnlyRegressionTest` which are named `CompileOnlyTest` and `RunOnlyTest`
respectively. These classes are used based on the type of test we want to execute.

## Prerequisites

This repository contains the [ReFrame](https://reframe-hpc.readthedocs.io/en/stable/)
implementations of various tests, regression tests, and performance benchmarks.

In order to run tests, you first need to install [uv](https://docs.astral.sh/uv/getting-started/installation/).
Installation instructions can be found [here](https://docs.astral.sh/uv/getting-started/installation/).
