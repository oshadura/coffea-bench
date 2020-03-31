#!/usr/bin/bash

# To be used from coffea-bench/:
# as a requirement install jupytext via pip.
jupytext --to notebook tests/*.py
jupytext --set-formats ipynb,py tests/*.py
