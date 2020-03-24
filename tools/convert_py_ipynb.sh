#/usr/bin/bash

jupytext --to notebook benchmarks/*/*.py
jupytext --set-formats ipynb,py benchmarks/*/*.py
