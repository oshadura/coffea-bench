# coffea-bench

[![open-swan](https://swan.web.cern.ch/sites/swan.web.cern.ch/files/pictures/open_in_swan.svg)](https://cern.ch/swanserver/cgi-bin/go?projurl=https://github.com/oshadura/coffea-bench.git)

A set of benchmarks to introduce the continious testing for Coffea and Laurelin ecosystem.


# UNL Tier-3 Dask cluster

conda create --name coffea-bench python=3.7 distributed
conda activate coffea-benc
conda env update --name coffea-bench --file unl_coffea_bench.yml
python -m pip install -U -r dev-requirements.txt


# Generic requirements
python -m pip install -e .


# Running tests for Dask setup

pytest -k benchmarks/dask/*

# for Pyspark 

pytest -k benchmarks/dask/*

# For uproot (futures.concurrent)

pytest -k benchmarks/uproot/*
