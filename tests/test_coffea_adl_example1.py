# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
# ---

# Run this cell if you do not have coffea installed
# (e.g. on SWAN with LCG 96Python3 stack)
# (for .py version: next line should be commented
# since they are converted to ipybn via jupytext)
# !pip install --user --upgrade coffea awkward numba
# Preparation for testing
# !pip install --user --upgrade ipytest pytest-benchmark pytest-csv
# !

# For single-machine scheduler:
# https://docs.dask.org/en/latest/setup.html
# https://docs.dask.org/en/latest/setup/single-machine.html
# ! pip install --user dask distributed dask-jobqueue blosc --upgrades
# ! pip install --force-reinstall --ignore-installed git+git://github.com/oshadura/distributed.git@coffea-casa-facility#egg=distributed

# Uncomment this if you want to test Dask on UNL HTCondor:  %env DASK_COFFEABENCH_SETUP="unl-htcondor"
# Uncomment this if you want to test Dask on UNL AF coffea-casa: %env DASK_COFFEABENCH_SETUP="coffea-casa"
# Uncomment this if you want to test Dask locally:  %env DASK_COFFEABENCH_SETUP="local"
# %env

# spark.jars.packages doesnt work with Spark 2.4 with kubernetes
# !wget -N https://repo1.maven.org/maven2/edu/vanderbilt/accre/laurelin/1.1.1/laurelin-1.1.1.jar
# !wget -N https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.11.2/log4j-api-2.11.2.jar
# !wget -N https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.11.2/log4j-core-2.11.2.jar
# !wget -N https://repo1.maven.org/maven2/org/lz4/lz4-java/1.5.1/lz4-java-1.5.1.jar
# !wget -N https://repo1.maven.org/maven2/org/tukaani/xz/1.2/xz-1.2.jar

# Uncomment this if you want to test Dask:  %env DASK_COFFEABENCH=1
# Uncomment this if you want to test Spark: %env PYSPARK_COFFEABENCH=1
# Uncomment this if you want to test uproot:  %env UPROOT_COFFEABENCH=1
# %env

import math
import psutil
import pytest
import os

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from uproot_methods import TLorentzVectorArray
import coffea.processor as processor
import numpy as np
import numba as nb
import awkward as ak

if hasattr(__builtins__, "__IPYTHON__"):
    import os
    import ipytest

    ipytest.config(rewrite_asserts=True, magics=True)
    __file__ = "test_coffea_adl_example8.ipynb"
    # Run this cell before establishing spark connection <<<<< IMPORTANT
    os.environ["PYTHONPATH"] = (
        os.environ["PYTHONPATH"] + ":" + "/usr/local/lib/python3.6/site-packages"
    )
    os.environ["PATH"] = os.environ["PATH"] + ":" + "/eos/user/o/oshadura/.local/bin"


if "PYSPARK_COFFEABENCH" in os.environ and os.environ["PYSPARK_COFFEABENCH"] == "1":
    import pyspark.sql
    from pyarrow.compat import guid
    from coffea.processor.spark.detail import _spark_initialize, _spark_stop
    from coffea.processor.spark.spark_executor import spark_executor

available_laurelin_version = [("edu.vanderbilt.accre:laurelin:1.1.1")]

if "DASK_COFFEABENCH" in os.environ and os.environ["DASK_COFFEABENCH"] == "1":
    from dask.distributed import Client, LocalCluster
    from dask_jobqueue import HTCondorCluster
    from coffea_casa import CoffeaCasaCluster

fileset = {
    "SingleMu": [
        "root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root"
    ]
}

# The processor class bundles our data analysis together while giving us some
# helpful tools.  It also leaves looping and chunks to the framework instead of us.
class Processor(processor.ProcessorABC):
    def __init__(self):
        # Bins and categories for the histogram are defined here. For format,
        # see https://coffeateam.github.io/coffea/stubs/coffea.hist.hist_tools.Hist.html && https://coffeateam.github.io/coffea/stubs/coffea.hist.hist_tools.Bin.html
        dataset_axis = hist.Cat("dataset", "")
        MET_axis = hist.Bin("MET", "MET [GeV]", 50, 0, 100)

        # The accumulator keeps our data chunks together for histogramming.
        # It also gives us cutflow, which can be used to keep track of data.
        self._accumulator = processor.dict_accumulator(
            {
                "MET": hist.Hist("Counts", dataset_axis, MET_axis),
                "cutflow": processor.defaultdict_accumulator(int),
            }
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()

        # This is where we do our actual analysis. The dataset has columns
        # similar to the TTree's; events.columns can tell you them, or events.
        # [object].columns for deeper depth.
        dataset = events.metadata["dataset"]
        MET = events.MET.pt

        # We can define a new key for cutflow (in this case 'all events').
        # Then we can put values into it. We need += because it's per-chunk (demonstrated below)
        output["cutflow"]["all events"] += MET.size
        output["cutflow"]["number of chunks"] += 1

        # This fills our histogram once our data is collected.
        # The hist key ('MET=') will be defined in the bin in __init__.
        output["MET"].fill(dataset=dataset, MET=MET.flatten())
        return output

    def postprocess(self, accumulator):
        return accumulator


if "DASK_COFFEABENCH" in os.environ and os.environ["DASK_COFFEABENCH"] == "1":

    def test_dask_adlexample1(benchmark):
        @benchmark
        def dask_adlexample1(n_cores=2):
            # Dask settings (three different cases)
            if os.environ["DASK_COFFEABENCH_SETUP"] == "unl-tier3":
                client = Client("t3.unl.edu:8786")
            if os.environ["DASK_COFFEABENCH_SETUP"] == "coffea-casa":
                cluster = CoffeaCasaCluster()
                cluster.scale(jobs=5)
                client = Client(cluster)
            if os.environ["DASK_COFFEABENCH_SETUP"] == "local":
                client = Client()
            exe_args = {"client": client, "nano": True}
            output = processor.run_uproot_job(
                fileset,
                treename="Events",
                processor_instance=Processor(),
                executor=processor.dask_executor,
                executor_args=exe_args,
            )
            hist.plot1d(
                output["MET"],
                overlay="dataset",
                fill_opts={"edgecolor": (0, 0, 0, 0.3), "alpha": 0.8},
            )
            return output


def coffea_laurelin_adlexample1(laurelin_version, n_workers, partition_size):
    spark_config = (
        pyspark.sql.SparkSession.builder.appName("spark-executor-test-%s" % guid())
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "6g")
        .config("spark.sql.execution.arrow.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", partition_size)
        .config("spark.kubernetes.container.image.pullPolicy", "true")
        .config(
            "spark.kubernetes.container.image",
            "gitlab-registry.cern.ch/db/spark-service/docker-registry/swan:laurelin",
        )
        .config(
            "spark.driver.extraClassPath",
            "./laurelin-1.0.0.jar:./lz4-java-1.5.1.jar:./log4j-core-2.11.2.jar:./log4j-api-2.11.2.jar:./xz-1.2.jar",
        )
        .config("spark.kubernetes.memoryOverheadFactor", "0.1")
    )
    spark = _spark_initialize(
        config=spark_config,
        log_level="WARN",
        spark_progress=False,
        laurelin_version="1.1.1-SNAPSHOT",
    )
    output = processor.run_spark_job(
        fileset,
        Processor(),
        spark_executor,
        spark=spark,
        partitionsize=partition_size,
        thread_workers=n_workers,
        executor_args={
            "file_type": "edu.vanderbilt.accre.laurelin.Root",
            "cache": False,
        },
    )
    hist.plot1d(
        output["MET"],
        overlay="dataset",
        fill_opts={"edgecolor": (0, 0, 0, 0.3), "alpha": 0.8},
    )
    return output


if "PYSPARK_COFFEABENCH" in os.environ and os.environ["PYSPARK_COFFEABENCH"] == "1":

    @pytest.mark.benchmark(group="coffea-laurelin-adl-example1")
    @pytest.mark.parametrize("laurelin_version", available_laurelin_version)
    @pytest.mark.parametrize("n_workers", range(1, psutil.cpu_count(logical=False)))
    @pytest.mark.parametrize("partition_size", range(200000, 500000, 200000))
    def test_laurelin_adlexample1(
        benchmark, laurelin_version, n_workers, partition_size
    ):
        benchmark(
            coffea_laurelin_adlexample1,
            available_laurelin_version,
            n_workers,
            partition_size,
        )


def coffea_uproot_adlexample1(n_workers, chunk_size):
    output = processor.run_uproot_job(
        fileset,
        treename="Events",
        processor_instance=Processor(),
        executor=processor.futures_executor,
        chunksize=chunk_size,
        executor_args={"workers": n_workers, "nano": True},
    )
    hist.plot1d(
        output["MET"],
        overlay="dataset",
        fill_opts={"edgecolor": (0, 0, 0, 0.3), "alpha": 0.8},
    )
    return output


if "UPROOT_COFFEABENCH" in os.environ and os.environ["UPROOT_COFFEABENCH"] == "1":

    @pytest.mark.parametrize("n_workers", range(1, psutil.cpu_count(logical=False)))
    @pytest.mark.parametrize("chunk_size", range(200000, 500000, 200000))
    def test_coffea_uproot_adl_example8(benchmark, n_workers, chunk_size):
        benchmark(coffea_uproot_adlexample1, n_workers, chunk_size)


if hasattr(__builtins__, "__IPYTHON__"):
    ipytest.run("-qq")
