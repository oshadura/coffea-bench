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

if hasattr(__builtins__, "__IPYTHON__"):
    import os
    import ipytest

    ipytest.config(rewrite_asserts=True, magics=True)
    __file__ = "test_coffea_dimuon_analysis.ipynb"
    # Run this cell before establishing spark connection <<<<< IMPORTANT
    os.environ["PYTHONPATH"] = (
        os.environ["PYTHONPATH"] + ":" + "/usr/local/lib/python3.6/site-packages"
    )
    os.environ["PATH"] = os.environ["PATH"] + ":" + "/eos/user/o/oshadura/.local/bin"


import psutil
import pytest
import os

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor

if "PYSPARK_COFFEABENCH" in os.environ and os.environ["PYSPARK_COFFEABENCH"] == "1":
    import pyspark.sql
    from pyarrow.compat import guid
    from coffea.processor.spark.detail import _spark_initialize, _spark_stop
    from coffea.processor.spark.spark_executor import spark_executor

available_laurelin_version = [("edu.vanderbilt.accre:laurelin:1.0.0")]

if "DASK_COFFEABENCH" in os.environ and os.environ["DASK_COFFEABENCH"] == "1":
    from dask.distributed import Client, LocalCluster
    from dask_jobqueue import HTCondorCluster
    from coffea_casa import CoffeaCasaCluster

fileset = {
    "DoubleMuon": {
        "files": [
            "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
            "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012C_DoubleMuParked.root",
        ],
        "treename": "Events",
    }
}


class DimuonProcessor(processor.ProcessorABC):
    def __init__(self):
        self._columns = [
            "nMuon",
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            "Muon_mass",
            "Muon_charge",
        ]
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        mass_axis = hist.Bin("mass", r"$m_{\mu\mu}$ [GeV]", 30000, 0.25, 300)
        self._accumulator = processor.dict_accumulator(
            {
                "mass": hist.Hist("Counts", dataset_axis, mass_axis),
                "cutflow": processor.defaultdict_accumulator(int),
            }
        )

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    def process(self, df):
        output = self.accumulator.identity()
        dataset = df["dataset"]
        muons = JaggedCandidateArray.candidatesfromcounts(
            df["nMuon"],
            pt=df["Muon_pt"].content,
            eta=df["Muon_eta"].content,
            phi=df["Muon_phi"].content,
            mass=df["Muon_mass"].content,
            charge=df["Muon_charge"].content,
        )
        output["cutflow"]["all events"] += muons.size
        twomuons = muons.counts == 2
        output["cutflow"]["two muons"] += twomuons.sum()
        opposite_charge = twomuons & (muons["charge"].prod() == -1)
        output["cutflow"]["opposite charge"] += opposite_charge.sum()
        dimuons = muons[opposite_charge].distincts()
        output["mass"].fill(dataset=dataset, mass=dimuons.mass.flatten())
        return output

    def postprocess(self, accumulator):
        return accumulator


if "DASK_COFFEABENCH" in os.environ and os.environ["DASK_COFFEABENCH"] == "1":

    def test_dask_dimuon_analysis(benchmark):
        @benchmark
        def dask_dimuon_analysis(n_cores=2):
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
                processor_instance=DimuonProcessor(),
                executor=processor.dask_executor,
                executor_args=exe_args,
            )
            hist.plot1d(
                output["MET"],
                overlay="dataset",
                fill_opts={"edgecolor": (0, 0, 0, 0.3), "alpha": 0.8},
            )
            return output


def coffea_laurelin_dimuon_analysis(laurelin_version, n_workers, partition_size):
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
        DimuonProcessor(),
        spark_executor,
        spark=spark,
        partitionsize=partition_size,
        thread_workers=n_workers,
        executor_args={
            "file_type": "edu.vanderbilt.accre.laurelin.Root",
            "cache": False,
        },
    )


if "PYSPARK_COFFEABENCH" in os.environ and os.environ["PYSPARK_COFFEABENCH"] == "1":

    @pytest.mark.benchmark(group="coffea-laurelin-dimuon-analysis")
    @pytest.mark.parametrize("n_workers", range(1, psutil.cpu_count(logical=False)))
    @pytest.mark.parametrize("chunk_size", range(200000, 500000, 200000))
    @pytest.mark.parametrize("maxchunk_size", range(300000, 600000, 200000))
    def test_coffea_laurelin_dimuon_analysis(
        benchmark, laurelin_version, n_workers, partition_size
    ):
        benchmark(
            coffea_laurelin_dimuon_analysis,
            available_laurelin_version,
            n_workers,
            partition_size,
        )


def coffea_uproot_dimuon_analysis(n_workers, chunk_size):
    output = processor.run_uproot_job(
        fileset,
        treename="Events",
        processor_instance=DimuonProcessor(),
        executor=processor.futures_executor,
        chunksize=chunk_size,
        executor_args={"workers": n_workers, "nano": True},
    )


if "UPROOT_COFFEABENCH" in os.environ and os.environ["UPROOT_COFFEABENCH"] == "1":

    @pytest.mark.benchmark(group="coffea-uproot-dimuon-analysis")
    @pytest.mark.parametrize("n_workers", range(1, psutil.cpu_count(logical=False)))
    @pytest.mark.parametrize("chunk_size", range(200000, 500000, 200000))
    def test_coffea_uproot_dimuon_analysis(
        benchmark, n_workers, chunk_size, maxchunk_size
    ):
        benchmark(coffea_uproot_dimuon_analysis, n_workers, chunk_size)


if hasattr(__builtins__, "__IPYTHON__"):
    ipytest.run("-qq")
