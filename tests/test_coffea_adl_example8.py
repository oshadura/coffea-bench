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
# This is a helper function which adds muon (0)
# and electron (1) identifiers to a stacked lepton JaggedArray.
def make_labeled_p4(x, indices, itype):
    p4 = TLorentzVectorArray.from_ptetaphim(x.pt, x.eta, x.phi, x.mass)
    return ak.JaggedArray.zip(
        p4=p4,
        ptype=itype * x.pt.ones_like().astype(np.int),
        flavor=indices,
        charge=x.charge,
    )


def stack_leptons(muons, electrons):
    """
    # This generates a stacked lepton JaggedArray,
    allowing combination of both muons and electrons for computations across flavor.
    """
    # Construct new lepton indices within every event array.
    muons_indices = (
        ak.JaggedArray.fromoffsets(
            muons.pt.offsets, np.arange(0, muons.pt.content.size)
        )
        - muons.pt.offsets[:-1]
    )
    electrons_indices = (
        ak.JaggedArray.fromoffsets(
            electrons.pt.offsets, np.arange(0, electrons.pt.content.size)
        )
        - electrons.pt.offsets[:-1]
    )
    # Assign 0/1 value depending on whether lepton is muon/electron.
    muons_p4 = make_labeled_p4(muons, muons_indices, 0)
    electrons_p4 = make_labeled_p4(electrons, electrons_indices, 1)
    # Concatenate leptons.
    stacked_p4 = ak.concatenate((muons_p4, electrons_p4), axis=1)

    return stacked_p4


# This program plots the transverse mass of MET and a third lepton,
# where the third lepton is associated with a lepton pair
# that has the same flavor, opposite charge, and closest mass to 91.2.
class Processor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat("dataset", "MET and Third Lepton")
        muon_axis = hist.Bin("massT", "Transverse Mass", 50, 15, 250)

        self._accumulator = processor.dict_accumulator(
            {
                "massT": hist.Hist("Counts", dataset_axis, muon_axis),
                "cutflow": processor.defaultdict_accumulator(int),
            }
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        dataset = events.metadata["dataset"]
        muons = events.Muon
        electrons = events.Electron
        MET = events.MET
        # A few reasonable muon and electron selection cuts
        muons = muons[(muons.pt > 10) & (np.abs(muons.eta) < 2.4)]
        electrons = electrons[(electrons.pt > 10) & (np.abs(electrons.eta) < 2.5)]
        leptons = stack_leptons(muons, electrons)
        # Filter out events with less than 3 leptons.
        MET = MET[leptons.counts >= 3]
        trileptons = leptons[leptons.counts >= 3]
        # Generate the indices of every pair;
        # indices because we'll be removing these elements later.
        lepton_pairs = trileptons.argchoose(2)
        # Select pairs that are SFOS.
        SFOS_pairs = lepton_pairs[
            (trileptons[lepton_pairs.i0].flavor == trileptons[lepton_pairs.i1].flavor)
            & (trileptons[lepton_pairs.i0].charge != trileptons[lepton_pairs.i1].charge)
        ]
        # Find the pair with mass closest to Z.
        closest_pairs = SFOS_pairs[
            np.abs(
                (trileptons[SFOS_pairs.i0].p4 + trileptons[SFOS_pairs.i1].p4).mass
                - 91.2
            ).argmin()
        ]
        # Remove elements of these pairs from leptons by negating the indices.
        is_in_pair_mask = trileptons[~closest_pairs.i0 | ~closest_pairs.i1]
        # Find the highest-pt lepton out of the ones that remain.
        leading_lepton = trileptons[trileptons.p4.pt.argmax()]
        # Can't cross MET with leading_lepton, but we need both phi and pt. So we build a crossable table.
        MET_tab = ak.JaggedArray.fromcounts(
            np.ones_like(MET.pt, dtype=np.int), ak.Table({"phi": MET.phi, "pt": MET.pt})
        )
        met_plus_lep = MET_tab.cross(leading_lepton)
        # Do some math to get what we want.
        dphi_met_lep = (met_plus_lep.i0.phi - met_plus_lep.i1.p4.phi + math.pi) % (
            2 * math.pi
        ) - math.pi
        mt_lep = np.sqrt(
            2.0
            * met_plus_lep.i0.pt
            * met_plus_lep.i1.p4.pt
            * (1.0 - np.cos(dphi_met_lep))
        )
        output["massT"].fill(dataset=dataset, massT=mt_lep.flatten())

        return output

    def postprocess(self, accumulator):
        return accumulator


if "DASK_COFFEABENCH" in os.environ and os.environ["DASK_COFFEABENCH"] == "1":

    def test_dask_adlexample8(benchmark):
        @benchmark
        def dask_adlexample8(n_cores=2):
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


def coffea_laurelin_adl_example8(laurelin_version, n_workers, partition_size):
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
        output["massT"],
        overlay="dataset",
        fill_opts={"edgecolor": (0, 0, 0, 0.3), "alpha": 0.8},
    )
    return output


if "PYSPARK_COFFEABENCH" in os.environ and os.environ["PYSPARK_COFFEABENCH"] == "1":

    @pytest.mark.benchmark(group="coffea-laurelin-adl-example8")
    @pytest.mark.parametrize("laurelin_version", available_laurelin_version)
    @pytest.mark.parametrize("n_workers", range(1, psutil.cpu_count(logical=False)))
    @pytest.mark.parametrize("partition_size", range(200000, 500000, 200000))
    def test_coffea_laurelin_adl_example8(
        benchmark, laurelin_version, n_workers, partition_size
    ):
        benchmark(
            coffea_laurelin_adl_example8,
            available_laurelin_version,
            n_workers,
            partition_size,
        )


def coffea_uproot_adl_example8(n_workers, chunk_size):
    output = processor.run_uproot_job(
        fileset,
        treename="Events",
        processor_instance=Processor(),
        executor=processor.futures_executor,
        chunksize=chunk_size,
        executor_args={"workers": n_workers, "nano": True},
    )


if "UPROOT_COFFEABENCH" in os.environ and os.environ["UPROOT_COFFEABENCH"] == "1":

    @pytest.mark.benchmark(group="coffea-uproot-adl-example8")
    @pytest.mark.parametrize("n_workers", range(1, psutil.cpu_count(logical=False)))
    @pytest.mark.parametrize("chunk_size", range(200000, 500000, 200000))
    def test_coffea_uproot_adl_example8(benchmark, n_workers, chunk_size):
        benchmark(coffea_uproot_adl_example8, n_workers, chunk_size)


if hasattr(__builtins__, "__IPYTHON__"):
    ipytest.run("-qq")
