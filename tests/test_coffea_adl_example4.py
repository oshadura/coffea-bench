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

# Run this cell if you do not have coffea installed (e.g. on SWAN with LCG 96Python3 stack)
# (for .py version -> next line should be commented since they are converted to ipybn via jupytext)
# !pip install --user --upgrade coffea
# Preparation for testing
# !pip install --user --upgrade ipytest
# !pip install --user --upgrade pytest-benchmark
# For single-machine scheduler:
# https://docs.dask.org/en/latest/setup.html
# https://docs.dask.org/en/latest/setup/single-machine.html
# ! pip install --user dask distributed --upgrade

# Uncomment this if you want to test Dask on UNL HTCCondor:  %env DASK_COFFEABENCH_SETUP="unl-htccondor"
# Uncomment this if you want to test Dask on UNL Tier3: %env DASK_COFFEABENCH_SETUP="unl-tier3"
# Uncomment this if you want to test Dask locally:  %env DASK_COFFEABENCH_SETUP="local"
# %env

# spark.jars.packages doesnt work with Spark 2.4 with kubernetes
# !wget -N https://repo1.maven.org/maven2/edu/vanderbilt/accre/laurelin/1.0.0/laurelin-1.0.0.jar
# !wget -N https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.11.2/log4j-api-2.11.2.jar
# !wget -N https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.11.2/log4j-core-2.11.2.jar
# !wget -N https://repo1.maven.org/maven2/org/lz4/lz4-java/1.5.1/lz4-java-1.5.1.jar
# !wget -N https://repo1.maven.org/maven2/org/tukaani/xz/1.2/xz-1.2.jar

# Uncomment this if you want to test Dask:  %env DASK_COFFEABENCH=1
# Uncomment this if you want to test Spark: %env PYSPARK_COFFEABENCH=1
# Uncomment this if you want to test uproot:  %env UPROOT_COFFEABENCH=1
# %env

if hasattr(__builtins__,'__IPYTHON__'):
    import os
    import ipytest
    ipytest.config(rewrite_asserts=True, magics=True)
    __file__ = 'test_coffea_adl_example4.ipynb'
    # Run this cell before establishing spark connection <<<<< IMPORTANT
    os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':' + '/usr/local/lib/python3.6/site-packages'
    os.environ['PATH'] = os.environ['PATH'] + ':' + '/eos/user/o/oshadura/.local/bin'


import psutil
import pytest
import os

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor

if os.environ["PYSPARK_COFFEABENCH"] == '1':
    import pyspark.sql
    from pyarrow.compat import guid
    from coffea.processor.spark.detail import _spark_initialize, _spark_stop
    from coffea.processor.spark.spark_executor import spark_executor

available_laurelin_version = [("edu.vanderbilt.accre:laurelin:1.0.0")]

if os.environ["DASK_COFFEABENCH"] == '1':
    from dask.distributed import Client, LocalCluster
    from dask_jobqueue import HTCondorCluster

fileset = {
    'MET Masked by Jet': { 'files': ['root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root'],
             'treename': 'Events'
            }
}

class JetMETProcessor(processor.ProcessorABC):
    def __init__(self):
        self._columns = ['MET_pt', 'nJet', 'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass']
        dataset_axis = hist.Cat("dataset", "")
        MET_axis = hist.Bin("MET_pt", "MET [GeV]", 50, 0, 125)
        self._accumulator = processor.dict_accumulator({
            'MET_pt': hist.Hist("Counts", dataset_axis, MET_axis),
            'cutflow': processor.defaultdict_accumulator(int)
        })
        
    @property
    def columns(self):
        return self._columns
    
    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, df):
        output = self.accumulator.identity()   
        dataset = df["dataset"]
        jets = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt=df['Jet_pt'].content,
            eta=df['Jet_eta'].content,
            phi=df['Jet_phi'].content,
            mass=df['Jet_mass'].content,
            )
        # We can access keys without appealing to a JCA, as well.
        MET = df['MET_pt']
        output['cutflow']['all events'] += jets.size 
        # We want jets with a pt of at least 40.
        pt_min = (jets['p4'].pt > 40)
        # We want MET where the above condition is met for at least two jets. The above is a list of Boolean sublists generated from the jet sublists (True if condition met, False if not). If we sum each sublist, we get the amount of jets matching the condition (since True = 1).
        good_MET = MET[(pt_min.sum() >= 2)]
        output['cutflow']['final events'] += good_MET.size
        output['MET_pt'].fill(dataset=dataset, MET_pt=good_MET.flatten())
        return output

    def postprocess(self, accumulator):
        return accumulator

if 'DASK_COFFEABENCH' in os.environ and os.environ["DASK_COFFEABENCH"] == '1':
    def test_dask_adlexample4(benchmark):
        @benchmark
        def dask_adlexample4(n_cores=2):
        # Dask settings (three different cases)
            if os.environ["DASK_COFFEABENCH_SETUP"] == 'unl-tier3':
                client = Client("t3.unl.edu:8786")
            if os.environ["DASK_COFFEABENCH_SETUP"] == 'unl-htccondor':
                cluster = HTCondorCluster(cores=n_cores, memory="2GB",disk="1GB",dashboard_address=9998)
                cluster.scale(jobs=5)
                client = Client(cluster)
            if os.environ["DASK_COFFEABENCH_SETUP"] == 'local':
                client = Client()
            cachestrategy = 'dask-worker'
            exe_args = {
                'client': client,
                'nano': True,
                'cachestrategy': cachestrategy,
                'savemetrics': True,
                'worker_affinity': True if cachestrategy is not None else False,
            }
            output = processor.run_uproot_job(fileset,
                                      treename = 'Events',
                                      processor_instance = JetMETProcessor(),
                                      executor = processor.dask_executor,
                                      executor_args = exe_args
                                      
            )
            return output 

def coffea_laurelin_adl_example4(laurelin_version, n_workers, partition_size):
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test-%s' % guid()) \
        .master('local[*]') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '6g') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', partition_size)\
        .config('spark.kubernetes.container.image.pullPolicy', 'true')\
        .config('spark.kubernetes.container.image', 'gitlab-registry.cern.ch/db/spark-service/docker-registry/swan:laurelin')\
        .config('spark.kubernetes.memoryOverheadFactor', '0.1')

    spark = _spark_initialize(config=spark_config, log_level='DEBUG', 
                          spark_progress=False, laurelin_version='1.0.1-SNAPSHOT')
    
    output = processor.run_spark_job(fileset,
                                     JetMETProcessor(),
                                     spark_executor, 
                                     spark=spark,
                                     partitionsize=partition_size,
                                     thread_workers=n_workers,
                                     executor_args={'file_type': 'edu.vanderbilt.accre.laurelin.Root', 'cache': False})

if 'PYSPARK_COFFEABENCH' in os.environ and os.environ["PYSPARK_COFFEABENCH"] == '1':
    @pytest.mark.benchmark(group="coffea-laurelin-adl-example4")
    @pytest.mark.parametrize("laurelin_version", available_laurelin_version)
    @pytest.mark.parametrize("n_workers", range(1,psutil.cpu_count(logical=False)))
    @pytest.mark.parametrize("partition_size", range(200000,500000,200000))
    def test_coffea_laurelin_adl_example4(benchmark, laurelin_version, n_workers, partition_size):
        benchmark(coffea_laurelin_adl_example4, available_laurelin_version, n_workers, partition_size)

def coffea_uproot_adl_example4(n_workers, chunk_size, maxchunk_size):
    output = processor.run_uproot_job(fileset,
                                      treename = 'Events',
                                      processor_instance = JetMETProcessor(),
                                      executor = processor.futures_executor,
                                      chunksize = chunk_size,
                                      maxchunks = maxchunk_size,
                                      executor_args = {'workers': n_workers}
                                      
    ) 

if 'UPROOT_COFFEABENCH' in os.environ and os.environ["UPROOT_COFFEABENCH"] == '1':
    @pytest.mark.benchmark(group="coffea-uproot-adl-example4")
    @pytest.mark.parametrize("n_workers", range(1,psutil.cpu_count(logical=False)))
    @pytest.mark.parametrize("chunk_size", range(200000,500000,200000))
    @pytest.mark.parametrize("maxchunk_size", range(300000,600000,200000))
    def test_coffea_uproot_adl_example4(benchmark, n_workers, chunk_size, maxchunk_size):
        benchmark(coffea_uproot_adl_example4, n_workers, chunk_size, maxchunk_size)

if hasattr(__builtins__,'__IPYTHON__'):
    ipytest.run('-qq')
