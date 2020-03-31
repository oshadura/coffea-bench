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


# spark.jars.packages doesnt work with Spark 2.4 with kubernetes
# !wget -N https://repo1.maven.org/maven2/edu/vanderbilt/accre/laurelin/1.0.0/laurelin-1.0.0.jar
# !wget -N https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.11.2/log4j-api-2.11.2.jar
# !wget -N https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.11.2/log4j-core-2.11.2.jar
# !wget -N https://repo1.maven.org/maven2/org/lz4/lz4-java/1.5.1/lz4-java-1.5.1.jar
# !wget -N https://repo1.maven.org/maven2/org/tukaani/xz/1.2/xz-1.2.jar

# Uncomment this if you want to test Dask:
# # %env DASK_COFFEABENCH=1

# Uncomment this if you want to test Spark:
# # %env PYSPARK_COFFEABENCH=1

# Uncomment this if you want to test uproot:
# # %env UPROOT_COFFEABENCH=1

if hasattr(__builtins__,'__IPYTHON__'):
    import os
    import ipytest
    ipytest.config(rewrite_asserts=True, magics=True)
    __file__ = 'test_coffea_adl_example6.ipynb'
    # Run this cell before establishing spark connection <<<<< IMPORTANT
    os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':' + '/usr/local/lib/python3.6/site-packages'
    os.environ['PATH'] = os.environ['PATH'] + ':' + '/eos/user/o/oshadura/.local/bin'


import psutil
import pytest
import os

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor

if 'PYSPARK_COFFEABENCH' in os.environ:
    import pyspark.sql
    from pyarrow.compat import guid
    from coffea.processor.spark.detail import _spark_initialize, _spark_stop
    from coffea.processor.spark.spark_executor import spark_executor

available_laurelin_version = [("edu.vanderbilt.accre:laurelin:1.0.0")]

if 'DASK_COFFEABENCH' in os.environ:
    from dask.distributed import Client, LocalCluster
    from dask_jobqueue import HTCondorCluster

fileset = {
    'Trijets': { 'files': ['root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root'],
             'treename': 'Events'
            }
}

class TrijetProcessor(processor.ProcessorABC):
    def __init__(self):
        self._columns = ['MET_pt', 'nJet', 'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass', 'Jet_btag']
        dataset_axis = hist.Cat("dataset", "")
        Jet_axis = hist.Bin("Jet_pt", "Jet [GeV]", 50, 15, 200)
        b_tag_axis = hist.Bin("b_tag", "b-tagging discriminant", 50, 0, 1)
        self._accumulator = processor.dict_accumulator({
            'Jet_pt': hist.Hist("Counts", dataset_axis, Jet_axis),
            'b_tag': hist.Hist("Counts", dataset_axis, b_tag_axis),
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
            b_tag=df['Jet_btag'].content
            )
        
        # Closest calculates the distance from 172.5 of a group of masses, finds the minimum distance, then returns a Boolean array of the original input array shape with True where the minimum-distance mass is located.
        def closest(masses):
            delta = abs(172.5 - masses)
            closest_masses = delta.min()
            is_closest = (delta == closest_masses)
            return is_closest
        
        # We're going to be generating combinations of three jets - that's a lot, and cutting pt off at 30 reduces jets by half.
        cut_jets = jets[jets.pt > 30]
        
        # Get all combinations of three jets.
        trijets = cut_jets.choose(3)
        # Get combined masses of those combinations.
        trijet_masses = trijets.mass
        # Get the masses closest to specified value (see function above)
        is_closest = closest(trijet_masses)
        closest_trijets = trijets[is_closest]
        # Get pt of the closest trijets.
        closest_pt = closest_trijets.pt
        # Get btag of the closest trijets. np.maximum(x,y) compares two arrays and gets element-wise maximums. We make two comparisons - once between the first and second jet, then between the first comparison and the third jet.
        closest_btag = np.maximum(np.maximum(closest_trijets.i0['b_tag'], closest_trijets.i1['b_tag']), closest_trijets.i2['b_tag'])
        
        output['Jet_pt'].fill(dataset=dataset, Jet_pt=closest_pt.flatten())
        output['b_tag'].fill(dataset=dataset, b_tag=closest_btag.flatten())
        return output

    def postprocess(self, accumulator):
        return accumulator

if 'DASK_COFFEABENCH' in os.environ:
    def test_dask_adlexample6(benchmark):
        @benchmark
        def dask_adlexample6(n_cores=2):
            # Dask settings (two different cases)
            client = Client("t3.unl.edu:8786")
            #cluster = HTCondorCluster(cores=n_cores, memory="2GB",disk="1GB",dashboard_address=9998)
            #cluster.scale(jobs=5)
            #client = Client(cluster)
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
                                      processor_instance = TrijetProcessor(),
                                      executor = processor.dask_executor,
                                      executor_args = exe_args
                                      
            )
            return output


def coffea_laurelin_adl_example6(laurelin_version, n_workers, partition_size):
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

    spark = _spark_initialize(config=spark_config, log_level='WARN', 
                          spark_progress=False, laurelin_version='1.0.1-SNAPSHOT')
    
    output = processor.run_spark_job(fileset,
                                     TrijetProcessor(),
                                     spark_executor, 
                                     spark=spark,
                                     partitionsize=partition_size,
                                     thread_workers=n_workers,
                                     executor_args={'file_type': 'edu.vanderbilt.accre.laurelin.Root', 'cache': False})

if 'PYSPARK_COFFEABENCH' in os.environ:
    @pytest.mark.benchmark(group="coffea-laurelin-adl-example6")
    @pytest.mark.parametrize("laurelin_version", available_laurelin_version)
    @pytest.mark.parametrize("n_workers", range(1,psutil.cpu_count(logical=False)))
    @pytest.mark.parametrize("partition_size", range(100000,200000,100000))
    def test_coffea_laurelin_adl_example6(benchmark, laurelin_version, n_workers, partition_size):
        benchmark(coffea_laurelin_adl_example6, available_laurelin_version, n_workers, partition_size)


def coffea_uproot_adl_example6(n_workers, chunk_size, maxchunk_size):
    output = processor.run_uproot_job(fileset,
                                      treename = 'Events',
                                      processor_instance = TrijetProcessor(),
                                      executor = processor.futures_executor,
                                      chunksize = chunk_size,
                                      maxchunks = maxchunk_size,
                                      executor_args = {'workers': n_workers}
                                      
    ) 

if 'UPROOT_COFFEABENCH' in os.environ:
    @pytest.mark.benchmark(group="coffea-uproot-adl-example6")
    @pytest.mark.parametrize("n_workers", range(1,psutil.cpu_count(logical=False)))
    @pytest.mark.parametrize("chunk_size", range(200000,400000,200000))
    @pytest.mark.parametrize("maxchunk_size", range(300000,500000,200000))
    def test_coffea_uproot_adl_example6(benchmark, n_workers, chunk_size, maxchunk_size):
        benchmark(coffea_uproot_adl_example6, n_workers, chunk_size, maxchunk_size)


if hasattr(__builtins__,'__IPYTHON__'):
    ipytest.run('-qq')
