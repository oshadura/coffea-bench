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

# spark.jars.packages doesnt work with Spark 2.4 with kubernetes
# !wget -N https://repo1.maven.org/maven2/edu/vanderbilt/accre/laurelin/1.0.0/laurelin-1.0.0.jar
# !wget -N https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.11.2/log4j-api-2.11.2.jar
# !wget -N https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.11.2/log4j-core-2.11.2.jar
# !wget -N https://repo1.maven.org/maven2/org/lz4/lz4-java/1.5.1/lz4-java-1.5.1.jar
# !wget -N https://repo1.maven.org/maven2/org/tukaani/xz/1.2/xz-1.2.jar


if hasattr(__builtins__,'__IPYTHON__'):
    import os
    import ipytest
    ipytest.config(rewrite_asserts=True, magics=True)
    __file__ = 'test_coffea_dask_adl_example1.ipynb'
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

available_laurelin_version = [("edu.vanderbilt.accre:laurelin:1.0.1-SNAPSHOT")]

if 'DASK_COFFEABENCH' in os.environ:
    from dask.distributed import Client, LocalCluster
    from dask_jobqueue import HTCondorCluster

fileset = {
    'Jets Masked by Leptons': { 'files': ['root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root'],
             'treename': 'Events'
            }
}

class JetLeptonProcessor(processor.ProcessorABC):
    def __init__(self):
        self._columns = ['MET_pt', 'nMuon', 'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_mass', 'Muon_charge',
                         'nJet', 'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass', 'Jet_charge', 
                         'nElectron', 'Electron_pt', 'Electron_eta', 'Electron_phi', 'Electron_mass', 'Electron_charge',
                         ]
        dataset_axis = hist.Cat("dataset", "")
        muon_axis = hist.Bin("Jet_pt", "Jet_pt [GeV]", 100, 15, 200)   
        self._accumulator = processor.dict_accumulator({
            'Jet_pt': hist.Hist("Counts", dataset_axis, muon_axis),
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
        # Unfortunately, there's two different types of leptons here, so we need to create three JCA's (one for each, one for jets)
        muons = JaggedCandidateArray.candidatesfromcounts(
            df['nMuon'],
            pt=df['Muon_pt'].content,
            eta=df['Muon_eta'].content,
            phi=df['Muon_phi'].content,
            mass=df['Muon_mass'].content,
            charge=df['Muon_charge'].content
            )
        electrons = JaggedCandidateArray.candidatesfromcounts(
            df['nElectron'],
            pt=df['Electron_pt'].content,
            eta=df['Electron_eta'].content,
            phi=df['Electron_phi'].content,
            mass=df['Electron_mass'].content,
            charge=df['Electron_charge'].content
            )
        jets = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt=df['Jet_pt'].content,
            eta=df['Jet_eta'].content,
            phi=df['Jet_phi'].content,
            mass=df['Jet_mass'].content,
            )
        
        output['cutflow']['all events'] += jets.size
        output['cutflow']['all jets'] += jets.counts.sum()
        
        # Get jets with higher GeV than 30.
        min_jetpt = (jets['p4'].pt > 30)
        output['cutflow']['jets with pt > 30'] += min_jetpt.sum().sum()
        
        # Get all leptons with higher GeV than 10.
        min_muonpt = (muons['p4'].pt > 10)
        output['cutflow']['muons with pt > 10'] += min_muonpt.sum().sum()
        min_electronpt = (electrons['p4'].pt > 10)
        output['cutflow']['electrons with pt > 10'] += min_electronpt.sum().sum()
        
        # Mask jets and leptons with their minimum requirements/
        goodjets = jets[min_jetpt]
        goodmuons = muons[min_muonpt]
        goodelectrons = electrons[min_electronpt]
        
        # Cross is like distincts, but across multiple JCA's. So we cross jets with each lepton to generate all (jet, lepton) pairs. We have nested=True so that all jet values are stored in sublists together, and thus maintain uniqueness so we can get them back later.
        jet_muon_pairs = goodjets['p4'].cross(goodmuons['p4'], nested=True)
        jet_electron_pairs = goodjets['p4'].cross(goodelectrons['p4'], nested=True)
    
        # This long conditional checks that the jet is at least 0.4 euclidean distance from each lepton. It then checks if each unique jet contains a False, i.e., that a jet is 0.4 euclidean distance from EVERY specific lepton in the event.
        good_jm_pairs = (jet_muon_pairs.i0.delta_r(jet_muon_pairs.i1) > 0.4).all() != False
        good_je_pairs = (jet_electron_pairs.i0.delta_r(jet_electron_pairs.i1) > 0.4).all() != False
        
        output['cutflow']['jet-muon pairs'] += good_jm_pairs.sum().sum()
        output['cutflow']['jet-electron pairs'] += good_je_pairs.sum().sum()
        output['cutflow']['jet-lepton pairs'] += (good_jm_pairs & good_je_pairs).sum().sum()
        
        # We then mask our jets with all three of the above good pairs to get only jets that are 0.4 distance from every type of lepton, and sum them.
        sumjets = goodjets['p4'][good_jm_pairs & good_je_pairs].pt.sum()
        output['cutflow']['final jets'] += goodjets['p4'][good_jm_pairs & good_je_pairs].counts.sum()
        output['Jet_pt'].fill(dataset=dataset, Jet_pt=sumjets.flatten())
        
        return output

    def postprocess(self, accumulator):
        return accumulator

if 'DASK_COFFEABENCH' in os.environ:
    def test_dask_adlexample7(benchmark):
        @benchmark
        def dask_adlexample7(n_cores=2):
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
                                      processor_instance = JetLeptonProcessor(),
                                      executor = processor.dask_executor,
                                      executor_args = exe_args
                                      
            )
            return output 


def coffea_laurelin_adl_example7(laurelin_version, n_workers, partition_size):
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
                                     JetLeptonProcessor(),
                                     spark_executor, 
                                     spark=spark,
                                     partitionsize=partition_size,
                                     thread_workers=n_workers,
                                     executor_args={'file_type': 'edu.vanderbilt.accre.laurelin.Root', 'cache': False})

if 'PYSPARK_COFFEABENCH' in os.environ:
    @pytest.mark.benchmark(group="coffea-laurelin-adl-example7")
    @pytest.mark.parametrize("laurelin_version", available_laurelin_version)
    @pytest.mark.parametrize("n_workers", range(1,psutil.cpu_count(logical=False)))
    @pytest.mark.parametrize("partition_size", range(100000,200000,100000))
    def test_coffea_laurelin_adl_example7(benchmark,laurelin_version, n_workers, partition_size):
        benchmark(coffea_laurelin_adl_example7, available_laurelin_version, n_workers, partition_size)
    
    
def coffea_uproot_adl_example7(n_workers, chunk_size, maxchunk_size):
    output = processor.run_uproot_job(fileset,
                                      treename = 'Events',
                                      processor_instance = JetLeptonProcessor(),
                                      executor = processor.futures_executor,
                                      chunksize = chunk_size,
                                      maxchunks = maxchunk_size,
                                      executor_args = {'workers': n_workers}
                                      
    ) 

if 'UPROOT_COFFEABENCH' in os.environ:
    @pytest.mark.benchmark(group="coffea-uproot-adl-example7")
    @pytest.mark.parametrize("n_workers", range(1,psutil.cpu_count(logical=False)))
    @pytest.mark.parametrize("chunk_size", range(200000,600000,200000))
    @pytest.mark.parametrize("maxchunk_size", range(300000,700000,200000))
    def test_coffea_uproot_adl_example7(benchmark, n_workers, chunk_size, maxchunk_size):
        benchmark(coffea_uproot_adl_example7, n_workers, chunk_size, maxchunk_size)
    
if hasattr(__builtins__,'__IPYTHON__'):
    ipytest.run('-qq')

