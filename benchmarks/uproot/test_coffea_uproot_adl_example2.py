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

if hasattr(__builtins__,'__IPYTHON__'):
    import os
    import ipytest
    ipytest.config(rewrite_asserts=True, magics=True)
    __file__ = 'test_coffea_uproot_adl_example2.ipynb'
    # Run this cell before establishing spark connection <<<<< IMPORTANT
    os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':' + '/usr/local/lib/python3.6/site-packages'
    os.environ['PATH'] = os.environ['PATH'] + ':' + '/eos/user/o/oshadura/.local/bin'

import psutil
import pytest

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor

fileset = {
    'Jets': { 'files': ['root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root'],
             'treename': 'Events'
            }
}

# This program plots a per-event array (in this case, Jet pT). In Coffea, this is not very dissimilar from the event-level process.
class JetProcessor(processor.ProcessorABC):
    def __init__(self):
        self._columns = ['MET_pt', 'Jet_pt']
        dataset_axis = hist.Cat("dataset", "")
        Jet_axis = hist.Bin("Jet_pt", "Jet_pt [GeV]", 100, 15, 60)   
        self._accumulator = processor.dict_accumulator({
            'Jet_pt': hist.Hist("Counts", dataset_axis, Jet_axis),
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
        dataset = df['dataset']
        Jet_pt = df['Jet_pt']
        # As before, we can get the number of events by checking the size of the array. To get the number of jets, which varies per event, though, we need to count up the number in each event, and then sum those counts (count subarray sizes, sum them).
        output['cutflow']['all events'] += Jet_pt.size
        output['cutflow']['all jets'] += Jet_pt.counts.sum()
        output['Jet_pt'].fill(dataset=dataset, Jet_pt=Jet_pt.flatten())
        return output

    def postprocess(self, accumulator):
        return accumulator

def coffea_uproot_adl_example2(n_workers, chunk_size, maxchunk_size):
    output = processor.run_uproot_job(fileset,
                                      treename = 'Events',
                                      processor_instance = JetProcessor(),
                                      executor = processor.futures_executor,
                                      chunksize = chunk_size,
                                      maxchunks = maxchunk_size,
                                      executor_args = {'workers': n_workers}
                                      
    ) 

@pytest.mark.benchmark(group="coffea-uproot-adl-example2")
@pytest.mark.parametrize("n_workers", range(1,psutil.cpu_count(logical=False)))
@pytest.mark.parametrize("chunk_size", range(200000,600000,200000))
@pytest.mark.parametrize("maxchunk_size", range(300000,700000,200000))
def test_coffea_uproot_adl_example2(benchmark, n_workers, chunk_size, maxchunk_size):
    benchmark(coffea_uproot_adl_example2, n_workers, chunk_size, maxchunk_size)

if hasattr(__builtins__,'__IPYTHON__'):
    ipytest.run('-qq')
