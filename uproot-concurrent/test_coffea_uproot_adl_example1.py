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
    __file__ = 'test_coffea_uproot_adl_example1.ipynb'
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

# This program plots an event-level variable (in this case, MET, but switching it is as easy as a dict-key change). It also demonstrates an easy use of the book-keeping cutflow tool, to keep track of the number of events processed.
# The processor class bundles our data analysis together while giving us some helpful tools.  It also leaves looping and chunks to the framework instead of us.
class METProcessor(processor.ProcessorABC):
    def __init__(self):
        # Bins and categories for the histogram are defined here. For format, see https://coffeateam.github.io/coffea/stubs/coffea.hist.hist_tools.Hist.html && https://coffeateam.github.io/coffea/stubs/coffea.hist.hist_tools.Bin.html
        self._columns = ['MET_pt']
        dataset_axis = hist.Cat("dataset", "")
        MET_axis = hist.Bin("MET", "MET [GeV]", 50, 0, 100)
        # The accumulator keeps our data chunks together for histogramming. It also gives us cutflow, which can be used to keep track of data.
        self._accumulator = processor.dict_accumulator({
            'MET': hist.Hist("Counts", dataset_axis, MET_axis),
            'cutflow': processor.defaultdict_accumulator(int)
        })

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    def process(self, df):
        output = self.accumulator.identity()
        # This is where we do our actual analysis. The df has dict keys equivalent to the TTree's.
        dataset = df['dataset']
        MET = df['MET_pt']
        # We can define a new key for cutflow (in this case 'all events'). Then we can put values into it. We need += because it's per-chunk (demonstrated below)
        output['cutflow']['all events'] += MET.size
        output['cutflow']['number of chunks'] += 1
        # This fills our histogram once our data is collected. Always use .flatten() to make sure the array is reduced. The output key will be as defined in __init__ for self._accumulator; the hist key ('MET=') will be defined in the bin.
        output['MET'].fill(dataset=dataset, MET=MET.flatten())
        return output

    def postprocess(self, accumulator):
        return accumulator

def coffea_uproot_adl_example1(n_workers, chunk_size, maxchunk_size):
    output = processor.run_uproot_job(fileset,
                                      treename = 'Events',
                                      processor_instance = METProcessor(),
                                      executor = processor.futures_executor,
                                      chunksize = chunk_size,
                                      maxchunks = maxchunk_size,
                                      executor_args = {'workers': n_workers}
                                      
    ) 

@pytest.mark.benchmark(group="coffea-uproot-adl-example1")
@pytest.mark.parametrize("n_workers", range(1,psutil.cpu_count(logical=False)))
@pytest.mark.parametrize("chunk_size", range(200000,600000,200000))
@pytest.mark.parametrize("maxchunk_size", range(300000,700000,200000))
def test_coffea_uproot_adl_example1(benchmark, n_workers, chunk_size, maxchunk_size):
    benchmark(coffea_uproot_adl_example1, n_workers, chunk_size, maxchunk_size)

if hasattr(__builtins__,'__IPYTHON__'):
    ipytest.run('-qq')