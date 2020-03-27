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
    __file__ = 'test_coffea_dask_adl_example5.ipynb'
    # Run this cell before establishing spark connection <<<<< IMPORTANT
    os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':' + '/usr/local/lib/python3.6/site-packages'
    os.environ['PATH'] = os.environ['PATH'] + ':' + '/eos/user/o/oshadura/.local/bin'


import psutil
import pytest

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor

from dask.distributed import Client

fileset = {
    'MET Masked by Muons': { 'files': ['root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root'],
             'treename': 'Events'
            }
}

# This program will plot the MET for events which have an opposite-sign muon pair that has mass in the range of 60-120 GeV.
class METMuonProcessor(processor.ProcessorABC):
    def __init__(self):
        self._columns = ['MET_pt', 'nMuon', 'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_mass', 'Muon_charge']
        dataset_axis = hist.Cat("dataset", "")
        muon_axis = hist.Bin("MET", "MET [GeV]", 50, 1, 100)
        self._accumulator = processor.dict_accumulator({
            'MET': hist.Hist("Counts", dataset_axis, muon_axis),
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
        muons = JaggedCandidateArray.candidatesfromcounts(
            df['nMuon'],
            pt=df['Muon_pt'].content,
            eta=df['Muon_eta'].content,
            phi=df['Muon_phi'].content,
            mass=df['Muon_mass'].content,
            charge=df['Muon_charge'].content
            )
        MET = df['MET_pt']  
        output['cutflow']['all events'] += muons.size
        output['cutflow']['all muons'] += muons.mass.counts.sum()
        # Get all combinations of muon pairs in every event.
        dimuons = muons.distincts()
        output['cutflow']['all pairs'] += dimuons.mass.counts.sum()
        # Check that pairs have opposite charge.
        opposites = (dimuons.i0.charge != dimuons.i1.charge)
        # Get only muons with energy between 60 and 120.
        limits = (dimuons.mass >= 60) & (dimuons.mass < 120) 
        # Mask the dimuons with the opposites and the limits to get dimuons with opposite charge and mass between 60 and 120 GeV.
        good_dimuons = dimuons[opposites & limits]
        output['cutflow']['final pairs'] += good_dimuons.mass.counts.sum()
        # Mask the MET to get it only if an associated dimuon pair meeting the conditions exists.
        good_MET = MET[good_dimuons.counts >= 1]
        output['cutflow']['final events'] += good_MET.size
        output['MET'].fill(dataset=dataset, MET=good_MET.flatten())
        return output

    def postprocess(self, accumulator):
        return accumulator

def coffea_dask_adl_example5():
    # Dask settings (two different cases)
    #client = Client("t3.unl.edu:8786")
    cluster = HTCondorCluster(cores=2, memory="2GB",disk="1GB",dashboard_address=9998)
    cluster.scale(jobs=64)
    client = Client(cluster)
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
                                      processor_instance = METMuonProcessor(),
                                      executor = processor.dask_executor,
                                      executor_args = exe_args
                                      
    )
    return output 

@pytest.mark.benchmark(group="coffea-dask-adl-example5")
def test_coffea_dask_adl_example5(benchmark):
    benchmark(coffea_dask_adl_example5)

if hasattr(__builtins__,'__IPYTHON__'):
    ipytest.run('-qq')
