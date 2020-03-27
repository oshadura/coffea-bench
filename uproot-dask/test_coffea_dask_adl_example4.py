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


if hasattr(__builtins__,'__IPYTHON__'):
    import os
    import ipytest
    ipytest.config(rewrite_asserts=True, magics=True)
    __file__ = 'test_coffea_dask_adl_example4.ipynb'
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

def test_dask_adlexample4(benchmark):
    @benchmark
    def dask_adlexample4(n_cores=2):
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
                                      processor_instance = JetMETProcessor(),
                                      executor = processor.dask_executor,
                                      executor_args = exe_args
                                      
        )
        return output  

if hasattr(__builtins__,'__IPYTHON__'):
    ipytest.run('-qq')
