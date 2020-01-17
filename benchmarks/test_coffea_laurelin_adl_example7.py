import pytest
import pyspark.sql
from pyarrow.compat import guid

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea.processor.spark.detail import _spark_initialize, _spark_stop
from coffea.processor.spark.spark_executor import spark_executor

# parameters to be changed
partitionsize = 200000
# parameters to be changed
thread_workers = 1

fileset = {
    'Jets Masked by Leptons': { 'files': ['root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root'],
             'treename': 'Events'
            }
}

# parameters to be changed
available_laurelin_version = [("edu.vanderbilt.accre:laurelin:1.0.1-SNAPSHOT")]

# This program will graph the sum of Jet pT's which are greater than 30 GeV and farther than a Euclidean distance of 0.4 from any lepton with pT > 10 GeV.
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
    
def coffea_laurelin_adl_example7(laurelin_version, fileset):
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test-%s' % guid()) \
        .master('local[*]') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '4g') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', 200000)

    spark = _spark_initialize(config=spark_config, log_level='WARN', 
                          spark_progress=False, laurelin_version='1.0.1-SNAPSHOT')
    
    output = processor.run_spark_job(fileset,
                                     JetLeptonProcessor(),
                                     spark_executor, 
                                     spark=spark,
                                     partitionsize=partitionsize,
                                     thread_workers=thread_workers,
                                     executor_args={'file_type': 'edu.vanderbilt.accre.laurelin.Root', 'cache': False})

@pytest.mark.skip(reason="Dataset is too big! no way of currently testing this...")
@pytest.mark.benchmark(group="coffea-laurelin-adl-example7")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
@pytest.mark.parametrize("root_file", fileset)

def test_coffea_laurelin_adl_example7(benchmark, laurelin_version, root_file):
    benchmark(coffea_laurelin_adl_example7, laurelin_version, fileset)