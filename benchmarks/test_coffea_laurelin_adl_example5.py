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
    'MET Masked by Muons': { 'files': ['root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root'],
             'treename': 'Events'
            }
}

# parameters to be changed
available_laurelin_version = [("edu.vanderbilt.accre:laurelin:1.0.1-SNAPSHOT")]

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
    
def coffea_laurelin_adl_example5(laurelin_version, fileset):
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test-%s' % guid()) \
        .master('local[*]') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '4g') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', 200000)

    spark = _spark_initialize(config=spark_config, log_level='WARN', 
                          spark_progress=False, _version='1.0.1-SNAPSHOT')
    
    output = processor.run_spark_job(fileset,
                                     METMuonProcessor(),
                                     spark_executor, 
                                     spark=spark,
                                     partitionsize=partitionsize,
                                     thread_workers=thread_workers,
                                     executor_args={'file_type': 'edu.vanderbilt.accre.laurelin.Root', 'cache': False})


@pytest.mark.skip(reason="Dataset is too big! no way of currently testing this...")
@pytest.mark.benchmark(group="coffea-laurelin-adl-example5")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
@pytest.mark.parametrize("root_file", fileset)

def test_coffea_laurelin_adl_example5(benchmark, laurelin_version, root_file):
    benchmark(coffea_laurelin_adl_example5, laurelin_version, fileset)