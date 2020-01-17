import pytest
import pyspark.sql
from pyarrow.compat import guid

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from coffea.processor.spark.detail import _spark_initialize, _spark_stop
from coffea.processor.spark.spark_executor import spark_executor
import coffea.processor as processor

# parameters to be changed
partitionsize = 200000
# parameters to be changed
thread_workers = 2

fileset = {
    'DoubleMuon': { 'files': [
        'root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root',
        'root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012C_DoubleMuParked.root',
                             ], 
                    'treename': 'Events'
                  }
}
available_laurelin_version = [("edu.vanderbilt.accre:laurelin:1.0.1-SNAPSHOT")]

# Look at ProcessorABC documentation to see the expected methods and what they are supposed to do
class DimuonProcessor(processor.ProcessorABC):
    def __init__(self):
        self._columns = ['nMuon', 'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_mass', 'Muon_charge']
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        mass_axis = hist.Bin("mass", r"$m_{\mu\mu}$ [GeV]", 30000, 0.25, 300)
        self._accumulator = processor.dict_accumulator({
            'mass': hist.Hist("Counts", dataset_axis, mass_axis),
            'cutflow': processor.defaultdict_accumulator(int),
        })
    
    @property
    def accumulator(self):
        return self._accumulator
    
    @property
    def columns(self):
        return self._columns
    
    def process(self, df):
        output = self.accumulator.identity() 
        dataset = df['dataset']
        muons = JaggedCandidateArray.candidatesfromcounts(
            df['nMuon'],
            pt=df['Muon_pt'].content,
            eta=df['Muon_eta'].content,
            phi=df['Muon_phi'].content,
            mass=df['Muon_mass'].content,
            charge=df['Muon_charge'].content,
            )      
        output['cutflow']['all events'] += muons.size    
        twomuons = (muons.counts == 2)
        output['cutflow']['two muons'] += twomuons.sum()   
        opposite_charge = twomuons & (muons['charge'].prod() == -1)
        output['cutflow']['opposite charge'] += opposite_charge.sum()    
        dimuons = muons[opposite_charge].distincts()
        output['mass'].fill(dataset=dataset, mass=dimuons.mass.flatten())
        return output

    def postprocess(self, accumulator):
        return accumulator
    
def spark_session_startup():
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test-%s' % guid()) \
        .master('local[*]') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '4g') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', 200000)
    spark = _spark_initialize(config=spark_config, log_level='WARN', 
                          spark_progress=False, laurelin_version='1.0.1-SNAPSHOT')
    return spark

def coffea_laurelin_dimuon_analysis(laurelin_version, fileset, spark):
    output = processor.run_spark_job(fileset,
                                     DimuonProcessor(),
                                     spark_executor, 
                                     spark=spark,
                                     partitionsize=partitionsize,
                                     thread_workers=thread_workers,
                                     executor_args={'file_type': 'edu.vanderbilt.accre.laurelin.Root', 'cache': False})

@pytest.mark.skip(reason="Dataset is too big! no way of currently testing this...")
@pytest.mark.benchmark(group="coffea-laurelin-dimuon-analysis")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
@pytest.mark.parametrize("root_file", fileset)

def test_coffea_laurelin_dimuon_analysis(benchmark, laurelin_version, root_file):
    spark = spark_session_startup()
    benchmark(coffea_laurelin_dimuon_analysis, laurelin_version, fileset, spark)
      