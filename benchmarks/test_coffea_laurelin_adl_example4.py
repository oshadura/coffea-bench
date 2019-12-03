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
    'MET Masked by Jet': { 'files': ['/home/oksana/CERN_sources/coffea-benchmarks/benchmarks/data/Run2012B_SingleMu.root'],
        #'MET Masked by Jet': { 'files': ['root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root'],
             'treename': 'Events'
            }
}

# parameters to be changed
available_laurelin_version = [("edu.vanderbilt.accre:laurelin:0.5.2-SNAPSHOT")]

# This program plots an event-level variable (MET) based on conditionals with its associated Jet arrays (in this case, where at least 2 have pT > 40)

class JetMETProcessor(processor.ProcessorABC):
    def __init__(self):
        self._columns = ['nJet', 'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass']
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
    
def coffea_laurelin_adl_example4(laurelin_version, fileset):
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test-%s' % guid()) \
        .master('local[*]') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '4g') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', 200000)

    spark = _spark_initialize(config=spark_config, log_level='WARN', 
                          spark_progress=False, laurelin_version='0.5.2-SNAPSHOT')
    
    output = processor.run_spark_job(fileset,
                                     JetMETProcessor(),
                                     spark_executor, 
                                     spark=spark,
                                     partitionsize=partitionsize,
                                     thread_workers=thread_workers,
                                     executor_args={'file_type': 'edu.vanderbilt.accre.laurelin.Root', 'cache': False})

@pytest.mark.benchmark(group="coffea-laurelin-adl-example4")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
@pytest.mark.parametrize("root_file", fileset)

def test_coffea_laurelin_adl_example4(benchmark, laurelin_version, root_file):
    benchmark(coffea_laurelin_adl_example4, laurelin_version, fileset)