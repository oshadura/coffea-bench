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
thread_workers = 2

fileset = {
    'Jets': { 'files': ['/home/oksana/CERN_sources/coffea-benchmarks/benchmarks/data/Run2012B_SingleMu.root'],
        #'Jets': { 'files': ['root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root'],
             'treename': 'Events'
            }
}

# parameters to be changed
available_laurelin_version = [("edu.vanderbilt.accre:laurelin:0.5.2-SNAPSHOT")]

# This program plots a per-event array (in this case, Jet pT). In Coffea, this is not very dissimilar from the event-level process.
class JetProcessor(processor.ProcessorABC):
    def __init__(self):
        self._columns = ['Jet_pt']
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
    
def coffea_laurelin_adl_example2(laurelin_version, fileset):
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
                                     JetProcessor(),
                                     spark_executor,
                                     spark=spark,
                                     partitionsize=partitionsize,
                                     thread_workers=thread_workers,
                                     executor_args={'file_type': 'edu.vanderbilt.accre.laurelin.Root', 'cache': False})

@pytest.mark.benchmark(group="coffea-laurelin-adl-example2")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
@pytest.mark.parametrize("root_file", fileset)

def test_coffea_laurelin_adl_example2(benchmark, laurelin_version, root_file):
    benchmark(coffea_laurelin_adl_example2, laurelin_version, fileset)