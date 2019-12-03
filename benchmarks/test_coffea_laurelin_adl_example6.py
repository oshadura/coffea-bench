import pytest
import pyspark.sql
from pyarrow.compat import guid

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea.processor.spark.detail import _spark_initialize, _spark_stop
from coffea.processor.spark.spark_executor import spark_executor

import numpy as np

# parameters to be changed
partitionsize = 200000
# parameters to be changed
thread_workers = 1

fileset = {
    'Trijets': { 'files': ['/home/oksana/CERN_sources/coffea-benchmarks/benchmarks/data/Run2012B_SingleMu.root'],
        #'Trijets': { 'files': ['root://eospublic.cern.ch//eos/root-eos/benchmark/Run2012B_SingleMu.root'],
             'treename': 'Events'
            }
}

available_laurelin_version = [("edu.vanderbilt.accre:laurelin:0.5.2-SNAPSHOT")]

# This program plots the pT of the trijet system in each event with mass closest to 172.5, as well as the maximum b-tag among the three plotted jets.

class TrijetProcessor(processor.ProcessorABC):
    def __init__(self):
        self._columns = ['nJet', 'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass', 'Jet_btag']
        dataset_axis = hist.Cat("dataset", "")
        Jet_axis = hist.Bin("Jet_pt", "Jet [GeV]", 50, 15, 200)
        b_tag_axis = hist.Bin("b_tag", "b-tagging discriminant", 50, 0, 1)
        self._accumulator = processor.dict_accumulator({
            'Jet_pt': hist.Hist("Counts", dataset_axis, Jet_axis),
            'b_tag': hist.Hist("Counts", dataset_axis, b_tag_axis),
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
            b_tag=df['Jet_btag'].content
            )
        
        # Closest calculates the distance from 172.5 of a group of masses, finds the minimum distance, then returns a Boolean array of the original input array shape with True where the minimum-distance mass is located.
        def closest(masses):
            delta = abs(172.5 - masses)
            closest_masses = delta.min()
            is_closest = (delta == closest_masses)
            return is_closest
        
        # We're going to be generating combinations of three jets - that's a lot, and cutting pt off at 30 reduces jets by half.
        cut_jets = jets[jets.pt > 30]
        
        # Get all combinations of three jets.
        trijets = cut_jets.choose(3)
        # Get combined masses of those combinations.
        trijet_masses = trijets.mass
        # Get the masses closest to specified value (see function above)
        is_closest = closest(trijet_masses)
        closest_trijets = trijets[is_closest]
        # Get pt of the closest trijets.
        closest_pt = closest_trijets.pt
        # Get btag of the closest trijets. np.maximum(x,y) compares two arrays and gets element-wise maximums. We make two comparisons - once between the first and second jet, then between the first comparison and the third jet.
        closest_btag = np.maximum(np.maximum(closest_trijets.i0['b_tag'], closest_trijets.i1['b_tag']), closest_trijets.i2['b_tag'])
        
        output['Jet_pt'].fill(dataset=dataset, Jet_pt=closest_pt.flatten())
        output['b_tag'].fill(dataset=dataset, b_tag=closest_btag.flatten())
        return output

    def postprocess(self, accumulator):
        return accumulator
    
def coffea_laurelin_adl_example6(laurelin_version, fileset):
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
                                     TrijetProcessor(),
                                     spark_executor, 
                                     spark=spark,
                                     partitionsize=partitionsize,
                                     thread_workers=thread_workers,
                                     executor_args={'file_type': 'edu.vanderbilt.accre.laurelin.Root', 'cache': False})


@pytest.mark.benchmark(group="coffea-laurelin-adl-example6")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
@pytest.mark.parametrize("root_file", fileset)

def test_coffea_laurelin_adl_example6(benchmark, laurelin_version, root_file):
    benchmark(coffea_laurelin_adl_example6, laurelin_version, fileset)