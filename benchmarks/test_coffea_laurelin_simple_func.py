import pytest
import pyspark.sql
from pyarrow.compat import guid

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea.processor.spark.detail import (_spark_initialize,
                                           _spark_make_dfs,
                                           _spark_stop)
from coffea.processor.spark.spark_executor import spark_executor
from coffea.processor.test_items import NanoTestProcessor
from coffea.processor import run_spark_job

from pyspark.sql.types import BinaryType, StringType, StructType, StructField
import pyspark.sql.functions as fn

from jinja2 import Environment, PackageLoader, select_autoescape
import pickle as pkl
import lz4.frame as lz4f


file = {'test': ['file:///home/oksana/CERN_sources/examples-spark-ttree/nano_lgray.root']}
#'root://eospublic.cern.ch//eos/user/o/oshadura/coffea/nano_lgray.root'

available_laurelin_version = [("edu.vanderbilt.accre:laurelin:0.5.2-SNAPSHOT")]
    
def spark_session_startup():
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test-%s' % guid()) \
        .master('local[*]') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '4g') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', 200000)
    spark_session = _spark_initialize(config=spark_config, log_level='WARN', 
                          spark_progress=False, laurelin_version='0.5.2-SNAPSHOT')
    return spark_session

def laurelin_read_loading(laurelin_version, file, treename, spark_session):
    df = spark_session.read.format('root') \
            .option("tree", "Events") \
            .load(file['test'])
    df.printSchema()
    
def laurelin_read_select(laurelin_version, file, treename, df):
    df = df.select(*['nMuon','Muon_pt','Muon_eta','Muon_phi','Muon_mass'])
    
def laurelin_read_show(laurelin_version, file, treename, df):
    df = df.withColumn('dataset', fn.lit('test'))
    
def laurelin_read_show(laurelin_version, file, treename, df):
    df = df.withColumn('dataset', fn.lit('test'))
    
def laurelin_simple_test(laurelin_version, file, treename, df):
    env = Environment(loader=PackageLoader('coffea.processor',
                                           'templates'),
                      autoescape=select_autoescape(['py']))
    columns = ['nMuon','Muon_pt','Muon_eta','Muon_phi','Muon_mass']
    cols_w_ds = ['dataset','nMuon','Muon_pt','Muon_eta','Muon_phi','Muon_mass']
    processor_instance = NanoTestProcessor(columns=columns)
    tmpl = env.get_template('spark.py.tmpl')
    render = tmpl.render(cols=columns)
    exec(render)
    histdf = df.select(coffea_udf(*cols_w_ds).alias('histos'))
    pds = histdf.toPandas()
    print(pds)

@pytest.mark.benchmark(group="laurelin-simple-func")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
def test_laurelin_read_loading(benchmark, laurelin_version):
    spark_session = spark_session_startup()
    benchmark(laurelin_read_loading, laurelin_version, file, spark_session)
    
@pytest.mark.benchmark(group="laurelin-simple-func")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
def test_laurelin_read_select(benchmark, laurelin_version):
    spark_session = spark_session_startup()
    df = laurelin_read_loading(laurelin_version, file, spark_session)
    benchmark(laurelin_read_select, laurelin_version, file, df)

@pytest.mark.benchmark(group="laurelin-simple-func")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
def test_laurelin_read_show(benchmark, laurelin_version):
    spark_session = spark_session_startup()
    df = laurelin_read_loading(laurelin_version, file, spark_session)
    benchmark(laurelin_read_show, laurelin_version, file, df)
    
@pytest.mark.benchmark(group="laurelin-simple-func")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
def test_laurelin_simple_test(benchmark, laurelin_version):
    spark_session = spark_session_startup()
    df = laurelin_read_loading(laurelin_version, file, spark_session)
    benchmark(laurelin_simple_test, laurelin_version, file,  df)

