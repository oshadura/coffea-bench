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


#file = {'test': ['root://eospublic.cern.ch//eos/user/o/oshadura/coffea/nano_lgray.root']}

file = {
    'test': { 'files': ['root://eosuser//eos/user/o/oshadura/coffea/nano_lgray.root'],
             'treename': 'Events'
            }
}

available_laurelin_version = [("edu.vanderbilt.accre:laurelin:1.0.1-SNAPSHOT")]
    
def spark_session_startup():
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test') \
        .master('local[*]') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '4g') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', 200000)
    spark_session = _spark_initialize(config=spark_config, log_level='WARN', 
                          spark_progress=False, laurelin_version='1.0.1-SNAPSHOT')
    return spark_session

def laurelin_read_loading(laurelin_version, file):
    spark_session = spark_session_startup()
    df = spark_session.read.format('edu.vanderbilt.accre.laurelin.Root') \
            .option("tree", "Events") \
            .load(file['test'])
    df.printSchema()
    return df
    
def laurelin_read_select(laurelin_version, file):
    spark_session = spark_session_startup()
    df = laurelin_read_loading(laurelin_version, file)
    df_final = df.select(*['nMuon','Muon_pt','Muon_eta','Muon_phi','Muon_mass'])
    df_final.printSchema()
    
def laurelin_read_show(laurelin_version, file):
    spark_session = spark_session_startup()
    df = laurelin_read_loading(laurelin_version, file)
    df_final = df.withColumn('dataset', fn.lit('test'))
    df_final.printSchema()
    
def laurelin_simple_test(laurelin_version, file):
    spark_session = spark_session_startup()
    df = laurelin_read_loading(laurelin_version, file)
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
    
@pytest.mark.benchmark(group="laurelin-simple-startup")
def test_spark_session_startup(benchmark):
    benchmark(spark_session_startup)

@pytest.mark.skip(reason="No way of currently testing this...")
@pytest.mark.benchmark(group="laurelin-simple-func")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
def test_laurelin_read_loading(benchmark, laurelin_version):
    benchmark(laurelin_read_loading, laurelin_version, file)
 
@pytest.mark.skip(reason="No way of currently testing this...")    
@pytest.mark.benchmark(group="laurelin-simple-func")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
def test_laurelin_read_select(benchmark, laurelin_version):
    benchmark(laurelin_read_select, laurelin_version, file)

@pytest.mark.skip(reason="No way of currently testing this...")
@pytest.mark.benchmark(group="laurelin-simple-func")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
def test_laurelin_read_show(benchmark, laurelin_version):
    benchmark(laurelin_read_show, laurelin_version, file)

@pytest.mark.skip(reason="No way of currently testing this...")    
@pytest.mark.benchmark(group="laurelin-simple-func")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
def test_laurelin_simple_test(benchmark, laurelin_version):
    benchmark(laurelin_simple_test, laurelin_version, file)

