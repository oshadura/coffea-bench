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
    __file__ = 'test_coffea_laurelin_simple_func.ipynb'
    # Run this cell before establishing spark connection <<<<< IMPORTANT
    os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':' + '/usr/local/lib/python3.6/site-packages'
    os.environ['PATH'] = os.environ['PATH'] + ':' + '/eos/user/o/oshadura/.local/bin'

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

# parameters to be changed
partitionsize = 200000
# parameters to be changed
thread_workers = 4
# parameters to be changed
available_laurelin_version = [("edu.vanderbilt.accre:laurelin:1.0.1-SNAPSHOT")]

fileset = {
    'test': { 'files': ['root://eosuser//eos/user/o/oshadura/coffea/nano_lgray.root'],
             'treename': 'Events'
            }
}

def spark_session_startup(laurelin_version):
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test-%s' % guid()) \
        .master('local[*]') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '6g') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', partitionsize) \
        .config('spark.kubernetes.container.image.pullPolicy', 'true') \
        .config('spark.kubernetes.container.image', 'gitlab-registry.cern.ch/db/spark-service/docker-registry/swan:laurelin') \
        .config('spark.kubernetes.memoryOverheadFactor', '0.1')
        
    spark_session = _spark_initialize(config=spark_config,
                                      log_level='WARN', 
                                      spark_progress=False,
                                      laurelin_version=laurelin_version)

def laurelin_read_loading(laurelin_version, file):
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test-%s' % guid()) \
        .master('local[*]') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '6g') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', partitionsize)\
        .config('spark.kubernetes.container.image.pullPolicy', 'true')\
        .config('spark.kubernetes.container.image', 'gitlab-registry.cern.ch/db/spark-service/docker-registry/swan:laurelin')\
        .config('spark.kubernetes.memoryOverheadFactor', '0.1')     
    spark_session = _spark_initialize(config=spark_config,
                                      log_level='WARN', 
                                      spark_progress=False,
                                      laurelin_version=laurelin_version)
    df = spark_session.read.format('edu.vanderbilt.accre.laurelin.Root') \
            .option("tree", "Events") \
            .load(file['test'])
    df.printSchema()
    return df

def laurelin_read_select(laurelin_version, file):
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test-%s' % guid()) \
        .master('local[*]') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '6g') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', partitionsize)\
        .config('spark.kubernetes.container.image.pullPolicy', 'true')\
        .config('spark.kubernetes.container.image', 'gitlab-registry.cern.ch/db/spark-service/docker-registry/swan:laurelin')\
        .config('spark.kubernetes.memoryOverheadFactor', '0.1')        
    spark_session = _spark_initialize(config=spark_config, log_level='WARN', 
                          spark_progress=False, laurelin_version='1.0.1-SNAPSHOT')
    return spark_session
    df = laurelin_read_loading(laurelin_version, file)
    df_final = df.select(*['nMuon','Muon_pt','Muon_eta','Muon_phi','Muon_mass'])
    df_final.printSchema()

def laurelin_read_show(laurelin_version, file):
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test-%s' % guid()) \
        .master('local[*]') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '6g') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', partitionsize)\
        .config('spark.kubernetes.container.image.pullPolicy', 'true')\
        .config('spark.kubernetes.container.image', 'gitlab-registry.cern.ch/db/spark-service/docker-registry/swan:laurelin')\
        .config('spark.kubernetes.memoryOverheadFactor', '0.1')      
    spark_session = _spark_initialize(config=spark_config,
                                      log_level='WARN', 
                                      spark_progress=False,
                                      laurelin_version=laurelin_version)

    df = laurelin_read_loading(laurelin_version, file)
    df_final = df.withColumn('dataset', fn.lit('test'))
    df_final.printSchema()

def laurelin_simple_test(laurelin_version, file):
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test-%s' % guid()) \
        .master('local[*]') \
        .config('spark.driver.memory', '4g') \
        .config('spark.executor.memory', '6g') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', partitionsize)\
        .config('spark.kubernetes.container.image.pullPolicy', 'true')\
        .config('spark.kubernetes.container.image', 'gitlab-registry.cern.ch/db/spark-service/docker-registry/swan:laurelin')\
        .config('spark.kubernetes.memoryOverheadFactor', '0.1')
        
    spark_session = _spark_initialize(config=spark_config,
                                      log_level='WARN', 
                                      spark_progress=False,
                                      laurelin_version=laurelin_version)
    df = laurelin_read_loading(laurelin_version, file)
    env = Environment(loader=PackageLoader('coffea.processor','templates'),autoescape=select_autoescape(['py']))
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

@pytest.mark.benchmark(group="laurelin-simple-func")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
@pytest.mark.parametrize("root_file", fileset)
def test_laurelin_read_loading(benchmark, laurelin_version):
    benchmark(laurelin_read_loading, laurelin_version, fileset)

@pytest.mark.benchmark(group="laurelin-simple-func")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
@pytest.mark.parametrize("root_file", fileset)
def test_laurelin_read_select(benchmark, laurelin_version):
    benchmark(laurelin_read_select, laurelin_version, fileset)

@pytest.mark.benchmark(group="laurelin-simple-func")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
@pytest.mark.parametrize("root_file", fileset)
def test_laurelin_read_show(benchmark, laurelin_version):
    benchmark(laurelin_read_show, laurelin_version, fileset)

@pytest.mark.benchmark(group="laurelin-simple-func")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
@pytest.mark.parametrize("root_file", fileset)
def test_laurelin_simple_test(benchmark, laurelin_version):
    benchmark(laurelin_simple_test, laurelin_version, fileset)

if hasattr(__builtins__,'__IPYTHON__'):
    ipytest.run('-qq')
