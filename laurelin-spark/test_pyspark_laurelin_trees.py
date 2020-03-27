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
# !pip install --user --upgrade pytest-csv

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
    __file__ = 'test_pyspark_laurelin_trees.ipynb'
    # Run this cell before establishing spark connection <<<<< IMPORTANT
    os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':' + '/usr/local/lib/python3.6/site-packages'
    os.environ['PATH'] = os.environ['PATH'] + ':' + '/eos/user/o/oshadura/.local/bin'

import pytest
import pyspark.sql
import glob
import re

files = [f for f in glob.glob("samples/*.root")]
available_laurelin_version = [("edu.vanderbilt.accre:laurelin:1.0.1-SNAPSHOT")]

class RegexSwitch(object):
  def __init__(self):
    self.last_match = None
  def match(self,pattern,text):
    self.last_match = re.match(pattern,text)
    return self.last_match
  def search(self,pattern,text):
    self.last_match = re.search(pattern,text)
    return self.last_match

def laurelin_read_simple_flat_tree(laurelin_version, file):
    gre = RegexSwitch()
    spark = pyspark.sql.SparkSession.builder \
        .master("local[1]") \
        .config('spark.jars.packages', laurelin_version) \
        .getOrCreate()
    sc = spark.sparkContext
    if gre.match(r'sample',file):
        treename = "sample"
    elif gre.match(r'HZZ-objects',file) or gre.match(r'Zmumu',file):
        treename = "events"
    else:
        treename = "tree"
    df = spark.read.format('edu.vanderbilt.accre.laurelin.Root') \
            .option("tree", treename) \
            .load(files)
    df.printSchema()

@pytest.mark.benchmark(group="laurelin-simple-root-tree")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
@pytest.mark.parametrize("root_file", files)
def test_laurelin_read_simple_flat_tree(benchmark, laurelin_version, root_file):
    benchmark(laurelin_read_simple_flat_tree, laurelin_version, root_file)

if hasattr(__builtins__,'__IPYTHON__'):
    ipytest.run('-qq')
