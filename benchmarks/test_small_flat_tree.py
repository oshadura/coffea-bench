import pytest
import pyspark.sql

available_laurelin_version = [("edu.vanderbilt.accre:laurelin:0.5.0")]
files = [("samples/small-flat-tree.root", "samples/hepdata-example.root")]

def laurelin_read_simple_flat_tree(laurelin_version, file):
    spark = pyspark.sql.SparkSession.builder \
        .master("local[1]") \
        .config('spark.jars.packages', laurelin_version) \
        .getOrCreate()
    sc = spark.sparkContext
    df = spark.read.format('edu.vanderbilt.accre.laurelin.Root') \
            .option("tree", "tree") \
            .load(files)
    df.printSchema()

@pytest.mark.benchmark(group="laurelin-simple-root-tree")
@pytest.mark.parametrize("laurelin_version", available_laurelin_version)
@pytest.mark.parametrize("root_file", files)
def test_laurelin_read_simple_flat_tree(benchmark, laurelin_version, root_file):
    benchmark(laurelin_read_simple_flat_tree, laurelin_version, root_file)