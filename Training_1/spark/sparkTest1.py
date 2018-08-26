from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("local[2]") \
    .getOrCreate()
    
    
df = spark.read.json("C:\MyWorkspace\Application\spark-2.2.0-bin-hadoop2.7\examples/src/main/resources/people.json")