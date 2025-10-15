from pyspark.sql import SparkSession

def simple_spark_test():
    spark = SparkSession.builder.appName("SimpleTest").master("local[*]").getOrCreate()
    
    data = [("Alice", 25), ("Bob", 30)]
    columns = ["name", "age"]
    
    df = spark.createDataFrame(data, columns)
    df.show()
    
    spark.stop()
    print("Simple Spark test completed!")

if __name__ == "__main__":
    simple_spark_test()
