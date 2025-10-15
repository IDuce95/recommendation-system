import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark_session_manager import SparkSessionManager

def test_spark_setup():
    print("Testing Spark setup...")
    
    try:
        spark_manager = SparkSessionManager()
        spark = spark_manager.get_spark_session()
        
        print(f"Spark version: {spark.version}")
        print(f"Spark master: {spark.sparkContext.master}")
        print(f"Spark app name: {spark.sparkContext.appName}")
        
        test_data = [(1, "Alice", 25), (2, "Bob", 30), (3, "Charlie", 35)]
        columns = ["id", "name", "age"]
        
        df = spark.createDataFrame(test_data, columns)
        print("Created test DataFrame:")
        df.show()
        
        print("DataFrame count:", df.count())
        
        spark_manager.stop_spark_session()
        print("Spark setup test completed successfully!")
        
    except Exception as e:
        print(f"Error in Spark setup test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_spark_setup()
