from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import os
import logging

logger = logging.getLogger(__name__)

class SparkSessionManager:
    _instance = None
    _spark = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_spark_session(self):
        if self._spark is None:
            self._spark = self._create_spark_session()
        return self._spark

    def _create_spark_session(self):
        conf = SparkConf().setAppName("ML-Recommendation-System")

        conf.set("spark.sql.adaptive.enabled", "true")
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        logger.info("Spark session created successfully")
        return spark

    def stop_spark_session(self):
        if self._spark:
            self._spark.stop()
            self._spark = None
            logger.info("Spark session stopped")

    def __del__(self):
        self.stop_spark_session()
