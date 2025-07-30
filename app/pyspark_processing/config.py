import os
import logging

logger = logging.getLogger(__name__)

SPARK_CONFIG = {
    "spark.app.name": "ML-Recommendation-System",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    "spark.sql.adaptive.localShuffleReader.enabled": "true"
}

SIMILARITY_METHODS = {
    "cosine": "cosine",
    "pearson": "pearson",
    "jaccard": "jaccard"
}

DEFAULT_BATCH_SIZE = 1000
DEFAULT_PARTITIONS = 200
MAX_RECOMMENDATIONS = 50

FEATURE_COLUMNS = [
    "user_age", "user_gender", "product_category",
    "product_price", "rating", "interaction_type"
]

SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "local[*]")
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "2g")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "1g")
SPARK_MAX_RESULT_SIZE = os.getenv("SPARK_MAX_RESULT_SIZE", "1g")
