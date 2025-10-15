from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sqrt, sum as spark_sum
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .spark_session_manager import SparkSessionManager

logger = logging.getLogger(__name__)

class SparkSimilarityCalculator:
    def __init__(self):
        self.spark_manager = SparkSessionManager()
        self.spark = self.spark_manager.get_spark_session()

    def calculate_user_similarity(self, user_interactions_df, method="cosine"):
        try:
            user_product_matrix = self._create_user_product_matrix(user_interactions_df)

            if method == "cosine":
                similarity_matrix = self._cosine_similarity(user_product_matrix)
            elif method == "pearson":
                similarity_matrix = self._pearson_correlation(user_product_matrix)
            else:
                raise ValueError(f"Unsupported similarity method: {method}")

            return similarity_matrix

        except Exception as e:
            logger.error(f"Error calculating user similarity: {e}")
            raise

    def calculate_item_similarity(self, user_interactions_df, method="cosine"):
        try:
            item_user_matrix = self._create_item_user_matrix(user_interactions_df)

            if method == "cosine":
                similarity_matrix = self._cosine_similarity(item_user_matrix)
            elif method == "pearson":
                similarity_matrix = self._pearson_correlation(item_user_matrix)
            else:
                raise ValueError(f"Unsupported similarity method: {method}")

            return similarity_matrix

        except Exception as e:
            logger.error(f"Error calculating item similarity: {e}")
            raise

    def _create_user_product_matrix(self, interactions_df):
        user_product_df = interactions_df.groupBy("user_id", "product_id").agg(
            spark_sum("rating").alias("rating")
        ).fillna(0)

        pivot_df = user_product_df.groupBy("user_id").pivot("product_id").agg(
            {"rating": "first"}
        ).fillna(0)

        return pivot_df

    def _create_item_user_matrix(self, interactions_df):
        item_user_df = interactions_df.groupBy("product_id", "user_id").agg(
            spark_sum("rating").alias("rating")
        ).fillna(0)

        pivot_df = item_user_df.groupBy("product_id").pivot("user_id").agg(
            {"rating": "first"}
        ).fillna(0)

        return pivot_df

    def _cosine_similarity(self, matrix_df):
        feature_cols = [c for c in matrix_df.columns if c not in ["user_id", "product_id"]]

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        vector_df = assembler.transform(matrix_df)

        correlation_matrix = Correlation.corr(vector_df, "features", "spearman")

        return correlation_matrix

    def _pearson_correlation(self, matrix_df):
        feature_cols = [c for c in matrix_df.columns if c not in ["user_id", "product_id"]]

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        vector_df = assembler.transform(matrix_df)

        correlation_matrix = Correlation.corr(vector_df, "features", "pearson")

        return correlation_matrix

    def find_similar_users(self, user_id: str, similarity_matrix, top_k: int = 10):
        try:
            user_similarities = similarity_matrix.filter(col("user_id") == user_id)

            similar_users = user_similarities.orderBy(col("similarity").desc()).limit(top_k)

            return similar_users.collect()

        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return []

    def find_similar_items(self, item_id: str, similarity_matrix, top_k: int = 10):
        try:
            item_similarities = similarity_matrix.filter(col("product_id") == item_id)

            similar_items = item_similarities.orderBy(col("similarity").desc()).limit(top_k)

            return similar_items.collect()

        except Exception as e:
            logger.error(f"Error finding similar items: {e}")
            return []

    def batch_calculate_recommendations(self, users_batch: List[str], similarity_matrix, interactions_df, top_k: int = 10):
        recommendations = {}

        for user_id in users_batch:
            try:
                user_recs = self._get_user_recommendations(
                    user_id, similarity_matrix, interactions_df, top_k
                )
                recommendations[user_id] = user_recs

            except Exception as e:
                logger.error(f"Error calculating recommendations for user {user_id}: {e}")
                recommendations[user_id] = []

        return recommendations

    def _get_user_recommendations(self, user_id: str, similarity_matrix, interactions_df, top_k: int):
        similar_users = self.find_similar_users(user_id, similarity_matrix, top_k=20)

        if not similar_users:
            return []

        similar_user_ids = [row.user_id for row in similar_users]

        user_interactions = interactions_df.filter(col("user_id") == user_id)
        user_products = set([row.product_id for row in user_interactions.collect()])

        similar_user_interactions = interactions_df.filter(
            col("user_id").isin(similar_user_ids)
        ).filter(
            ~col("product_id").isin(list(user_products))
        )

        recommendations = similar_user_interactions.groupBy("product_id").agg(
            spark_sum("rating").alias("total_score")
        ).orderBy(col("total_score").desc()).limit(top_k)

        return [row.product_id for row in recommendations.collect()]

    def cleanup(self):
        if hasattr(self, 'spark_manager'):
            self.spark_manager.stop_spark_session()
