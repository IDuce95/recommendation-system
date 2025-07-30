import logging
from typing import Dict, List, Optional, Any
import os

from app.feature_store import FeatureStore, FeatureStoreConfig

logger = logging.getLogger(__name__)

class RecommendationFeatureStore:
    def __init__(self):
        config = FeatureStoreConfig(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            default_ttl=3600,  # 1 hour
            embedding_ttl=86400,  # 24 hours for embeddings
            enable_versioning=True,
            enable_monitoring=True
        )

        try:
            self.feature_store = FeatureStore(config)
            logger.info("Recommendation Feature Store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Feature Store: {e}")
            self.feature_store = None

    def is_available(self) -> bool:

        if not self.feature_store:
            return False

        try:
            health = self.feature_store.health_check()
            return health.get("redis_connected", False)
        except Exception:
            return False

    def store_product_features(
        self,
        product_id: int,
        product_data: Dict[str, Any],
        embeddings: Optional[Dict[str, Any]] = None
    ) -> bool:

        if not self.is_available():
            logger.warning("Feature Store not available, skipping feature storage")
            return False

        try:
            features = {
                "product_id": product_id,
                "name": product_data.get("name", ""),
                "category": product_data.get("category", ""),
                "description": product_data.get("description", ""),
                "image_path": product_data.get("image_path", ""),
                "brand": product_data.get("brand", "")
            }

            if embeddings:
                if "text_embedding" in embeddings:
                    features["text_embedding"] = embeddings["text_embedding"]
                if "image_embedding" in embeddings:
                    features["image_embedding"] = embeddings["image_embedding"]

            success = self.feature_store.write_features(
                entity_id=f"product_{product_id}",
                features=features,
                feature_group="product_features"
            )

            if success:
                logger.debug(f"Stored features for product {product_id}")
            else:
                logger.warning(f"Failed to store features for product {product_id}")

            return success

        except Exception as e:
            logger.error(f"Error storing product features: {e}")
            return False

    def get_product_features(
        self,
        product_id: int,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:

        if not self.is_available():
            return {}

        try:
            if feature_names is None:
                feature_names = [
                    "name", "category", "description",
                    "text_embedding", "image_embedding"
                ]

            features = self.feature_store.get_features(
                entity_id=f"product_{product_id}",
                feature_names=feature_names
            )

            logger.debug(f"Retrieved {len(features)} features for product {product_id}")
            return features

        except Exception as e:
            logger.error(f"Error retrieving product features: {e}")
            return {}

    def get_batch_product_features(
        self,
        product_ids: List[int],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:

        if not self.is_available():
            return {}

        try:
            entity_ids = [f"product_{pid}" for pid in product_ids]

            if feature_names is None:
                feature_names = [
                    "name", "category", "description",
                    "text_embedding", "image_embedding"
                ]

            batch_features = self.feature_store.get_features_batch(
                entity_ids=entity_ids,
                feature_names=feature_names
            )

            result = {}
            for entity_id, features in batch_features.items():
                product_id = int(entity_id.replace("product_", ""))
                result[product_id] = features

            logger.debug(f"Retrieved batch features for {len(result)} products")
            return result

        except Exception as e:
            logger.error(f"Error retrieving batch product features: {e}")
            return {}

    def get_recommendation_features(
        self,
        product_ids: List[int]
    ) -> Dict[str, Any]:

        recommendation_features = [
            "text_embedding",
            "image_embedding",
            "category",
            "name"
        ]

        batch_features = self.get_batch_product_features(
            product_ids,
            recommendation_features
        )

        return batch_features

    def compute_and_cache_features(
        self,
        product_id: int,
        raw_product_data: Dict[str, Any]
    ) -> bool:

        if not self.is_available():
            return False

        try:
            success = self.feature_store.compute_and_store_features(
                entity_id=f"product_{product_id}",
                raw_data=raw_product_data
            )

            if success:
                logger.info(f"Computed and cached features for product {product_id}")

            return success

        except Exception as e:
            logger.error(f"Error computing features for product {product_id}: {e}")
            return False

    def get_feature_statistics(self) -> Dict[str, Any]:

        if not self.is_available():
            return {"error": "Feature Store not available"}

        try:
            return self.feature_store.get_performance_metrics()
        except Exception as e:
            logger.error(f"Error getting feature statistics: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:

        if not self.feature_store:
            return {
                "status": "unavailable",
                "error": "Feature Store not initialized"
            }

        try:
            return self.feature_store.health_check()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

recommendation_feature_store = RecommendationFeatureStore()

def get_feature_store() -> RecommendationFeatureStore:

    return recommendation_feature_store
