import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd

from .config import FeatureStoreConfig, DEFAULT_CONFIG, PRODUCT_FEATURES
from .redis_feature_store import RedisFeatureStore
from .feature_registry import FeatureRegistry
from .feature_computer import FeatureComputer
from .version_manager import FeatureVersionManager

logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Main Feature Store interface for production ML systems.

    Provides high-level API for:
    - Feature serving for real-time inference
    - Feature storage and retrieval
    - Feature versioning and lineage
    - Feature computation and validation
    - Performance monitoring

    Example usage:
        fs = FeatureStore()

        fs.write_features(
            entity_id="product_123",
            features={
                "name": "iPhone 15",
                "category": "smartphones",
                "text_embedding": [0.1, 0.2, ...]
            }
        )

        features = fs.get_features(
            entity_id="product_123",
            feature_names=["name", "category", "text_embedding"]
        )
    """

    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        self.config = config or DEFAULT_CONFIG

        self.redis_store = RedisFeatureStore(self.config)
        self.registry = FeatureRegistry(self.config)
        self.computer = FeatureComputer(self.config)
        self.version_manager = FeatureVersionManager(self.config)

        self.metrics = {
            "reads": 0,
            "writes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "computation_time": 0.0
        }

        logger.info("Feature Store initialized successfully")

    def write_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        feature_group: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:

        start_time = time.time()

        try:
            validation_result = self.registry.validate_features(features)
            if not validation_result["valid"]:
                logger.error(f"Feature validation failed: {validation_result['errors']}")
                return False

            features["last_updated"] = datetime.now().isoformat()

            if self.config.enable_versioning:
                version = self.version_manager.create_version(entity_id, features)
                features["_version"] = version

            success = self.redis_store.write_features(
                entity_id=entity_id,
                features=features,
                feature_group=feature_group,
                ttl=ttl or self.config.default_ttl
            )

            if success:
                self.metrics["writes"] += 1
                logger.info(f"Successfully wrote {len(features)} features for entity {entity_id}")

            return success

        except Exception as e:
            logger.error(f"Error writing features for entity {entity_id}: {e}")
            return False
        finally:
            self.metrics["computation_time"] += time.time() - start_time

    def get_features(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
        version: Optional[str] = None
    ) -> Dict[str, Any]:

        start_time = time.time()

        try:
            features = self.redis_store.get_features(
                entity_id=entity_id,
                feature_names=feature_names,
                version=version
            )

            if features:
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1

                logger.info(f"Cache miss for entity {entity_id}, computing features...")
                features = self._compute_missing_features(entity_id, feature_names)

                if features:
                    self.write_features(entity_id, features)

            self.metrics["reads"] += 1
            return features or {}

        except Exception as e:
            logger.error(f"Error retrieving features for entity {entity_id}: {e}")
            return {}
        finally:
            self.metrics["computation_time"] += time.time() - start_time

    def get_features_batch(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:

        start_time = time.time()
        results = {}

        try:
            batch_results = self.redis_store.get_features_batch(
                entity_ids=entity_ids,
                feature_names=feature_names
            )

            missing_entities = [
                entity_id for entity_id in entity_ids
                if entity_id not in batch_results or not batch_results[entity_id]
            ]

            if missing_entities:
                logger.info(f"Computing features for {len(missing_entities)} missing entities")

                for entity_id in missing_entities:
                    computed_features = self._compute_missing_features(entity_id, feature_names)
                    if computed_features:
                        batch_results[entity_id] = computed_features
                        self.write_features(entity_id, computed_features)

            results = batch_results
            self.metrics["reads"] += len(entity_ids)

        except Exception as e:
            logger.error(f"Error in batch feature retrieval: {e}")

        finally:
            self.metrics["computation_time"] += time.time() - start_time

        return results

    def compute_and_store_features(
        self,
        entity_id: str,
        raw_data: Dict[str, Any],
        force_recompute: bool = False
    ) -> bool:

        try:
            if not force_recompute:
                existing_features = self.get_features(entity_id)
                if existing_features and self._are_features_fresh(existing_features):
                    logger.info(f"Fresh features already exist for entity {entity_id}")
                    return True

            logger.info(f"Computing features for entity {entity_id}")
            computed_features = self.computer.compute_all_features(raw_data)

            if not computed_features:
                logger.error(f"Failed to compute features for entity {entity_id}")
                return False

            return self.write_features(entity_id, computed_features)

        except Exception as e:
            logger.error(f"Error computing and storing features for {entity_id}: {e}")
            return False

    def get_feature_lineage(self, entity_id: str, feature_name: str) -> Dict[str, Any]:

        return self.version_manager.get_feature_lineage(entity_id, feature_name)

    def list_feature_versions(self, entity_id: str, feature_name: str) -> List[str]:

        return self.version_manager.list_versions(entity_id, feature_name)

    def get_feature_statistics(self, feature_name: str) -> Dict[str, Any]:

        return self.registry.get_feature_statistics(feature_name)

    def health_check(self) -> Dict[str, Any]:

        return {
            "status": "healthy",
            "redis_connected": self.redis_store.health_check(),
            "metrics": self.metrics,
            "config": {
                "redis_host": self.config.redis_host,
                "redis_port": self.config.redis_port,
                "versioning_enabled": self.config.enable_versioning,
                "monitoring_enabled": self.config.enable_monitoring
            }
        }

    def get_performance_metrics(self) -> Dict[str, Any]:

        cache_hit_rate = (
            self.metrics["cache_hits"] / max(1, self.metrics["reads"])
            if self.metrics["reads"] > 0 else 0
        )

        return {
            "cache_hit_rate": cache_hit_rate,
            "total_reads": self.metrics["reads"],
            "total_writes": self.metrics["writes"],
            "average_computation_time": (
                self.metrics["computation_time"] / max(1, self.metrics["reads"])
            ),
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"]
        }

    def _compute_missing_features(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:

        try:
            logger.warning(f"Feature computation not implemented for entity {entity_id}")
            return {}
        except Exception as e:
            logger.error(f"Error computing missing features: {e}")
            return {}

    def _are_features_fresh(self, features: Dict[str, Any]) -> bool:

        try:
            if "last_updated" not in features:
                return False

            last_updated = datetime.fromisoformat(features["last_updated"])
            age = datetime.now() - last_updated

            return age < timedelta(hours=1)

        except Exception:
            return False

    def cleanup_old_versions(self, max_age_days: int = 30) -> int:

        return self.version_manager.cleanup_old_versions(max_age_days)

    def export_features(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None,
        format: str = "pandas"
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Export features for batch training or analysis.

        Args:
            entity_ids: List of entities to export
            feature_names: Specific features to export
            format: Export format ('pandas', 'dict', 'json')

        Returns:
            Features in the requested format
        """
        features_data = self.get_features_batch(entity_ids, feature_names)

        if format == "pandas":
            return pd.DataFrame.from_dict(features_data, orient='index')
        elif format == "json":
            return json.dumps(features_data, indent=2)
        else:
            return features_data
