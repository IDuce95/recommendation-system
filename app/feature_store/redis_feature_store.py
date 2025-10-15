import json
import pickle
import logging
from typing import Dict, List, Optional, Any, Union
import redis
from redis.connection import ConnectionPool
import numpy as np

from .config import FeatureStoreConfig, REDIS_KEY_PATTERNS

logger = logging.getLogger(__name__)

class RedisFeatureStore:
    def __init__(self, config: FeatureStoreConfig):
        self.config = config

        self.pool = ConnectionPool(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            ssl=config.redis_ssl,
            max_connections=config.connection_pool_size,
            socket_timeout=config.read_timeout,
            socket_connect_timeout=config.write_timeout,
            decode_responses=False  # Keep binary for pickle serialization
        )

        self.redis_client = redis.Redis(connection_pool=self.pool)

        try:
            self.redis_client.ping()
            logger.info("Redis Feature Store connected successfully")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def write_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        feature_group: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:

        try:
            pipeline = self.redis_client.pipeline()

            for feature_name, feature_value in features.items():
                key = REDIS_KEY_PATTERNS["feature"].format(
                    feature_name=feature_name,
                    entity_id=entity_id
                )

                serialized_value = self._serialize_feature_value(feature_value)

                pipeline.setex(
                    key,
                    ttl or self.config.default_ttl,
                    serialized_value
                )

            if feature_group:
                group_key = REDIS_KEY_PATTERNS["feature_group"].format(
                    group_name=feature_group,
                    entity_id=entity_id
                )

                feature_names = list(features.keys())
                pipeline.setex(
                    group_key,
                    ttl or self.config.default_ttl,
                    json.dumps(feature_names)
                )

            pipeline.execute()

            logger.debug(f"Wrote {len(features)} features for entity {entity_id}")
            return True

        except Exception as e:
            logger.error(f"Error writing features to Redis: {e}")
            return False

    def get_features(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
        version: Optional[str] = None
    ) -> Dict[str, Any]:

        try:
            if not feature_names:
                logger.warning("No feature names provided for retrieval")
                return {}

            keys = [
                REDIS_KEY_PATTERNS["feature"].format(
                    feature_name=feature_name,
                    entity_id=entity_id
                )
                for feature_name in feature_names
            ]

            pipeline = self.redis_client.pipeline()
            for key in keys:
                pipeline.get(key)

            raw_values = pipeline.execute()

            features = {}
            for feature_name, raw_value in zip(feature_names, raw_values):
                if raw_value is not None:
                    try:
                        features[feature_name] = self._deserialize_feature_value(raw_value)
                    except Exception as e:
                        logger.warning(f"Failed to deserialize feature {feature_name}: {e}")
                        continue

            logger.debug(f"Retrieved {len(features)} features for entity {entity_id}")
            return features

        except Exception as e:
            logger.error(f"Error retrieving features from Redis: {e}")
            return {}

    def get_features_batch(
        self,
        entity_ids: List[str],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:

        if not feature_names:
            logger.warning("No feature names provided for batch retrieval")
            return {}

        try:
            all_keys = []
            key_mapping = {}  # Map key back to (entity_id, feature_name)

            for entity_id in entity_ids:
                for feature_name in feature_names:
                    key = REDIS_KEY_PATTERNS["feature"].format(
                        feature_name=feature_name,
                        entity_id=entity_id
                    )
                    all_keys.append(key)
                    key_mapping[key] = (entity_id, feature_name)

            pipeline = self.redis_client.pipeline()
            for key in all_keys:
                pipeline.get(key)

            raw_values = pipeline.execute()

            results = {entity_id: {} for entity_id in entity_ids}

            for key, raw_value in zip(all_keys, raw_values):
                if raw_value is not None:
                    entity_id, feature_name = key_mapping[key]
                    try:
                        feature_value = self._deserialize_feature_value(raw_value)
                        results[entity_id][feature_name] = feature_value
                    except Exception as e:
                        logger.warning(f"Failed to deserialize {feature_name} for {entity_id}: {e}")
                        continue

            logger.debug(f"Batch retrieved features for {len(entity_ids)} entities")
            return results

        except Exception as e:
            logger.error(f"Error in batch feature retrieval: {e}")
            return {entity_id: {} for entity_id in entity_ids}

    def delete_features(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None
    ) -> bool:

        try:
            if not feature_names:
                pattern = REDIS_KEY_PATTERNS["feature"].format(
                    feature_name="*",
                    entity_id=entity_id
                )
                keys = self.redis_client.keys(pattern)
            else:
                keys = [
                    REDIS_KEY_PATTERNS["feature"].format(
                        feature_name=feature_name,
                        entity_id=entity_id
                    )
                    for feature_name in feature_names
                ]

            if keys:
                deleted_count = self.redis_client.delete(*keys)
                logger.info(f"Deleted {deleted_count} features for entity {entity_id}")

            return True

        except Exception as e:
            logger.error(f"Error deleting features: {e}")
            return False

    def exists(self, entity_id: str, feature_name: str) -> bool:

        key = REDIS_KEY_PATTERNS["feature"].format(
            feature_name=feature_name,
            entity_id=entity_id
        )
        return bool(self.redis_client.exists(key))

    def get_ttl(self, entity_id: str, feature_name: str) -> int:

        key = REDIS_KEY_PATTERNS["feature"].format(
            feature_name=feature_name,
            entity_id=entity_id
        )
        return self.redis_client.ttl(key)

    def extend_ttl(
        self,
        entity_id: str,
        feature_name: str,
        ttl: int
    ) -> bool:

        key = REDIS_KEY_PATTERNS["feature"].format(
            feature_name=feature_name,
            entity_id=entity_id
        )
        return self.redis_client.expire(key, ttl)

    def get_memory_usage(self) -> Dict[str, Any]:

        try:
            info = self.redis_client.info('memory')
            return {
                "used_memory": info.get('used_memory', 0),
                "used_memory_human": info.get('used_memory_human', '0B'),
                "used_memory_peak": info.get('used_memory_peak', 0),
                "used_memory_peak_human": info.get('used_memory_peak_human', '0B'),
                "total_system_memory": info.get('total_system_memory', 0)
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}

    def health_check(self) -> bool:

        try:
            response = self.redis_client.ping()
            return response is True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def _serialize_feature_value(self, value: Any) -> bytes:

        try:
            if isinstance(value, np.ndarray):
                return pickle.dumps(value)

            if isinstance(value, list) and len(value) > 10:  # Likely embedding
                return pickle.dumps(np.array(value))

            if isinstance(value, (str, int, float, bool)):
                return json.dumps(value).encode('utf-8')

            return pickle.dumps(value)

        except Exception as e:
            logger.warning(f"Serialization failed, using pickle: {e}")
            return pickle.dumps(value)

    def _deserialize_feature_value(self, raw_value: bytes) -> Any:

        try:
            try:
                return json.loads(raw_value.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            return pickle.loads(raw_value)

        except Exception as e:
            logger.error(f"Failed to deserialize feature value: {e}")
            return None

    def flush_all(self) -> bool:

        try:
            self.redis_client.flushdb()
            logger.warning("Flushed all feature store data")
            return True
        except Exception as e:
            logger.error(f"Error flushing Redis: {e}")
            return False

    def get_connection_info(self) -> Dict[str, Any]:

        return {
            "host": self.config.redis_host,
            "port": self.config.redis_port,
            "db": self.config.redis_db,
            "ssl": self.config.redis_ssl,
            "pool_size": self.config.connection_pool_size,
            "connected": self.health_check()
        }
