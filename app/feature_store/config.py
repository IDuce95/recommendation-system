import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT_EMBEDDING = "text_embedding"
    IMAGE_EMBEDDING = "image_embedding"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"

class FeatureGroup(Enum):
    PRODUCT_BASIC = "product_basic"
    PRODUCT_EMBEDDINGS = "product_embeddings"
    USER_PROFILE = "user_profile"
    INTERACTION_FEATURES = "interaction_features"
    COMPUTED_FEATURES = "computed_features"

@dataclass
class FeatureStoreConfig:
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    redis_ssl: bool = os.getenv("REDIS_SSL", "false").lower() == "true"

    default_ttl: int = 3600  # 1 hour in seconds
    embedding_ttl: int = 86400  # 24 hours for embeddings
    batch_size: int = 1000
    max_retries: int = 3

    enable_versioning: bool = True
    max_versions: int = 10

    enable_monitoring: bool = True
    metrics_prefix: str = "feature_store"

    connection_pool_size: int = 10
    read_timeout: int = 5
    write_timeout: int = 10

PRODUCT_FEATURES = {
    "product_id": {
        "type": FeatureType.NUMERIC,
        "group": FeatureGroup.PRODUCT_BASIC,
        "description": "Unique product identifier",
        "is_key": True,
        "nullable": False
    },
    "name": {
        "type": FeatureType.CATEGORICAL,
        "group": FeatureGroup.PRODUCT_BASIC,
        "description": "Product name",
        "nullable": False
    },
    "category": {
        "type": FeatureType.CATEGORICAL,
        "group": FeatureGroup.PRODUCT_BASIC,
        "description": "Product category",
        "nullable": False
    },
    "description": {
        "type": FeatureType.CATEGORICAL,
        "group": FeatureGroup.PRODUCT_BASIC,
        "description": "Product description text",
        "nullable": True
    },
    "brand": {
        "type": FeatureType.CATEGORICAL,
        "group": FeatureGroup.PRODUCT_BASIC,
        "description": "Product brand",
        "nullable": True
    },
    "price": {
        "type": FeatureType.NUMERIC,
        "group": FeatureGroup.PRODUCT_BASIC,
        "description": "Product price",
        "nullable": True
    },
    "text_embedding": {
        "type": FeatureType.TEXT_EMBEDDING,
        "group": FeatureGroup.PRODUCT_EMBEDDINGS,
        "description": "Text-based product embedding vector",
        "nullable": True,
        "dimension": 384  # For all-MiniLM-L6-v2
    },
    "image_embedding": {
        "type": FeatureType.IMAGE_EMBEDDING,
        "group": FeatureGroup.PRODUCT_EMBEDDINGS,
        "description": "Image-based product embedding vector",
        "nullable": True,
        "dimension": 512  # Typical CNN output
    },
    "popularity_score": {
        "type": FeatureType.NUMERIC,
        "group": FeatureGroup.COMPUTED_FEATURES,
        "description": "Computed popularity score based on interactions",
        "nullable": True,
        "computation": "interaction_count / total_products"
    },
    "category_similarity_score": {
        "type": FeatureType.NUMERIC,
        "group": FeatureGroup.COMPUTED_FEATURES,
        "description": "Average similarity within category",
        "nullable": True
    },
    "last_updated": {
        "type": FeatureType.TIMESTAMP,
        "group": FeatureGroup.PRODUCT_BASIC,
        "description": "Feature last update timestamp",
        "nullable": False
    }
}

COMPUTED_FEATURES = {
    "popularity_score": {
        "depends_on": ["interaction_count", "view_count", "rating_avg"],
        "computation_function": "compute_popularity_score",
        "update_frequency": "daily",
        "ttl": 86400  # 24 hours
    },
    "category_similarity_avg": {
        "depends_on": ["text_embedding", "category"],
        "computation_function": "compute_category_similarity",
        "update_frequency": "weekly",
        "ttl": 604800  # 7 days
    },
    "embedding_freshness": {
        "depends_on": ["text_embedding", "image_embedding", "last_updated"],
        "computation_function": "compute_embedding_freshness",
        "update_frequency": "hourly",
        "ttl": 3600  # 1 hour
    }
}

REDIS_KEY_PATTERNS = {
    "feature": "fs:feature:{feature_name}:{entity_id}",
    "feature_group": "fs:group:{group_name}:{entity_id}",
    "metadata": "fs:meta:{feature_name}",
    "version": "fs:version:{feature_name}:{version}:{entity_id}",
    "lineage": "fs:lineage:{feature_name}:{entity_id}",
    "lock": "fs:lock:{feature_name}:{entity_id}",
    "monitoring": "fs:metrics:{metric_name}"
}

MONITORING_CONFIG = {
    "metrics": [
        "feature_read_latency",
        "feature_write_latency",
        "feature_cache_hit_rate",
        "feature_computation_time",
        "feature_validation_errors",
        "feature_staleness"
    ],
    "alerts": {
        "high_latency_threshold": 100,  # ms
        "low_cache_hit_rate": 0.8,
        "stale_features_threshold": 0.1
    }
}

DEFAULT_CONFIG = FeatureStoreConfig()
