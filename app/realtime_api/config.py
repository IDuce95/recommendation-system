import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum

class EventType(Enum):
    USER_VIEW = "user_view"
    USER_CLICK = "user_click"
    USER_PURCHASE = "user_purchase"
    PRODUCT_UPDATE = "product_update"
    RECOMMENDATION_REQUEST = "recommendation_request"
    RECOMMENDATION_RESPONSE = "recommendation_response"
    FEATURE_UPDATE = "feature_update"

class KafkaTopic(Enum):
    USER_EVENTS = "user-events"
    PRODUCT_EVENTS = "product-events"
    RECOMMENDATION_REQUESTS = "recommendation-requests"
    RECOMMENDATION_RESPONSES = "recommendation-responses"
    FEATURE_UPDATES = "feature-updates"
    METRICS = "metrics"

@dataclass
class RealtimeConfig:
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_client_id: str = os.getenv("KAFKA_CLIENT_ID", "recommendation-system")
    kafka_group_id: str = os.getenv("KAFKA_GROUP_ID", "recommendation-consumers")
    kafka_auto_offset_reset: str = "latest"
    kafka_enable_auto_commit: bool = True
    kafka_auto_commit_interval_ms: int = 1000
    kafka_session_timeout_ms: int = 30000
    kafka_heartbeat_interval_ms: int = 3000
    kafka_max_poll_records: int = 500
    kafka_fetch_min_bytes: int = 1024
    kafka_fetch_max_wait_ms: int = 500

    kafka_producer_acks: str = "1"  # 0=no wait, 1=leader, all=all replicas
    kafka_producer_retries: int = 3
    kafka_producer_batch_size: int = 16384
    kafka_producer_linger_ms: int = 10
    kafka_producer_buffer_memory: int = 33554432
    kafka_producer_compression_type: str = "gzip"

    max_response_time_ms: int = 100  # Sub-100ms target
    batch_size: int = 1000
    parallel_workers: int = 4
    cache_ttl_seconds: int = 300  # 5 minutes
    max_concurrent_requests: int = 1000

    feature_store_enabled: bool = True
    feature_cache_size: int = 10000
    feature_refresh_interval_seconds: int = 60

    default_recommendation_count: int = 10
    max_recommendation_count: int = 50
    similarity_threshold: float = 0.3
    enable_real_time_learning: bool = True

    enable_metrics: bool = True
    metrics_interval_seconds: int = 30
    enable_detailed_logging: bool = False
    log_sample_rate: float = 0.1  # Log 10% of requests

    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    circuit_breaker_half_open_max_calls: int = 3

KAFKA_TOPICS = {
    KafkaTopic.USER_EVENTS: {
        "partitions": 3,
        "replication_factor": 1,
        "cleanup_policy": "delete",
        "retention_ms": 86400000,  # 24 hours
        "segment_ms": 3600000      # 1 hour segments
    },
    KafkaTopic.PRODUCT_EVENTS: {
        "partitions": 2,
        "replication_factor": 1,
        "cleanup_policy": "compact",
        "retention_ms": 604800000,  # 7 days
        "segment_ms": 3600000
    },
    KafkaTopic.RECOMMENDATION_REQUESTS: {
        "partitions": 4,
        "replication_factor": 1,
        "cleanup_policy": "delete",
        "retention_ms": 3600000,   # 1 hour
        "segment_ms": 600000       # 10 minute segments
    },
    KafkaTopic.RECOMMENDATION_RESPONSES: {
        "partitions": 4,
        "replication_factor": 1,
        "cleanup_policy": "delete",
        "retention_ms": 3600000,
        "segment_ms": 600000
    },
    KafkaTopic.FEATURE_UPDATES: {
        "partitions": 2,
        "replication_factor": 1,
        "cleanup_policy": "compact",
        "retention_ms": 86400000,
        "segment_ms": 3600000
    },
    KafkaTopic.METRICS: {
        "partitions": 1,
        "replication_factor": 1,
        "cleanup_policy": "delete",
        "retention_ms": 86400000,
        "segment_ms": 3600000
    }
}

EVENT_SCHEMAS = {
    EventType.USER_VIEW: {
        "user_id": {"type": "string", "required": True},
        "product_id": {"type": "string", "required": True},
        "timestamp": {"type": "timestamp", "required": True},
        "session_id": {"type": "string", "required": False},
        "view_duration_ms": {"type": "integer", "required": False},
        "device_type": {"type": "string", "required": False},
        "referrer": {"type": "string", "required": False}
    },

    EventType.USER_CLICK: {
        "user_id": {"type": "string", "required": True},
        "product_id": {"type": "string", "required": True},
        "timestamp": {"type": "timestamp", "required": True},
        "session_id": {"type": "string", "required": False},
        "click_position": {"type": "integer", "required": False},
        "recommendation_id": {"type": "string", "required": False}
    },

    EventType.USER_PURCHASE: {
        "user_id": {"type": "string", "required": True},
        "product_id": {"type": "string", "required": True},
        "timestamp": {"type": "timestamp", "required": True},
        "session_id": {"type": "string", "required": False},
        "quantity": {"type": "integer", "required": True},
        "price": {"type": "float", "required": True},
        "currency": {"type": "string", "required": False}
    },

    EventType.RECOMMENDATION_REQUEST: {
        "user_id": {"type": "string", "required": True},
        "request_id": {"type": "string", "required": True},
        "timestamp": {"type": "timestamp", "required": True},
        "context": {"type": "object", "required": False},
        "max_recommendations": {"type": "integer", "required": False},
        "filters": {"type": "object", "required": False}
    },

    EventType.RECOMMENDATION_RESPONSE: {
        "user_id": {"type": "string", "required": True},
        "request_id": {"type": "string", "required": True},
        "timestamp": {"type": "timestamp", "required": True},
        "recommendations": {"type": "array", "required": True},
        "response_time_ms": {"type": "integer", "required": True},
        "algorithm_version": {"type": "string", "required": False}
    }
}

PERFORMANCE_THRESHOLDS = {
    "max_response_time_ms": 100,
    "p95_response_time_ms": 50,
    "p99_response_time_ms": 80,
    "min_throughput_rps": 1000,
    "max_error_rate": 0.01,  # 1%
    "max_memory_usage_mb": 1024,
    "max_cpu_usage_percent": 80
}

CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 5,
    "timeout_seconds": 60,
    "half_open_max_calls": 3,
    "monitoring_window_seconds": 300
}

CACHE_CONFIG = {
    "recommendation_cache_ttl": 300,    # 5 minutes
    "feature_cache_ttl": 600,           # 10 minutes
    "user_profile_cache_ttl": 1800,     # 30 minutes
    "product_cache_ttl": 3600,          # 1 hour
    "max_cache_size": 10000
}

MONITORING_CONFIG = {
    "metrics_collection_interval": 30,
    "health_check_interval": 10,
    "log_sampling_rate": 0.1,
    "enable_distributed_tracing": True,
    "trace_sampling_rate": 0.01
}

DEFAULT_REALTIME_CONFIG = RealtimeConfig()
