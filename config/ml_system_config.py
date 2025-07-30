import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 100
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    decode_responses: bool = True

@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    topic_recommendations: str = "recommendations"
    topic_interactions: str = "user_interactions"
    topic_features: str = "feature_updates"
    consumer_group: str = "ml_recommendation_system"
    auto_offset_reset: str = "latest"
    max_poll_records: int = 500
    session_timeout_ms: int = 30000

@dataclass
class DatabaseConfig:
    url: str = "sqlite:///ml_system.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_pre_ping: bool = True
    echo: bool = False

@dataclass
class FeatureStoreConfig:
    redis: RedisConfig = field(default_factory=RedisConfig)
    default_ttl_seconds: int = 3600
    enable_versioning: bool = True
    max_feature_size_bytes: int = 1024 * 1024  # 1MB
    compression_enabled: bool = True
    batch_size: int = 100

@dataclass
class RecommendationConfig:
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    max_response_time_ms: int = 100
    default_num_recommendations: int = 10
    max_num_recommendations: int = 50
    similarity_threshold: float = 0.1
    enable_real_time_updates: bool = True
    model_refresh_interval_minutes: int = 60

@dataclass
class ABTestingConfig:
    redis: RedisConfig = field(default_factory=RedisConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    default_confidence_level: float = 0.95
    minimum_sample_size: int = 100
    maximum_experiment_duration_days: int = 30
    bandit_algorithm: str = "thompson_sampling"
    bandit_exploration_rate: float = 0.1
    bandit_update_frequency_minutes: int = 60
    enable_early_stopping: bool = True

@dataclass
class APIConfig:
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000
    fastapi_workers: int = 4
    streamlit_host: str = "0.0.0.0"
    streamlit_port: int = 8501
    enable_cors: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])
    request_timeout_seconds: int = 30
    max_request_size_bytes: int = 1024 * 1024  # 1MB

@dataclass
class MonitoringConfig:
    health_check_interval_seconds: int = 300
    metrics_collection_interval_seconds: int = 60
    log_level: str = "INFO"
    enable_prometheus_metrics: bool = True
    prometheus_port: int = 9090
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.05,
        "response_time_p95": 200.0,
        "memory_usage": 0.85,
        "cpu_usage": 0.80
    })

@dataclass
class MLSystemConfig:
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)
    ab_testing: ABTestingConfig = field(default_factory=ABTestingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    environment: str = "production"
    debug_mode: bool = False
    enable_async_processing: bool = True
    max_concurrent_requests: int = 1000
    graceful_shutdown_timeout_seconds: int = 30

    @classmethod
    def from_environment(cls) -> "MLSystemConfig":

        config = cls()

        config.feature_store.redis.host = os.getenv("REDIS_HOST", config.feature_store.redis.host)
        config.feature_store.redis.port = int(os.getenv("REDIS_PORT", config.feature_store.redis.port))
        config.feature_store.redis.password = os.getenv("REDIS_PASSWORD", config.feature_store.redis.password)

        config.recommendation.kafka.bootstrap_servers = os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            config.recommendation.kafka.bootstrap_servers
        )

        config.ab_testing.database.url = os.getenv("DATABASE_URL", config.ab_testing.database.url)

        config.api.fastapi_host = os.getenv("FASTAPI_HOST", config.api.fastapi_host)
        config.api.fastapi_port = int(os.getenv("FASTAPI_PORT", config.api.fastapi_port))
        config.api.streamlit_port = int(os.getenv("STREAMLIT_PORT", config.api.streamlit_port))

        config.environment = os.getenv("ENVIRONMENT", config.environment)
        config.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"

        return config

    @classmethod
    def from_file(cls, config_path: str) -> "MLSystemConfig":

        import yaml

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MLSystemConfig":

        config = cls()

        if "feature_store" in config_dict:
            fs_config = config_dict["feature_store"]
            if "redis" in fs_config:
                redis_config = fs_config["redis"]
                config.feature_store.redis = RedisConfig(**redis_config)

            for key, value in fs_config.items():
                if key != "redis" and hasattr(config.feature_store, key):
                    setattr(config.feature_store, key, value)

        if "recommendation" in config_dict:
            rec_config = config_dict["recommendation"]
            if "kafka" in rec_config:
                kafka_config = rec_config["kafka"]
                config.recommendation.kafka = KafkaConfig(**kafka_config)

            for key, value in rec_config.items():
                if key != "kafka" and hasattr(config.recommendation, key):
                    setattr(config.recommendation, key, value)

        if "ab_testing" in config_dict:
            ab_config = config_dict["ab_testing"]
            if "redis" in ab_config:
                redis_config = ab_config["redis"]
                config.ab_testing.redis = RedisConfig(**redis_config)
            if "database" in ab_config:
                db_config = ab_config["database"]
                config.ab_testing.database = DatabaseConfig(**db_config)

            for key, value in ab_config.items():
                if key not in ["redis", "database"] and hasattr(config.ab_testing, key):
                    setattr(config.ab_testing, key, value)

        if "api" in config_dict:
            api_config = config_dict["api"]
            for key, value in api_config.items():
                if hasattr(config.api, key):
                    setattr(config.api, key, value)

        if "monitoring" in config_dict:
            mon_config = config_dict["monitoring"]
            for key, value in mon_config.items():
                if hasattr(config.monitoring, key):
                    setattr(config.monitoring, key, value)

        for key, value in config_dict.items():
            if key not in ["feature_store", "recommendation", "ab_testing", "api", "monitoring"]:
                if hasattr(config, key):
                    setattr(config, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:

        return {
            "feature_store": {
                "redis": {
                    "host": self.feature_store.redis.host,
                    "port": self.feature_store.redis.port,
                    "password": self.feature_store.redis.password,
                    "db": self.feature_store.redis.db,
                    "max_connections": self.feature_store.redis.max_connections,
                    "socket_timeout": self.feature_store.redis.socket_timeout,
                    "socket_connect_timeout": self.feature_store.redis.socket_connect_timeout,
                    "decode_responses": self.feature_store.redis.decode_responses
                },
                "default_ttl_seconds": self.feature_store.default_ttl_seconds,
                "enable_versioning": self.feature_store.enable_versioning,
                "max_feature_size_bytes": self.feature_store.max_feature_size_bytes,
                "compression_enabled": self.feature_store.compression_enabled,
                "batch_size": self.feature_store.batch_size
            },
            "recommendation": {
                "kafka": {
                    "bootstrap_servers": self.recommendation.kafka.bootstrap_servers,
                    "topic_recommendations": self.recommendation.kafka.topic_recommendations,
                    "topic_interactions": self.recommendation.kafka.topic_interactions,
                    "topic_features": self.recommendation.kafka.topic_features,
                    "consumer_group": self.recommendation.kafka.consumer_group,
                    "auto_offset_reset": self.recommendation.kafka.auto_offset_reset,
                    "max_poll_records": self.recommendation.kafka.max_poll_records,
                    "session_timeout_ms": self.recommendation.kafka.session_timeout_ms
                },
                "max_response_time_ms": self.recommendation.max_response_time_ms,
                "default_num_recommendations": self.recommendation.default_num_recommendations,
                "max_num_recommendations": self.recommendation.max_num_recommendations,
                "similarity_threshold": self.recommendation.similarity_threshold,
                "enable_real_time_updates": self.recommendation.enable_real_time_updates,
                "model_refresh_interval_minutes": self.recommendation.model_refresh_interval_minutes
            },
            "ab_testing": {
                "redis": {
                    "host": self.ab_testing.redis.host,
                    "port": self.ab_testing.redis.port,
                    "password": self.ab_testing.redis.password,
                    "db": self.ab_testing.redis.db
                },
                "database": {
                    "url": self.ab_testing.database.url,
                    "pool_size": self.ab_testing.database.pool_size,
                    "max_overflow": self.ab_testing.database.max_overflow
                },
                "default_confidence_level": self.ab_testing.default_confidence_level,
                "minimum_sample_size": self.ab_testing.minimum_sample_size,
                "maximum_experiment_duration_days": self.ab_testing.maximum_experiment_duration_days,
                "bandit_algorithm": self.ab_testing.bandit_algorithm,
                "bandit_exploration_rate": self.ab_testing.bandit_exploration_rate,
                "bandit_update_frequency_minutes": self.ab_testing.bandit_update_frequency_minutes,
                "enable_early_stopping": self.ab_testing.enable_early_stopping
            },
            "api": {
                "fastapi_host": self.api.fastapi_host,
                "fastapi_port": self.api.fastapi_port,
                "fastapi_workers": self.api.fastapi_workers,
                "streamlit_host": self.api.streamlit_host,
                "streamlit_port": self.api.streamlit_port,
                "enable_cors": self.api.enable_cors,
                "cors_origins": self.api.cors_origins,
                "request_timeout_seconds": self.api.request_timeout_seconds,
                "max_request_size_bytes": self.api.max_request_size_bytes
            },
            "monitoring": {
                "health_check_interval_seconds": self.monitoring.health_check_interval_seconds,
                "metrics_collection_interval_seconds": self.monitoring.metrics_collection_interval_seconds,
                "log_level": self.monitoring.log_level,
                "enable_prometheus_metrics": self.monitoring.enable_prometheus_metrics,
                "prometheus_port": self.monitoring.prometheus_port,
                "alert_thresholds": self.monitoring.alert_thresholds
            },
            "environment": self.environment,
            "debug_mode": self.debug_mode,
            "enable_async_processing": self.enable_async_processing,
            "max_concurrent_requests": self.max_concurrent_requests,
            "graceful_shutdown_timeout_seconds": self.graceful_shutdown_timeout_seconds
        }

    def save_to_file(self, config_path: str):
        import yaml

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def validate(self) -> bool:

        errors = []

        if not (1 <= self.api.fastapi_port <= 65535):
            errors.append(f"Invalid FastAPI port: {self.api.fastapi_port}")

        if not (1 <= self.api.streamlit_port <= 65535):
            errors.append(f"Invalid Streamlit port: {self.api.streamlit_port}")

        if not (1 <= self.feature_store.redis.port <= 65535):
            errors.append(f"Invalid Redis port: {self.feature_store.redis.port}")

        if not (0.0 < self.ab_testing.default_confidence_level < 1.0):
            errors.append(f"Invalid confidence level: {self.ab_testing.default_confidence_level}")

        if not (0.0 <= self.ab_testing.bandit_exploration_rate <= 1.0):
            errors.append(f"Invalid exploration rate: {self.ab_testing.bandit_exploration_rate}")

        if self.api.request_timeout_seconds <= 0:
            errors.append(f"Invalid request timeout: {self.api.request_timeout_seconds}")

        if errors:
            for error in errors:
                print(f"Configuration validation error: {error}")
            return False

        return True

DEFAULT_ML_SYSTEM_CONFIG = MLSystemConfig()

def create_system_config() -> MLSystemConfig:
    config = MLSystemConfig()
    config.environment = "production"
    config.debug_mode = False
    config.monitoring.log_level = "INFO"
    config.api.fastapi_workers = 4
    config.feature_store.redis.max_connections = 50
    config.recommendation.max_response_time_ms = 50
    return config

