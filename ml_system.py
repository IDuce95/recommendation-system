import asyncio
import logging
import signal
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

from app.feature_store import FeatureStore, FeatureStoreConfig
from app.recommendation.recommender import RecommendationEngine
from app.ab_testing import ABTestingSystem, ABTestConfig
from app.api.fastapi_server import create_app
from app.streamlit_app.streamlit_ui import StreamlitApp

from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class MLSystemStatus:
    feature_store_healthy: bool = False
    recommendation_api_healthy: bool = False
    ab_testing_healthy: bool = False
    kafka_healthy: bool = False
    redis_healthy: bool = False
    system_uptime: float = 0.0
    total_requests: int = 0
    active_experiments: int = 0
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

    @property
    def overall_healthy(self) -> bool:

        return all([
            self.feature_store_healthy,
            self.recommendation_api_healthy,
            self.ab_testing_healthy
        ])

class MLRecommendationSystem:
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        self.is_initialized = False
        self.start_time = datetime.now()
        self.background_tasks = []
        self.shutdown_event = asyncio.Event()

        self.feature_store: Optional[FeatureStore] = None
        self.recommendation_engine: Optional[RecommendationEngine] = None
        self.ab_testing_system: Optional[ABTestingSystem] = None
        self.fastapi_app = None
        self.streamlit_app: Optional[StreamlitApp] = None

        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "features_served": 0,
            "experiments_created": 0
        }

        logger.info("ML Recommendation System initialized")

    async def initialize(self) -> bool:

        try:
            logger.info("Starting ML Recommendation System initialization...")

            await self._initialize_feature_store()

            await self._initialize_recommendation_engine()

            await self._initialize_ab_testing()

            await self._initialize_api_servers()

            await self._start_background_tasks()

            self._setup_signal_handlers()

            self.is_initialized = True
            logger.info("ML Recommendation System successfully initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ML Recommendation System: {e}")
            await self.shutdown()
            return False

    async def _initialize_feature_store(self):
        logger.info("Initializing Feature Store...")

        feature_store_config = FeatureStoreConfig(
            redis_host=self.config.get("redis.host", "localhost"),
            redis_port=self.config.get("redis.port", 6379),
            redis_password=self.config.get("redis.password"),
            feature_ttl_seconds=self.config.get("feature_store.ttl_seconds", 3600),
            enable_versioning=self.config.get("feature_store.enable_versioning", True)
        )

        self.feature_store = FeatureStore(feature_store_config)
        await self.feature_store.initialize()

        logger.info("Feature Store initialized successfully")

    async def _initialize_recommendation_engine(self):
        logger.info("Initializing Recommendation Engine...")

        self.recommendation_engine = RecommendationEngine(
            feature_store=self.feature_store,
            config=self.config
        )
        await self.recommendation_engine.initialize()

        logger.info("Recommendation Engine initialized successfully")

    async def _initialize_ab_testing(self):
        logger.info("Initializing A/B Testing Framework...")

        ab_config = ABTestConfig(
            redis_host=self.config.get("redis.host", "localhost"),
            redis_port=self.config.get("redis.port", 6379),
            database_url=self.config.get("database.url", "sqlite:///ab_tests.db"),
            default_confidence_level=self.config.get("ab_testing.confidence_level", 0.95),
            minimum_sample_size=self.config.get("ab_testing.min_sample_size", 100)
        )

        self.ab_testing_system = ABTestingSystem(ab_config)
        await self.ab_testing_system.initialize()

        logger.info("A/B Testing Framework initialized successfully")

    async def _initialize_api_servers(self):
        logger.info("Initializing API servers...")

        self.fastapi_app = create_app(
            feature_store=self.feature_store,
            recommendation_engine=self.recommendation_engine,
            ab_testing_system=self.ab_testing_system
        )

        self.streamlit_app = StreamlitApp(
            feature_store=self.feature_store,
            recommendation_engine=self.recommendation_engine,
            ab_testing_system=self.ab_testing_system
        )

        logger.info("API servers initialized successfully")

    async def _start_background_tasks(self):
        health_task = asyncio.create_task(self._health_monitoring_task())
        self.background_tasks.append(health_task)

        metrics_task = asyncio.create_task(self._metrics_collection_task())
        self.background_tasks.append(metrics_task)

        feature_maintenance_task = asyncio.create_task(self._feature_store_maintenance_task())
        self.background_tasks.append(feature_maintenance_task)

        ab_monitoring_task = asyncio.create_task(self._ab_testing_monitoring_task())
        self.background_tasks.append(ab_monitoring_task)

        logger.info(f"Started {len(self.background_tasks)} background tasks")

    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def get_recommendations(
        self,
        user_id: str,
        product_context: Optional[Dict[str, Any]] = None,
        num_recommendations: int = 10,
        experiment_id: Optional[str] = None
    ) -> Dict[str, Any]:

        try:
            start_time = datetime.now()

            if experiment_id and self.ab_testing_system:
                variant = await self.ab_testing_system.assign_user(experiment_id, user_id)
                if variant:
                    product_context = self._apply_experiment_variant(product_context, variant)

            recommendations = await self.recommendation_engine.get_recommendations(
                user_id=user_id,
                context=product_context,
                num_recommendations=num_recommendations
            )

            response_time = (datetime.now() - start_time).total_seconds()
            self.metrics["total_requests"] += 1
            self.metrics["successful_requests"] += 1
            self.metrics["avg_response_time"] = (
                (self.metrics["avg_response_time"] * (self.metrics["total_requests"] - 1) + response_time) /
                self.metrics["total_requests"]
            )

            if experiment_id and self.ab_testing_system:
                await self.ab_testing_system.record_metric(
                    experiment_id=experiment_id,
                    user_id=user_id,
                    metric_name="recommendation_served",
                    metric_value=1.0,
                    variant=variant
                )

            return {
                "recommendations": recommendations,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": response_time * 1000,
                "experiment_info": {
                    "experiment_id": experiment_id,
                    "variant": variant if experiment_id else None
                }
            }

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            self.metrics["failed_requests"] += 1
            raise

    async def record_user_interaction(
        self,
        user_id: str,
        product_id: str,
        interaction_type: str,
        experiment_id: Optional[str] = None
    ) -> bool:

        try:
            interaction_data = {
                "user_id": user_id,
                "product_id": product_id,
                "interaction_type": interaction_type,
                "timestamp": datetime.now().isoformat()
            }

            await self.feature_store.store_feature(
                f"interaction:{user_id}:{product_id}",
                interaction_data
            )

            if experiment_id and self.ab_testing_system:
                metric_value = 1.0 if interaction_type in ["purchase", "add_to_cart"] else 0.5

                await self.ab_testing_system.record_metric(
                    experiment_id=experiment_id,
                    user_id=user_id,
                    metric_name=f"interaction_{interaction_type}",
                    metric_value=metric_value
                )

            return True

        except Exception as e:
            logger.error(f"Error recording user interaction: {e}")
            return False

    async def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        target_metric: str = "conversion_rate",
        duration_days: int = 14
    ) -> Optional[str]:

        if not self.ab_testing_system:
            logger.error("A/B Testing system not initialized")
            return None

        experiment_id = await self.ab_testing_system.create_experiment(
            name=name,
            description=description,
            variants=variants,
            target_metric=target_metric,
            duration_days=duration_days
        )

        if experiment_id:
            self.metrics["experiments_created"] += 1
            logger.info(f"Created experiment: {experiment_id}")

        return experiment_id

    async def get_system_status(self) -> MLSystemStatus:

        try:
            feature_store_health = await self._check_feature_store_health()
            recommendation_health = await self._check_recommendation_health()
            ab_testing_health = await self._check_ab_testing_health()
            kafka_health = await self._check_kafka_health()
            redis_health = await self._check_redis_health()

            ab_status = await self.ab_testing_system.get_system_status() if self.ab_testing_system else None
            active_experiments = ab_status.active_experiments if ab_status else 0

            uptime = (datetime.now() - self.start_time).total_seconds()

            return MLSystemStatus(
                feature_store_healthy=feature_store_health,
                recommendation_api_healthy=recommendation_health,
                ab_testing_healthy=ab_testing_health,
                kafka_healthy=kafka_health,
                redis_healthy=redis_health,
                system_uptime=uptime,
                total_requests=self.metrics["total_requests"],
                active_experiments=active_experiments,
                last_updated=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return MLSystemStatus()

    async def health_check(self) -> Dict[str, Any]:

        health_results = {}

        try:
            if self.feature_store:
                health_results["feature_store"] = await self.feature_store.health_check()

            if self.recommendation_engine:
                health_results["recommendation_engine"] = await self.recommendation_engine.health_check()

            if self.ab_testing_system:
                health_results["ab_testing"] = await self.ab_testing_system.health_check()

            system_status = await self.get_system_status()
            health_results["system"] = {
                "status": "healthy" if system_status.overall_healthy else "degraded",
                "uptime_seconds": system_status.system_uptime,
                "total_requests": system_status.total_requests,
                "active_experiments": system_status.active_experiments,
                "metrics": self.metrics
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_results["system"] = {
                "status": "error",
                "error": str(e)
            }

        return health_results

    async def _health_monitoring_task(self):
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                health = await self.health_check()
                system_health = health.get("system", {}).get("status", "unknown")

                if system_health != "healthy":
                    logger.warning(f"System health degraded: {health}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring task: {e}")
                await asyncio.sleep(60)

    async def _metrics_collection_task(self):
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Collect every minute

                if self.feature_store:
                    feature_metrics = await self.feature_store.get_metrics()
                    self.metrics["features_served"] = feature_metrics.get("features_served", 0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection task: {e}")
                await asyncio.sleep(60)

    async def _feature_store_maintenance_task(self):
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # Run every hour

                if self.feature_store:
                    await self.feature_store.cleanup_expired_features()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in feature store maintenance: {e}")
                await asyncio.sleep(300)

    async def _ab_testing_monitoring_task(self):
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(1800)  # Check every 30 minutes

                if self.ab_testing_system:
                    pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in A/B testing monitoring: {e}")
                await asyncio.sleep(300)

    def _apply_experiment_variant(
        self,
        context: Optional[Dict[str, Any]],
        variant: str
    ) -> Dict[str, Any]:

        if context is None:
            context = {}

        context["experiment_variant"] = variant

        if variant == "variant_a":
            context["boost_popular_items"] = True
        elif variant == "variant_b":
            context["boost_new_items"] = True

        return context

    async def _check_feature_store_health(self) -> bool:

        try:
            if self.feature_store:
                health = await self.feature_store.health_check()
                return health.get("status") == "healthy"
            return False
        except Exception:
            return False

    async def _check_recommendation_health(self) -> bool:

        try:
            if self.recommendation_engine:
                health = await self.recommendation_engine.health_check()
                return health.get("status") == "healthy"
            return False
        except Exception:
            return False

    async def _check_ab_testing_health(self) -> bool:

        try:
            if self.ab_testing_system:
                health = await self.ab_testing_system.health_check()
                return health.get("system", {}).get("status") == "healthy"
            return False
        except Exception:
            return False

    async def _check_kafka_health(self) -> bool:

        try:
            return True
        except Exception:
            return False

    async def _check_redis_health(self) -> bool:

        try:
            if self.feature_store:
                return await self.feature_store._check_redis_connection()
            return False
        except Exception:
            return False

    async def shutdown(self):
        try:
            logger.info("Starting ML Recommendation System shutdown...")

            self.shutdown_event.set()

            for task in self.background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            if self.ab_testing_system:
                await self.ab_testing_system.shutdown()

            if self.feature_store:
                await self.feature_store.cleanup()

            logger.info("ML Recommendation System shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

async def main():
    system = MLRecommendationSystem()

    try:
        if not await system.initialize():
            logger.error("Failed to initialize system")
            sys.exit(1)

        logger.info("ML Recommendation System is running...")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        sys.exit(1)
