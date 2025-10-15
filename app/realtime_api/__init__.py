import asyncio
import logging
from typing import Optional, Dict, Any

from .config import RealtimeConfig, DEFAULT_REALTIME_CONFIG
from .event_processor import EventProcessor
from .streaming_recommender import StreamingRecommender
from .kafka_producer import KafkaEventProducer
from .kafka_consumer import KafkaEventConsumer
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

__version__ = "1.0.0"

__all__ = [
    "RealtimeConfig",
    "EventProcessor",
    "StreamingRecommender",
    "KafkaEventProducer",
    "KafkaEventConsumer",
    "MetricsCollector",
    "RealtimeRecommendationSystem",
    "create_realtime_system"
]

class RealtimeRecommendationSystem:
    def __init__(
        self,
        config: Optional[RealtimeConfig] = None,
        feature_store_client: Optional[Any] = None
    ):
        self.config = config or DEFAULT_REALTIME_CONFIG
        self.feature_store_client = feature_store_client

        self.metrics_collector = MetricsCollector(self.config)
        self.event_processor = EventProcessor(self.config, feature_store_client)

        self.is_running = False
        self.background_tasks = set()

        logger.info(f"Real-time recommendation system initialized (v{__version__})")

    async def start(self):
        if self.is_running:
            logger.warning("System is already running")
            return

        logger.info("Starting real-time recommendation system...")
        self.is_running = True

        try:
            metrics_task = asyncio.create_task(
                self.metrics_collector.start_collection()
            )

            processing_task = asyncio.create_task(
                self.event_processor.start_processing()
            )

            self.background_tasks.update([metrics_task, processing_task])

            await self._wait_for_system_ready()

            logger.info("Real-time recommendation system started successfully")

            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error starting recommendation system: {e}")
            await self.stop()
            raise
        finally:
            self.is_running = False

    async def stop(self):
        logger.info("Stopping real-time recommendation system...")
        self.is_running = False

        self.event_processor.stop_processing()
        self.metrics_collector.stop_collection()

        for task in self.background_tasks:
            if not task.done():
                task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.background_tasks.clear()
        logger.info("Real-time recommendation system stopped")

    async def _wait_for_system_ready(self, timeout_seconds: int = 30):
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout_seconds:
            health = self.health_check()

            if health["status"] == "healthy":
                logger.info("System is ready and healthy")
                return

            await asyncio.sleep(1)

        logger.warning("System startup timeout - continuing anyway")

    def health_check(self) -> Dict[str, Any]:

        event_processor_health = self.event_processor.health_check()
        metrics_health = self.metrics_collector.health_check()

        components_healthy = (
            event_processor_health["status"] == "healthy" and
            metrics_health["status"] == "healthy"
        )

        overall_healthy = components_healthy and self.is_running

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "version": __version__,
            "is_running": self.is_running,
            "components": {
                "event_processor": event_processor_health,
                "metrics_collector": metrics_health
            },
            "system_ready": overall_healthy
        }

    def get_metrics(self) -> Dict[str, Any]:

        event_metrics = self.event_processor.get_metrics()
        system_metrics = self.metrics_collector.get_metrics_summary()

        return {
            "system": {
                "version": __version__,
                "uptime_seconds": getattr(self, "_start_time", 0),
                "is_running": self.is_running
            },
            "event_processing": event_metrics,
            "system_monitoring": system_metrics
        }

    async def send_test_event(self, event_type: str, event_data: Dict[str, Any]):
        try:
            producer = self.event_processor.kafka_producer
            await producer.send_event({
                "event_type": event_type,
                **event_data
            })
            logger.info(f"Test event sent: {event_type}")

        except Exception as e:
            logger.error(f"Error sending test event: {e}")
            raise

def create_realtime_system(
    config: Optional[RealtimeConfig] = None,
    feature_store_client: Optional[Any] = None
) -> RealtimeRecommendationSystem:

    return RealtimeRecommendationSystem(config, feature_store_client)
