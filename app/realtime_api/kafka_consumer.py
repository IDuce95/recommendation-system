import json
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("Kafka library not available. Install with: pip install kafka-python")

from .config import RealtimeConfig, KafkaTopic, EventType, DEFAULT_REALTIME_CONFIG

logger = logging.getLogger(__name__)

class KafkaEventConsumer:
    def __init__(self, config: Optional[RealtimeConfig] = None):
        self.config = config or DEFAULT_REALTIME_CONFIG
        self.consumer = None
        self.is_running = False
        self.is_connected = False

        self.event_handlers: Dict[EventType, List[Callable]] = {}

        self.metrics = {
            "messages_processed": 0,
            "messages_failed": 0,
            "total_processing_time": 0.0,
            "last_processing_time": 0.0,
            "consumer_lag": 0,
            "rebalances": 0
        }

        self.max_processing_time = self.config.max_response_time_ms / 1000.0
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        self.shutdown_event = threading.Event()

        if KAFKA_AVAILABLE:
            self._initialize_consumer()
        else:
            logger.warning("Kafka not available - running in mock mode")

    def _initialize_consumer(self):
        try:
            consumer_config = {
                'bootstrap_servers': self.config.kafka_bootstrap_servers,
                'group_id': self.config.kafka_group_id,
                'client_id': f"{self.config.kafka_client_id}-consumer",
                'auto_offset_reset': self.config.kafka_auto_offset_reset,
                'enable_auto_commit': self.config.kafka_enable_auto_commit,
                'auto_commit_interval_ms': self.config.kafka_auto_commit_interval_ms,
                'session_timeout_ms': self.config.kafka_session_timeout_ms,
                'heartbeat_interval_ms': self.config.kafka_heartbeat_interval_ms,
                'max_poll_records': self.config.kafka_max_poll_records,
                'fetch_min_bytes': self.config.kafka_fetch_min_bytes,
                'fetch_max_wait_ms': self.config.kafka_fetch_max_wait_ms,
                'value_deserializer': lambda m: json.loads(m.decode('utf-8')) if m else None,
                'key_deserializer': lambda k: k.decode('utf-8') if k else None
            }

            self.consumer = KafkaConsumer(**consumer_config)
            self.is_connected = True
            logger.info("Kafka consumer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            self.is_connected = False

    def register_handler(
        self,
        event_type: EventType,
        handler: Callable[[Dict[str, Any]], Awaitable[bool]]
    ):
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []

        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")

    def subscribe_to_topics(self, topics: List[KafkaTopic]):
        if not KAFKA_AVAILABLE or not self.is_connected:
            logger.warning("Cannot subscribe - Kafka not available")
            return

        try:
            topic_names = [topic.value for topic in topics]
            self.consumer.subscribe(topic_names)
            logger.info(f"Subscribed to topics: {topic_names}")

        except Exception as e:
            logger.error(f"Failed to subscribe to topics: {e}")

    async def start_consuming(self):
        if self.is_running:
            logger.warning("Consumer is already running")
            return

        self.is_running = True
        logger.info("Starting Kafka event consumption...")

        try:
            if KAFKA_AVAILABLE and self.is_connected:
                await self._consume_real_events()
            else:
                await self._consume_mock_events()

        except Exception as e:
            logger.error(f"Error in event consumption: {e}")
        finally:
            self.is_running = False
            logger.info("Event consumption stopped")

    async def _consume_real_events(self):
        while self.is_running and not self.shutdown_event.is_set():
            try:
                message_batch = self.consumer.poll(
                    timeout_ms=1000,
                    max_records=self.config.kafka_max_poll_records
                )

                if not message_batch:
                    continue

                processing_tasks = []

                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        task = self._process_message_async(message)
                        processing_tasks.append(task)

                if processing_tasks:
                    await asyncio.gather(*processing_tasks, return_exceptions=True)

                if self.config.kafka_enable_auto_commit:
                    self.consumer.commit()

            except Exception as e:
                logger.error(f"Error in message consumption loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

    async def _consume_mock_events(self):
        logger.info("Running in mock consumption mode")

        while self.is_running and not self.shutdown_event.is_set():
            mock_event = {
                "event_type": EventType.USER_VIEW.value,
                "user_id": f"user_{int(time.time()) % 1000}",
                "product_id": f"product_{int(time.time()) % 100}",
                "timestamp": datetime.now().isoformat()
            }

            await self._process_event_data(mock_event)
            await asyncio.sleep(2)  # Process every 2 seconds

    async def _process_message_async(self, message):
        start_time = time.time()

        try:
            event_data = message.value
            if not event_data:
                return

            await self._process_event_data(event_data)

            processing_time = time.time() - start_time
            self._update_success_metrics(processing_time)

        except Exception as e:
            self._handle_processing_error(e, message)

    async def _process_event_data(self, event_data: Dict[str, Any]):
        try:
            event_type_str = event_data.get("event_type")
            if not event_type_str:
                logger.warning("Event missing event_type field")
                return

            try:
                event_type = EventType(event_type_str)
            except ValueError:
                logger.warning(f"Unknown event type: {event_type_str}")
                return

            handlers = self.event_handlers.get(event_type, [])

            if not handlers:
                logger.debug(f"No handlers registered for {event_type.value}")
                return

            handler_tasks = []
            for handler in handlers:
                task = handler(event_data)
                handler_tasks.append(task)

            results = await asyncio.gather(*handler_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Handler {i} failed for {event_type.value}: {result}")

        except Exception as e:
            logger.error(f"Error processing event data: {e}")

    def _update_success_metrics(self, processing_time: float):
        self.metrics["messages_processed"] += 1
        self.metrics["total_processing_time"] += processing_time
        self.metrics["last_processing_time"] = processing_time

        if processing_time > self.max_processing_time:
            logger.warning(f"Slow processing detected: {processing_time:.3f}s > {self.max_processing_time:.3f}s")

    def _handle_processing_error(self, error: Exception, message=None):
        self.metrics["messages_failed"] += 1
        logger.error(f"Failed to process message: {error}")

        if message and hasattr(message, 'value'):
            logger.debug(f"Failed message content: {message.value}")

    def stop_consuming(self):
        logger.info("Stopping event consumption...")
        self.is_running = False
        self.shutdown_event.set()

        if self.consumer:
            try:
                self.consumer.close()
            except Exception as e:
                logger.error(f"Error closing consumer: {e}")

        self.executor.shutdown(wait=True)

    def get_metrics(self) -> Dict[str, Any]:

        total_messages = self.metrics["messages_processed"] + self.metrics["messages_failed"]
        avg_processing_time = (
            self.metrics["total_processing_time"] / max(1, self.metrics["messages_processed"])
        )

        return {
            "messages_processed": self.metrics["messages_processed"],
            "messages_failed": self.metrics["messages_failed"],
            "success_rate": self.metrics["messages_processed"] / max(1, total_messages),
            "average_processing_time_ms": avg_processing_time * 1000,
            "last_processing_time_ms": self.metrics["last_processing_time"] * 1000,
            "consumer_lag": self.metrics["consumer_lag"],
            "rebalances": self.metrics["rebalances"],
            "is_running": self.is_running,
            "is_connected": self.is_connected
        }

    def health_check(self) -> Dict[str, Any]:

        metrics = self.get_metrics()

        is_healthy = (
            self.is_connected and
            not self.shutdown_event.is_set() and
            metrics["average_processing_time_ms"] < self.config.max_response_time_ms
        )

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "kafka_connected": self.is_connected,
            "is_running": self.is_running,
            "registered_handlers": len(self.event_handlers),
            "metrics": metrics
        }

async def handle_user_view_event(event_data: Dict[str, Any]) -> bool:

    try:
        user_id = event_data.get("user_id")
        product_id = event_data.get("product_id")

        logger.info(f"Processing user view: user={user_id}, product={product_id}")

        return True

    except Exception as e:
        logger.error(f"Error handling user view event: {e}")
        return False

async def handle_recommendation_request(event_data: Dict[str, Any]) -> bool:

    try:
        user_id = event_data.get("user_id")
        request_id = event_data.get("request_id")

        logger.info(f"Processing recommendation request: user={user_id}, request={request_id}")

        return True

    except Exception as e:
        logger.error(f"Error handling recommendation request: {e}")
        return False

def create_consumer_with_handlers(config: Optional[RealtimeConfig] = None) -> KafkaEventConsumer:

    consumer = KafkaEventConsumer(config)

    consumer.register_handler(EventType.USER_VIEW, handle_user_view_event)
    consumer.register_handler(EventType.RECOMMENDATION_REQUEST, handle_recommendation_request)

    consumer.subscribe_to_topics([
        KafkaTopic.USER_EVENTS,
        KafkaTopic.RECOMMENDATION_REQUESTS
    ])

    return consumer
