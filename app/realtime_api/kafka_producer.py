import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError, KafkaTimeoutError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("Kafka library not available. Install with: pip install kafka-python")

from .config import RealtimeConfig, KafkaTopic, EventType, DEFAULT_REALTIME_CONFIG

logger = logging.getLogger(__name__)

class KafkaEventProducer:
    def __init__(self, config: Optional[RealtimeConfig] = None):
        self.config = config or DEFAULT_REALTIME_CONFIG
        self.producer = None
        self.is_connected = False
        self.circuit_breaker_open = False
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None

        self.metrics = {
            "messages_sent": 0,
            "messages_failed": 0,
            "total_send_time": 0.0,
            "batch_sends": 0,
            "circuit_breaker_trips": 0
        }

        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)

        if KAFKA_AVAILABLE:
            self._initialize_producer()
        else:
            logger.warning("Kafka not available - running in mock mode")

    def _initialize_producer(self):
        try:
            producer_config = {
                'bootstrap_servers': self.config.kafka_bootstrap_servers,
                'client_id': self.config.kafka_client_id,
                'acks': self.config.kafka_producer_acks,
                'retries': self.config.kafka_producer_retries,
                'batch_size': self.config.kafka_producer_batch_size,
                'linger_ms': self.config.kafka_producer_linger_ms,
                'buffer_memory': self.config.kafka_producer_buffer_memory,
                'compression_type': self.config.kafka_producer_compression_type,
                'max_request_size': 1048576,  # 1MB
                'request_timeout_ms': 30000,
                'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
                'key_serializer': lambda k: k.encode('utf-8') if k else None
            }

            self.producer = KafkaProducer(**producer_config)
            self.is_connected = True
            logger.info("Kafka producer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self.is_connected = False

    async def send_event(
        self,
        event_type: EventType,
        event_data: Dict[str, Any],
        topic: Optional[KafkaTopic] = None,
        key: Optional[str] = None
    ) -> bool:

        if not self._check_circuit_breaker():
            return False

        try:
            enriched_event = self._enrich_event(event_type, event_data)
            target_topic = topic or self._determine_topic(event_type)
            message_key = key or self._generate_key(enriched_event)

            start_time = time.time()

            if KAFKA_AVAILABLE and self.is_connected:
                loop = asyncio.get_event_loop()
                future = self.producer.send(
                    topic=target_topic.value,
                    value=enriched_event,
                    key=message_key
                )

                await loop.run_in_executor(
                    self.executor,
                    lambda: future.get(timeout=1.0)
                )
            else:
                await asyncio.sleep(0.001)  # Simulate network delay
                logger.debug(f"Mock sent event: {event_type.value}")

            send_time = time.time() - start_time
            self._update_success_metrics(send_time)

            if self.circuit_breaker_failures > 0:
                self.circuit_breaker_failures = 0
                logger.info("Circuit breaker reset after successful send")

            return True

        except Exception as e:
            self._handle_send_error(e)
            return False

    async def send_batch(
        self,
        events: List[Dict[str, Any]],
        topic: Optional[KafkaTopic] = None
    ) -> int:

        if not events or not self._check_circuit_breaker():
            return 0

        start_time = time.time()
        successful_sends = 0

        try:
            tasks = []
            for event in events:
                event_type = EventType(event.get('event_type', EventType.USER_VIEW.value))
                task = self.send_event(event_type, event, topic)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful_sends = sum(1 for result in results if result is True)

            batch_time = time.time() - start_time
            self.metrics["batch_sends"] += 1
            self.metrics["total_send_time"] += batch_time

            logger.info(f"Batch send completed: {successful_sends}/{len(events)} events in {batch_time:.3f}s")

        except Exception as e:
            logger.error(f"Batch send failed: {e}")

        return successful_sends

    def send_user_view_event(
        self,
        user_id: str,
        product_id: str,
        session_id: Optional[str] = None,
        view_duration_ms: Optional[int] = None,
        device_type: Optional[str] = None
    ) -> bool:

        event_data = {
            "user_id": user_id,
            "product_id": product_id,
            "session_id": session_id,
            "view_duration_ms": view_duration_ms,
            "device_type": device_type
        }

        event_data = {k: v for k, v in event_data.items() if v is not None}

        return asyncio.create_task(
            self.send_event(EventType.USER_VIEW, event_data)
        )

    def send_recommendation_request(
        self,
        user_id: str,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_recommendations: Optional[int] = None
    ) -> bool:

        event_data = {
            "user_id": user_id,
            "request_id": request_id or str(uuid.uuid4()),
            "context": context or {},
            "max_recommendations": max_recommendations or self.config.default_recommendation_count
        }

        return asyncio.create_task(
            self.send_event(EventType.RECOMMENDATION_REQUEST, event_data)
        )

    def _enrich_event(self, event_type: EventType, event_data: Dict[str, Any]) -> Dict[str, Any]:

        enriched = event_data.copy()

        enriched.update({
            "event_type": event_type.value,
            "timestamp": datetime.now().isoformat(),
            "event_id": str(uuid.uuid4()),
            "producer_id": self.config.kafka_client_id,
            "schema_version": "1.0"
        })

        return enriched

    def _determine_topic(self, event_type: EventType) -> KafkaTopic:

        topic_mapping = {
            EventType.USER_VIEW: KafkaTopic.USER_EVENTS,
            EventType.USER_CLICK: KafkaTopic.USER_EVENTS,
            EventType.USER_PURCHASE: KafkaTopic.USER_EVENTS,
            EventType.PRODUCT_UPDATE: KafkaTopic.PRODUCT_EVENTS,
            EventType.RECOMMENDATION_REQUEST: KafkaTopic.RECOMMENDATION_REQUESTS,
            EventType.RECOMMENDATION_RESPONSE: KafkaTopic.RECOMMENDATION_RESPONSES,
            EventType.FEATURE_UPDATE: KafkaTopic.FEATURE_UPDATES
        }

        return topic_mapping.get(event_type, KafkaTopic.USER_EVENTS)

    def _generate_key(self, event_data: Dict[str, Any]) -> str:

        if "user_id" in event_data:
            return f"user_{event_data['user_id']}"
        elif "product_id" in event_data:
            return f"product_{event_data['product_id']}"
        else:
            return f"event_{event_data.get('event_id', 'unknown')}"

    def _check_circuit_breaker(self) -> bool:

        if not self.config.circuit_breaker_enabled:
            return True

        if not self.circuit_breaker_open:
            return True

        if self.circuit_breaker_last_failure:
            time_since_failure = time.time() - self.circuit_breaker_last_failure
            if time_since_failure > self.config.circuit_breaker_timeout_seconds:
                self.circuit_breaker_open = False
                logger.info("Circuit breaker half-open - allowing test requests")
                return True

        return False

    def _handle_send_error(self, error: Exception):
        self.metrics["messages_failed"] += 1

        if self.config.circuit_breaker_enabled:
            self.circuit_breaker_failures += 1

            if self.circuit_breaker_failures >= self.config.circuit_breaker_failure_threshold:
                self.circuit_breaker_open = True
                self.circuit_breaker_last_failure = time.time()
                self.metrics["circuit_breaker_trips"] += 1
                logger.warning(f"Circuit breaker opened after {self.circuit_breaker_failures} failures")

        logger.error(f"Failed to send event: {error}")

    def _update_success_metrics(self, send_time: float):
        self.metrics["messages_sent"] += 1
        self.metrics["total_send_time"] += send_time

    def get_metrics(self) -> Dict[str, Any]:

        total_messages = self.metrics["messages_sent"] + self.metrics["messages_failed"]
        avg_send_time = (
            self.metrics["total_send_time"] / max(1, self.metrics["messages_sent"])
        )

        return {
            "messages_sent": self.metrics["messages_sent"],
            "messages_failed": self.metrics["messages_failed"],
            "success_rate": self.metrics["messages_sent"] / max(1, total_messages),
            "average_send_time_ms": avg_send_time * 1000,
            "batch_sends": self.metrics["batch_sends"],
            "circuit_breaker_open": self.circuit_breaker_open,
            "circuit_breaker_trips": self.metrics["circuit_breaker_trips"],
            "is_connected": self.is_connected
        }

    def health_check(self) -> Dict[str, Any]:

        return {
            "status": "healthy" if self.is_connected and not self.circuit_breaker_open else "unhealthy",
            "kafka_connected": self.is_connected,
            "circuit_breaker_open": self.circuit_breaker_open,
            "recent_failures": self.circuit_breaker_failures,
            "metrics": self.get_metrics()
        }

    def close(self):
        try:
            if self.producer:
                self.producer.flush(timeout=5)
                self.producer.close(timeout=5)

            self.executor.shutdown(wait=True)
            logger.info("Kafka producer closed successfully")

        except Exception as e:
            logger.error(f"Error closing producer: {e}")

async def send_user_view(user_id: str, product_id: str, **kwargs) -> bool:

    producer = KafkaEventProducer()
    return await producer.send_event(
        EventType.USER_VIEW,
        {"user_id": user_id, "product_id": product_id, **kwargs}
    )

async def send_recommendation_request(user_id: str, **kwargs) -> bool:

    producer = KafkaEventProducer()
    return await producer.send_event(
        EventType.RECOMMENDATION_REQUEST,
        {"user_id": user_id, **kwargs}
    )
