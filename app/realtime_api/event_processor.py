import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime
from dataclasses import dataclass, asdict

from .config import RealtimeConfig, EventType, KafkaTopic, DEFAULT_REALTIME_CONFIG
from .kafka_consumer import KafkaEventConsumer
from .kafka_producer import KafkaEventProducer
from .streaming_recommender import StreamingRecommender, RecommendationRequest

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    success: bool
    processing_time_ms: float
    event_type: str
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EventProcessor:
    def __init__(
        self,
        config: Optional[RealtimeConfig] = None,
        feature_store_client: Optional[Any] = None
    ):
        self.config = config or DEFAULT_REALTIME_CONFIG
        self.feature_store_client = feature_store_client

        self.kafka_producer = KafkaEventProducer(self.config)
        self.kafka_consumer = KafkaEventConsumer(self.config)
        self.recommender = StreamingRecommender(self.config, self.kafka_producer)

        self.is_running = False
        self.processing_tasks = set()

        self.metrics = {
            "events_processed": 0,
            "events_failed": 0,
            "recommendations_generated": 0,
            "user_events_processed": 0,
            "total_processing_time": 0.0,
            "last_hour_events": [],
            "error_counts": {}
        }

        self._register_event_handlers()

    def _register_event_handlers(self):
        self.kafka_consumer.register_handler(
            EventType.USER_VIEW,
            self.handle_user_view_event
        )

        self.kafka_consumer.register_handler(
            EventType.USER_PURCHASE,
            self.handle_user_purchase_event
        )

        self.kafka_consumer.register_handler(
            EventType.USER_RATING,
            self.handle_user_rating_event
        )

        self.kafka_consumer.register_handler(
            EventType.RECOMMENDATION_REQUEST,
            self.handle_recommendation_request
        )

        self.kafka_consumer.register_handler(
            EventType.USER_FEEDBACK,
            self.handle_user_feedback_event
        )

        logger.info("Event handlers registered successfully")

    async def start_processing(self):
        if self.is_running:
            logger.warning("Event processor is already running")
            return

        self.is_running = True
        logger.info("Starting real-time event processing...")

        try:
            self.kafka_consumer.subscribe_to_topics([
                KafkaTopic.USER_EVENTS,
                KafkaTopic.RECOMMENDATION_REQUESTS,
                KafkaTopic.USER_INTERACTIONS
            ])

            processing_task = asyncio.create_task(self.kafka_consumer.start_consuming())
            metrics_task = asyncio.create_task(self._metrics_reporter())

            self.processing_tasks.update([processing_task, metrics_task])

            await asyncio.gather(*self.processing_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in event processing pipeline: {e}")
        finally:
            self.is_running = False
            logger.info("Event processing stopped")

    def stop_processing(self):
        logger.info("Stopping event processing...")
        self.is_running = False

        self.kafka_consumer.stop_consuming()

        for task in self.processing_tasks:
            if not task.done():
                task.cancel()

        self.processing_tasks.clear()

    async def handle_user_view_event(self, event_data: Dict[str, Any]) -> bool:

        start_time = time.time()

        try:
            user_id = event_data.get("user_id")
            product_id = event_data.get("product_id")
            timestamp = event_data.get("timestamp")
            session_id = event_data.get("session_id")

            if not user_id or not product_id:
                logger.warning("User view event missing required fields")
                return False

            logger.debug(f"Processing user view: user={user_id}, product={product_id}")

            await self._update_user_interaction(
                user_id, product_id, "view", timestamp, session_id
            )

            await self._update_product_metrics(product_id, "view")

            if await self._should_trigger_recommendations(user_id, event_data):
                await self._trigger_personalized_recommendations(user_id, event_data)

            processing_time = (time.time() - start_time) * 1000
            await self._record_processing_result(
                ProcessingResult(
                    success=True,
                    processing_time_ms=processing_time,
                    event_type="user_view",
                    metadata={"user_id": user_id, "product_id": product_id}
                )
            )

            self.metrics["user_events_processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Error handling user view event: {e}")
            await self._record_processing_error("user_view", str(e))
            return False

    async def handle_user_purchase_event(self, event_data: Dict[str, Any]) -> bool:

        start_time = time.time()

        try:
            user_id = event_data.get("user_id")
            product_id = event_data.get("product_id")
            purchase_amount = event_data.get("amount", 0.0)
            timestamp = event_data.get("timestamp")

            if not user_id or not product_id:
                logger.warning("Purchase event missing required fields")
                return False

            logger.info(f"Processing purchase: user={user_id}, product={product_id}, amount={purchase_amount}")

            await self._update_user_interaction(
                user_id, product_id, "purchase", timestamp,
                metadata={"amount": purchase_amount}
            )

            await self._update_product_metrics(product_id, "purchase", purchase_amount)

            await self._update_user_ltv(user_id, purchase_amount)

            await self._trigger_complementary_recommendations(user_id, product_id)

            processing_time = (time.time() - start_time) * 1000
            await self._record_processing_result(
                ProcessingResult(
                    success=True,
                    processing_time_ms=processing_time,
                    event_type="user_purchase",
                    metadata={"user_id": user_id, "product_id": product_id, "amount": purchase_amount}
                )
            )

            self.metrics["user_events_processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Error handling purchase event: {e}")
            await self._record_processing_error("user_purchase", str(e))
            return False

    async def handle_user_rating_event(self, event_data: Dict[str, Any]) -> bool:

        start_time = time.time()

        try:
            user_id = event_data.get("user_id")
            product_id = event_data.get("product_id")
            rating = event_data.get("rating")
            timestamp = event_data.get("timestamp")

            if not user_id or not product_id or rating is None:
                logger.warning("Rating event missing required fields")
                return False

            logger.debug(f"Processing rating: user={user_id}, product={product_id}, rating={rating}")

            await self._update_user_interaction(
                user_id, product_id, "rating", timestamp,
                metadata={"rating": rating}
            )

            await self._update_product_metrics(product_id, "rating", rating)

            await self._update_user_preferences(user_id, product_id, rating)

            processing_time = (time.time() - start_time) * 1000
            await self._record_processing_result(
                ProcessingResult(
                    success=True,
                    processing_time_ms=processing_time,
                    event_type="user_rating",
                    metadata={"user_id": user_id, "product_id": product_id, "rating": rating}
                )
            )

            self.metrics["user_events_processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Error handling rating event: {e}")
            await self._record_processing_error("user_rating", str(e))
            return False

    async def handle_recommendation_request(self, event_data: Dict[str, Any]) -> bool:

        start_time = time.time()

        try:
            request_id = event_data.get("request_id")
            user_id = event_data.get("user_id")
            num_recommendations = event_data.get("num_recommendations", 10)
            context = event_data.get("context", {})

            if not request_id or not user_id:
                logger.warning("Recommendation request missing required fields")
                return False

            logger.info(f"Processing recommendation request: {request_id} for user {user_id}")

            rec_request = RecommendationRequest(
                request_id=request_id,
                user_id=user_id,
                num_recommendations=num_recommendations,
                context=context,
                timestamp=datetime.now()
            )

            response = await self.recommender.generate_recommendations(rec_request)

            logger.info(f"Generated {len(response.recommendations)} recommendations for user {user_id} in {response.processing_time_ms:.2f}ms")

            processing_time = (time.time() - start_time) * 1000
            await self._record_processing_result(
                ProcessingResult(
                    success=True,
                    processing_time_ms=processing_time,
                    event_type="recommendation_request",
                    metadata={
                        "user_id": user_id,
                        "request_id": request_id,
                        "recommendations_count": len(response.recommendations),
                        "inference_time_ms": response.processing_time_ms
                    }
                )
            )

            self.metrics["recommendations_generated"] += 1
            return True

        except Exception as e:
            logger.error(f"Error handling recommendation request: {e}")
            await self._record_processing_error("recommendation_request", str(e))
            return False

    async def handle_user_feedback_event(self, event_data: Dict[str, Any]) -> bool:

        start_time = time.time()

        try:
            user_id = event_data.get("user_id")
            request_id = event_data.get("request_id")
            product_id = event_data.get("product_id")
            feedback_type = event_data.get("feedback_type")  # "click", "dismiss", "like", "dislike"
            timestamp = event_data.get("timestamp")

            if not user_id or not feedback_type:
                logger.warning("Feedback event missing required fields")
                return False

            logger.debug(f"Processing feedback: user={user_id}, type={feedback_type}, product={product_id}")

            await self._update_recommendation_metrics(
                user_id, request_id, product_id, feedback_type
            )

            await self._update_user_feedback_signals(user_id, product_id, feedback_type)

            processing_time = (time.time() - start_time) * 1000
            await self._record_processing_result(
                ProcessingResult(
                    success=True,
                    processing_time_ms=processing_time,
                    event_type="user_feedback",
                    metadata={
                        "user_id": user_id,
                        "feedback_type": feedback_type,
                        "product_id": product_id
                    }
                )
            )

            self.metrics["user_events_processed"] += 1
            return True

        except Exception as e:
            logger.error(f"Error handling feedback event: {e}")
            await self._record_processing_error("user_feedback", str(e))
            return False

    async def _update_user_interaction(
        self,
        user_id: str,
        product_id: str,
        interaction_type: str,
        timestamp: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        try:
            logger.debug(f"User interaction: {user_id} -> {product_id} ({interaction_type})")

            await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error updating user interaction: {e}")

    async def _update_product_metrics(
        self,
        product_id: str,
        metric_type: str,
        value: Optional[float] = None
    ):
        try:
            logger.debug(f"Product metric update: {product_id} -> {metric_type}")

            await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error updating product metrics: {e}")

    async def _update_user_ltv(self, user_id: str, purchase_amount: float):
        try:
            logger.debug(f"LTV update: {user_id} += {purchase_amount}")

            await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error updating user LTV: {e}")

    async def _update_user_preferences(self, user_id: str, product_id: str, rating: float):
        try:
            logger.debug(f"Preference update: {user_id} rated {product_id} as {rating}")

            await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")

    async def _update_recommendation_metrics(
        self,
        user_id: str,
        request_id: Optional[str],
        product_id: Optional[str],
        feedback_type: str
    ):
        try:
            logger.debug(f"Recommendation feedback: {feedback_type} for {product_id}")

            await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error updating recommendation metrics: {e}")

    async def _update_user_feedback_signals(
        self,
        user_id: str,
        product_id: Optional[str],
        feedback_type: str
    ):
        try:
            logger.debug(f"Feedback signal: {user_id} -> {feedback_type}")

            await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error updating feedback signals: {e}")

    async def _should_trigger_recommendations(
        self,
        user_id: str,
        event_data: Dict[str, Any]
    ) -> bool:

        user_hash = hash(user_id) % 5
        return user_hash == 0

    async def _trigger_personalized_recommendations(
        self,
        user_id: str,
        context: Dict[str, Any]
    ):
        try:
            request_id = f"auto_{user_id}_{int(time.time())}"

            rec_request_event = {
                "event_type": EventType.RECOMMENDATION_REQUEST.value,
                "request_id": request_id,
                "user_id": user_id,
                "num_recommendations": 5,
                "context": {
                    "triggered_by": "user_view",
                    "category": context.get("category"),
                    "auto_generated": True
                },
                "timestamp": datetime.now().isoformat()
            }

            await self.kafka_producer.send_event(rec_request_event)
            logger.debug(f"Triggered auto-recommendations for user {user_id}")

        except Exception as e:
            logger.error(f"Error triggering recommendations: {e}")

    async def _trigger_complementary_recommendations(
        self,
        user_id: str,
        purchased_product_id: str
    ):
        try:
            request_id = f"complement_{user_id}_{int(time.time())}"

            rec_request_event = {
                "event_type": EventType.RECOMMENDATION_REQUEST.value,
                "request_id": request_id,
                "user_id": user_id,
                "num_recommendations": 3,
                "context": {
                    "triggered_by": "purchase",
                    "purchased_product": purchased_product_id,
                    "recommendation_type": "complementary"
                },
                "timestamp": datetime.now().isoformat()
            }

            await self.kafka_producer.send_event(rec_request_event)
            logger.info(f"Triggered complementary recommendations for user {user_id}")

        except Exception as e:
            logger.error(f"Error triggering complementary recommendations: {e}")

    async def _record_processing_result(self, result: ProcessingResult):
        self.metrics["events_processed"] += 1
        self.metrics["total_processing_time"] += result.processing_time_ms / 1000

        current_time = time.time()
        self.metrics["last_hour_events"].append({
            "timestamp": current_time,
            "event_type": result.event_type,
            "processing_time_ms": result.processing_time_ms,
            "success": result.success
        })

        hour_ago = current_time - 3600
        self.metrics["last_hour_events"] = [
            event for event in self.metrics["last_hour_events"]
            if event["timestamp"] > hour_ago
        ]

    async def _record_processing_error(self, event_type: str, error_message: str):
        self.metrics["events_failed"] += 1

        if event_type not in self.metrics["error_counts"]:
            self.metrics["error_counts"][event_type] = 0
        self.metrics["error_counts"][event_type] += 1

        logger.error(f"Processing error for {event_type}: {error_message}")

    async def _metrics_reporter(self):
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Report every minute

                if self.metrics["events_processed"] > 0:
                    avg_processing_time = (
                        self.metrics["total_processing_time"] /
                        self.metrics["events_processed"]
                    ) * 1000

                    success_rate = (
                        self.metrics["events_processed"] /
                        (self.metrics["events_processed"] + self.metrics["events_failed"])
                    )

                    logger.info(
                        f"Event processing metrics: "
                        f"events={self.metrics['events_processed']}, "
                        f"success_rate={success_rate:.3f}, "
                        f"avg_time={avg_processing_time:.2f}ms, "
                        f"recommendations={self.metrics['recommendations_generated']}"
                    )

            except Exception as e:
                logger.error(f"Error in metrics reporting: {e}")

    def get_metrics(self) -> Dict[str, Any]:

        total_events = self.metrics["events_processed"] + self.metrics["events_failed"]

        recent_events = self.metrics["last_hour_events"]
        recent_success_rate = 0.0
        recent_avg_time = 0.0

        if recent_events:
            successful_recent = sum(1 for e in recent_events if e["success"])
            recent_success_rate = successful_recent / len(recent_events)
            recent_avg_time = sum(e["processing_time_ms"] for e in recent_events) / len(recent_events)

        return {
            "total_events_processed": self.metrics["events_processed"],
            "total_events_failed": self.metrics["events_failed"],
            "total_recommendations_generated": self.metrics["recommendations_generated"],
            "user_events_processed": self.metrics["user_events_processed"],
            "overall_success_rate": self.metrics["events_processed"] / max(1, total_events),
            "recent_success_rate": recent_success_rate,
            "recent_avg_processing_time_ms": recent_avg_time,
            "events_per_hour": len(recent_events),
            "error_counts": self.metrics["error_counts"].copy(),
            "is_running": self.is_running,
            "component_health": {
                "kafka_consumer": self.kafka_consumer.health_check(),
                "kafka_producer": self.kafka_producer.health_check(),
                "recommender": self.recommender.health_check()
            }
        }

    def health_check(self) -> Dict[str, Any]:

        metrics = self.get_metrics()

        component_health = metrics["component_health"]
        all_healthy = all(
            comp["status"] == "healthy"
            for comp in component_health.values()
        )

        performance_healthy = (
            metrics["recent_success_rate"] > 0.95 and  # 95% success rate
            metrics["recent_avg_processing_time_ms"] < self.config.max_response_time_ms
        )

        overall_healthy = all_healthy and performance_healthy and self.is_running

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "is_running": self.is_running,
            "components_healthy": all_healthy,
            "performance_healthy": performance_healthy,
            "metrics": metrics
        }
