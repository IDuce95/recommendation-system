import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using in-memory storage")

from .config import (
    ABTestConfig, MetricType, MetricConfig,
    DEFAULT_AB_TEST_CONFIG
)

logger = logging.getLogger(__name__)

@dataclass
class MetricEvent:
    experiment_id: str
    variant: str
    user_id: str
    metric_type: MetricType
    metric_value: float
    timestamp: datetime
    session_id: Optional[str] = None
    user_attributes: Optional[Dict[str, Any]] = None
    event_metadata: Optional[Dict[str, Any]] = None

@dataclass
class MetricSummary:
    experiment_id: str
    variant: str
    metric_type: MetricType
    count: int
    sum_value: float
    mean_value: float
    median_value: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    time_period_start: datetime
    time_period_end: datetime
    conversion_rate: Optional[float] = None  # For binary metrics

    def to_dict(self) -> Dict[str, Any]:

        return {
            "experiment_id": self.experiment_id,
            "variant": self.variant,
            "metric_type": self.metric_type.value,
            "count": self.count,
            "sum_value": self.sum_value,
            "mean_value": self.mean_value,
            "median_value": self.median_value,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "percentile_95": self.percentile_95,
            "time_period_start": self.time_period_start.isoformat(),
            "time_period_end": self.time_period_end.isoformat(),
            "conversion_rate": self.conversion_rate
        }

@dataclass
class ExperimentMetrics:
    experiment_id: str
    variant_metrics: Dict[str, Dict[MetricType, MetricSummary]]
    primary_metrics: Dict[MetricType, Dict[str, MetricSummary]]
    secondary_metrics: Dict[MetricType, Dict[str, MetricSummary]]
    time_series_data: Dict[str, List[Dict[str, Any]]]
    last_updated: datetime
    total_events: int

    def to_dict(self) -> Dict[str, Any]:

        return {
            "experiment_id": self.experiment_id,
            "variant_metrics": {
                variant: {metric_type.value: summary.to_dict()
                         for metric_type, summary in metrics.items()}
                for variant, metrics in self.variant_metrics.items()
            },
            "primary_metrics": {
                metric_type.value: {variant: summary.to_dict()
                                   for variant, summary in variants.items()}
                for metric_type, variants in self.primary_metrics.items()
            },
            "secondary_metrics": {
                metric_type.value: {variant: summary.to_dict()
                                   for variant, summary in variants.items()}
                for metric_type, variants in self.secondary_metrics.items()
            },
            "time_series_data": self.time_series_data,
            "last_updated": self.last_updated.isoformat(),
            "total_events": self.total_events
        }

class MetricsTracker:
    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or DEFAULT_AB_TEST_CONFIG

        self.redis_client = None
        self._initialize_redis()

        self.metric_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_summaries: Dict[str, ExperimentMetrics] = {}

        self.metrics = {
            "events_processed": 0,
            "summaries_calculated": 0,
            "anomalies_detected": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "processing_time_ms": []
        }

        self.is_running = False
        self.background_tasks = set()

        logger.info("Metrics tracker initialized")

    def _initialize_redis(self):
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(
                    self.config.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()
                logger.info("Redis connection established for metrics")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None

    async def start_tracking(self):
        if self.is_running:
            logger.warning("Metrics tracking already running")
            return

        self.is_running = True
        logger.info("Starting metrics tracking...")

        try:
            aggregation_task = asyncio.create_task(self._background_aggregation())
            self.background_tasks.add(aggregation_task)

            anomaly_task = asyncio.create_task(self._background_anomaly_detection())
            self.background_tasks.add(anomaly_task)

            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in metrics tracking: {e}")
        finally:
            self.is_running = False
            logger.info("Metrics tracking stopped")

    def stop_tracking(self):
        logger.info("Stopping metrics tracking...")
        self.is_running = False

        for task in self.background_tasks:
            if not task.done():
                task.cancel()

        self.background_tasks.clear()

    async def record_metric(
        self,
        experiment_id: str,
        variant: str,
        user_id: str,
        metric_type: MetricType,
        metric_value: float,
        session_id: Optional[str] = None,
        user_attributes: Optional[Dict[str, Any]] = None,
        event_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:

        start_time = time.time()

        try:
            event = MetricEvent(
                experiment_id=experiment_id,
                variant=variant,
                user_id=user_id,
                metric_type=metric_type,
                metric_value=metric_value,
                timestamp=datetime.now(),
                session_id=session_id,
                user_attributes=user_attributes,
                event_metadata=event_metadata
            )

            key = f"{experiment_id}:{variant}:{metric_type.value}"
            self.metric_events[key].append(event)

            if self.redis_client:
                await self._store_event_in_redis(event)

            await self._check_metric_anomaly(event)

            self.metrics["events_processed"] += 1

            processing_time = (time.time() - start_time) * 1000
            self.metrics["processing_time_ms"].append(processing_time)

            logger.debug(f"Metric recorded: {metric_type.value}={metric_value} for {variant} in {experiment_id}")
            return True

        except Exception as e:
            logger.error(f"Error recording metric: {e}")
            return False

    async def record_conversion(
        self,
        experiment_id: str,
        variant: str,
        user_id: str,
        converted: bool = True,
        conversion_value: float = 1.0,
        session_id: Optional[str] = None
    ) -> bool:

        return await self.record_metric(
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            metric_type=MetricType.CONVERSION_RATE,
            metric_value=conversion_value if converted else 0.0,
            session_id=session_id,
            event_metadata={"converted": converted}
        )

    async def record_revenue(
        self,
        experiment_id: str,
        variant: str,
        user_id: str,
        revenue_amount: float,
        session_id: Optional[str] = None
    ) -> bool:

        return await self.record_metric(
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            metric_type=MetricType.REVENUE_PER_USER,
            metric_value=revenue_amount,
            session_id=session_id
        )

    async def get_experiment_metrics(
        self,
        experiment_id: str,
        time_window_hours: int = 24
    ) -> Optional[ExperimentMetrics]:

        try:
            cache_key = f"metrics:{experiment_id}:{time_window_hours}h"
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    self.metrics["cache_hits"] += 1
                    return self._deserialize_experiment_metrics(json.loads(cached_data))

            self.metrics["cache_misses"] += 1

            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_window_hours)

            experiment_metrics = await self._calculate_experiment_metrics(
                experiment_id, start_time, end_time
            )

            if self.redis_client and experiment_metrics:
                self.redis_client.setex(
                    cache_key,
                    300,  # 5 minute TTL
                    json.dumps(experiment_metrics.to_dict())
                )

            return experiment_metrics

        except Exception as e:
            logger.error(f"Error getting experiment metrics: {e}")
            return None

    async def _calculate_experiment_metrics(
        self,
        experiment_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[ExperimentMetrics]:

        experiment_events = []

        for key, events in self.metric_events.items():
            if key.startswith(experiment_id):
                for event in events:
                    if start_time <= event.timestamp <= end_time:
                        experiment_events.append(event)

        if not experiment_events:
            return None

        variant_data = defaultdict(lambda: defaultdict(list))

        for event in experiment_events:
            variant_data[event.variant][event.metric_type].append(event)

        variant_metrics = {}
        primary_metrics = defaultdict(dict)
        secondary_metrics = defaultdict(dict)

        for variant, metric_data in variant_data.items():
            variant_metrics[variant] = {}

            for metric_type, events in metric_data.items():
                summary = self._calculate_metric_summary(
                    experiment_id, variant, metric_type, events, start_time, end_time
                )

                variant_metrics[variant][metric_type] = summary

                if metric_type in self.config.primary_metrics:
                    primary_metrics[metric_type][variant] = summary
                elif metric_type in self.config.secondary_metrics:
                    secondary_metrics[metric_type][variant] = summary

        time_series = await self._generate_time_series(experiment_id, start_time, end_time)

        return ExperimentMetrics(
            experiment_id=experiment_id,
            variant_metrics=variant_metrics,
            primary_metrics=dict(primary_metrics),
            secondary_metrics=dict(secondary_metrics),
            time_series_data=time_series,
            last_updated=datetime.now(),
            total_events=len(experiment_events)
        )

    def _calculate_metric_summary(
        self,
        experiment_id: str,
        variant: str,
        metric_type: MetricType,
        events: List[MetricEvent],
        start_time: datetime,
        end_time: datetime
    ) -> MetricSummary:

        values = [event.metric_value for event in events]

        if not values:
            return MetricSummary(
                experiment_id=experiment_id,
                variant=variant,
                metric_type=metric_type,
                count=0,
                sum_value=0.0,
                mean_value=0.0,
                median_value=0.0,
                std_dev=0.0,
                min_value=0.0,
                max_value=0.0,
                percentile_95=0.0,
                time_period_start=start_time,
                time_period_end=end_time
            )

        count = len(values)
        sum_value = sum(values)
        mean_value = sum_value / count

        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            median_value = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            median_value = sorted_values[n//2]

        variance = sum((x - mean_value) ** 2 for x in values) / count
        std_dev = variance ** 0.5

        percentile_95_idx = int(0.95 * (n - 1))
        percentile_95 = sorted_values[percentile_95_idx]

        conversion_rate = None
        if metric_type == MetricType.CONVERSION_RATE:
            conversions = sum(1 for x in values if x > 0)
            conversion_rate = conversions / count if count > 0 else 0.0

        return MetricSummary(
            experiment_id=experiment_id,
            variant=variant,
            metric_type=metric_type,
            count=count,
            sum_value=sum_value,
            mean_value=mean_value,
            median_value=median_value,
            std_dev=std_dev,
            min_value=min(values),
            max_value=max(values),
            percentile_95=percentile_95,
            time_period_start=start_time,
            time_period_end=end_time,
            conversion_rate=conversion_rate
        )

    async def _generate_time_series(
        self,
        experiment_id: str,
        start_time: datetime,
        end_time: datetime,
        bucket_size_minutes: int = 60
    ) -> Dict[str, List[Dict[str, Any]]]:

        time_series = defaultdict(list)

        current_time = start_time
        while current_time < end_time:
            bucket_end = min(current_time + timedelta(minutes=bucket_size_minutes), end_time)

            bucket_events = defaultdict(lambda: defaultdict(list))

            for key, events in self.metric_events.items():
                if key.startswith(experiment_id):
                    for event in events:
                        if current_time <= event.timestamp < bucket_end:
                            bucket_events[event.variant][event.metric_type].append(event.metric_value)

            for variant, metric_data in bucket_events.items():
                for metric_type, values in metric_data.items():
                    if values:
                        time_series[f"{variant}_{metric_type.value}"].append({
                            "timestamp": current_time.isoformat(),
                            "value": sum(values) / len(values),
                            "count": len(values)
                        })

            current_time = bucket_end

        return dict(time_series)

    async def _store_event_in_redis(self, event: MetricEvent):
        if not self.redis_client:
            return

        try:
            key = f"metrics:events:{event.experiment_id}:{event.variant}:{event.metric_type.value}"
            event_data = {
                "user_id": event.user_id,
                "metric_value": event.metric_value,
                "timestamp": event.timestamp.isoformat(),
                "session_id": event.session_id,
                "user_attributes": event.user_attributes,
                "event_metadata": event.event_metadata
            }

            pipe = self.redis_client.pipeline()
            pipe.lpush(key, json.dumps(event_data))
            pipe.ltrim(key, 0, 9999)  # Keep last 10k events
            pipe.expire(key, 86400 * 7)  # 7 days TTL
            pipe.execute()

        except Exception as e:
            logger.error(f"Error storing event in Redis: {e}")

    async def _check_metric_anomaly(self, event: MetricEvent):
        try:
            key = f"{event.experiment_id}:{event.variant}:{event.metric_type.value}"
            recent_events = list(self.metric_events[key])

            if len(recent_events) < 100:  # Need sufficient data
                return

            recent_values = [e.metric_value for e in recent_events[-100:]]
            mean_value = sum(recent_values) / len(recent_values)

            if len(recent_values) > 1:
                variance = sum((x - mean_value) ** 2 for x in recent_values) / (len(recent_values) - 1)
                std_dev = variance ** 0.5
            else:
                std_dev = 0

            if std_dev > 0 and abs(event.metric_value - mean_value) > 3 * std_dev:
                self.metrics["anomalies_detected"] += 1
                logger.warning(
                    f"Metric anomaly detected: {event.metric_type.value}={event.metric_value} "
                    f"in {event.variant} (mean={mean_value:.2f}, std={std_dev:.2f})"
                )

        except Exception as e:
            logger.error(f"Error checking metric anomaly: {e}")

    async def _background_aggregation(self):
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                experiment_ids = set()
                for key in self.metric_events.keys():
                    experiment_id = key.split(':')[0]
                    experiment_ids.add(experiment_id)

                for experiment_id in experiment_ids:
                    await self.get_experiment_metrics(experiment_id)
                    self.metrics["summaries_calculated"] += 1

                logger.debug(f"Background aggregation completed for {len(experiment_ids)} experiments")

            except Exception as e:
                logger.error(f"Error in background aggregation: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def _background_anomaly_detection(self):
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Run every minute

                current_time = datetime.now()
                anomaly_count = 0

                for key, events in self.metric_events.items():
                    if not events:
                        continue

                    recent_events = [e for e in list(events)[-10:]
                                   if (current_time - e.timestamp).seconds < 300]  # Last 5 minutes

                    for event in recent_events:
                        await self._check_metric_anomaly(event)
                        anomaly_count += 1

                logger.debug(f"Anomaly detection completed: checked {anomaly_count} recent events")

            except Exception as e:
                logger.error(f"Error in background anomaly detection: {e}")
                await asyncio.sleep(60)

    def _deserialize_experiment_metrics(self, data: Dict[str, Any]) -> ExperimentMetrics:

        return ExperimentMetrics(
            experiment_id=data["experiment_id"],
            variant_metrics={},  # Would deserialize properly
            primary_metrics={},
            secondary_metrics={},
            time_series_data=data.get("time_series_data", {}),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            total_events=data["total_events"]
        )

    def get_metrics(self) -> Dict[str, Any]:

        avg_processing_time = 0
        if self.metrics["processing_time_ms"]:
            avg_processing_time = sum(self.metrics["processing_time_ms"]) / len(self.metrics["processing_time_ms"])

        cache_hit_rate = 0
        total_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total_requests > 0:
            cache_hit_rate = self.metrics["cache_hits"] / total_requests

        return {
            "events_processed": self.metrics["events_processed"],
            "summaries_calculated": self.metrics["summaries_calculated"],
            "anomalies_detected": self.metrics["anomalies_detected"],
            "cache_hit_rate": cache_hit_rate,
            "avg_processing_time_ms": avg_processing_time,
            "active_experiments": len(set(key.split(':')[0] for key in self.metric_events.keys())),
            "total_metric_keys": len(self.metric_events),
            "redis_connected": self.redis_client is not None
        }

    def health_check(self) -> Dict[str, Any]:

        redis_healthy = True
        if self.redis_client:
            try:
                self.redis_client.ping()
            except Exception:
                redis_healthy = False

        return {
            "status": "healthy" if redis_healthy or not REDIS_AVAILABLE else "unhealthy",
            "is_running": self.is_running,
            "redis_healthy": redis_healthy,
            "events_in_memory": sum(len(events) for events in self.metric_events.values()),
            "metrics": self.get_metrics()
        }
