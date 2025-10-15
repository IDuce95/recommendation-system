import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - system metrics disabled")

from .config import RealtimeConfig, DEFAULT_REALTIME_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AggregatedMetric:
    name: str
    count: int
    sum_value: float
    min_value: float
    max_value: float
    avg_value: float
    p50_value: float
    p95_value: float
    p99_value: float
    window_start: datetime
    window_end: datetime
    tags: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    def __init__(self, config: Optional[RealtimeConfig] = None):
        self.config = config or DEFAULT_REALTIME_CONFIG

        self.raw_metrics: deque = deque(maxlen=10000)  # Keep last 10k measurements
        self.aggregated_metrics: Dict[str, List[AggregatedMetric]] = defaultdict(list)

        self.aggregation_windows = [60, 300, 3600]  # 1min, 5min, 1hour

        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)

        self.alert_thresholds = {
            "response_time_ms": self.config.max_response_time_ms * 1.5,
            "error_rate": 0.05,  # 5% error rate
            "memory_usage_percent": 85.0,
            "cpu_usage_percent": 90.0,
            "kafka_lag": 1000,  # 1000 messages
            "cache_miss_rate": 0.3  # 30% cache miss rate
        }

        self.is_running = False
        self.collection_tasks = set()
        self.last_aggregation = time.time()
        self.last_system_check = time.time()

        self.executor = ThreadPoolExecutor(max_workers=2)

        logger.info("Metrics collector initialized")

    async def start_collection(self):
        if self.is_running:
            logger.warning("Metrics collector is already running")
            return

        self.is_running = True
        logger.info("Starting metrics collection...")

        try:
            aggregation_task = asyncio.create_task(self._aggregation_loop())
            system_monitoring_task = asyncio.create_task(self._system_monitoring_loop())
            alert_task = asyncio.create_task(self._alert_loop())

            self.collection_tasks.update([
                aggregation_task,
                system_monitoring_task,
                alert_task
            ])

            await asyncio.gather(*self.collection_tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error in metrics collection: {e}")
        finally:
            self.is_running = False
            logger.info("Metrics collection stopped")

    def stop_collection(self):
        logger.info("Stopping metrics collection...")
        self.is_running = False

        for task in self.collection_tasks:
            if not task.done():
                task.cancel()

        self.collection_tasks.clear()
        self.executor.shutdown(wait=True)

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata
        )

        self.raw_metrics.append(metric_point)

        if name.endswith("_count"):
            self.counters[name] += value
        elif name.endswith("_gauge"):
            self.gauges[name] = value
        elif name.endswith("_histogram"):
            self.histograms[name].append(value)
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]

    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        self.record_metric(f"{name}_count", value, tags)

    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        self.record_metric(f"{name}_gauge", value, tags)

    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        self.record_metric(f"{name}_histogram", value, tags)

    def record_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        self.record_histogram(f"{name}_timing", duration_ms, tags)

    async def _aggregation_loop(self):
        while self.is_running:
            try:
                current_time = time.time()

                if current_time - self.last_aggregation >= 10:
                    await self._aggregate_metrics()
                    self.last_aggregation = current_time

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(5)

    async def _aggregate_metrics(self):
        try:
            current_time = datetime.now()

            for window_seconds in self.aggregation_windows:
                window_start = current_time - timedelta(seconds=window_seconds)

                grouped_metrics = defaultdict(list)

                for metric in self.raw_metrics:
                    if metric.timestamp >= window_start:
                        key = (metric.name, tuple(sorted(metric.tags.items())))
                        grouped_metrics[key].append(metric.value)

                for (name, tags_tuple), values in grouped_metrics.items():
                    if not values:
                        continue

                    tags = dict(tags_tuple)
                    aggregated = AggregatedMetric(
                        name=name,
                        count=len(values),
                        sum_value=sum(values),
                        min_value=min(values),
                        max_value=max(values),
                        avg_value=statistics.mean(values),
                        p50_value=statistics.median(values),
                        p95_value=self._percentile(values, 0.95),
                        p99_value=self._percentile(values, 0.99),
                        window_start=window_start,
                        window_end=current_time,
                        tags=tags
                    )

                    window_key = f"{name}_{window_seconds}s"
                    self.aggregated_metrics[window_key].append(aggregated)

                    if len(self.aggregated_metrics[window_key]) > 100:
                        self.aggregated_metrics[window_key] = self.aggregated_metrics[window_key][-100:]

            logger.debug("Metrics aggregation completed")

        except Exception as e:
            logger.error(f"Error aggregating metrics: {e}")

    def _percentile(self, values: List[float], percentile: float) -> float:

        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        if index >= len(sorted_values):
            index = len(sorted_values) - 1

        return sorted_values[index]

    async def _system_monitoring_loop(self):
        while self.is_running:
            try:
                current_time = time.time()

                if current_time - self.last_system_check >= 30:
                    await self._collect_system_metrics()
                    self.last_system_check = current_time

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(10)

    async def _collect_system_metrics(self):
        try:
            if not PSUTIL_AVAILABLE:
                return

            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("system_cpu_usage", cpu_percent, {"unit": "percent"})

            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_usage", memory.percent, {"unit": "percent"})
            self.set_gauge("system_memory_available", memory.available / (1024**3), {"unit": "GB"})

            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.set_gauge("system_disk_usage", disk_percent, {"unit": "percent"})

            try:
                network = psutil.net_io_counters()
                self.set_gauge("system_network_bytes_sent", network.bytes_sent, {"unit": "bytes"})
                self.set_gauge("system_network_bytes_recv", network.bytes_recv, {"unit": "bytes"})
            except Exception:
                pass  # Network stats might not be available

            process = psutil.Process()
            self.set_gauge("process_memory_usage", process.memory_percent(), {"unit": "percent"})
            self.set_gauge("process_cpu_usage", process.cpu_percent(), {"unit": "percent"})

            logger.debug("System metrics collected")

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    async def _alert_loop(self):
        while self.is_running:
            try:
                await self._check_alerts()
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
                await asyncio.sleep(60)

    async def _check_alerts(self):
        try:
            alerts = []

            for metric_key, aggregated_list in self.aggregated_metrics.items():
                if not metric_key.endswith("_60s") or not aggregated_list:
                    continue

                latest_metric = aggregated_list[-1]
                metric_name = latest_metric.name

                if "response_time" in metric_name or "processing_time" in metric_name:
                    if latest_metric.p95_value > self.alert_thresholds["response_time_ms"]:
                        alerts.append({
                            "type": "high_response_time",
                            "metric": metric_name,
                            "value": latest_metric.p95_value,
                            "threshold": self.alert_thresholds["response_time_ms"],
                            "severity": "warning"
                        })

                if "error_rate" in metric_name:
                    if latest_metric.avg_value > self.alert_thresholds["error_rate"]:
                        alerts.append({
                            "type": "high_error_rate",
                            "metric": metric_name,
                            "value": latest_metric.avg_value,
                            "threshold": self.alert_thresholds["error_rate"],
                            "severity": "critical"
                        })

            if "system_cpu_usage_gauge" in self.gauges:
                cpu_usage = self.gauges["system_cpu_usage_gauge"]
                if cpu_usage > self.alert_thresholds["cpu_usage_percent"]:
                    alerts.append({
                        "type": "high_cpu_usage",
                        "metric": "system_cpu_usage",
                        "value": cpu_usage,
                        "threshold": self.alert_thresholds["cpu_usage_percent"],
                        "severity": "warning"
                    })

            if "system_memory_usage_gauge" in self.gauges:
                memory_usage = self.gauges["system_memory_usage_gauge"]
                if memory_usage > self.alert_thresholds["memory_usage_percent"]:
                    alerts.append({
                        "type": "high_memory_usage",
                        "metric": "system_memory_usage",
                        "value": memory_usage,
                        "threshold": self.alert_thresholds["memory_usage_percent"],
                        "severity": "warning"
                    })

            for alert in alerts:
                logger.warning(f"ALERT [{alert['severity']}]: {alert['type']} - {alert['metric']}={alert['value']:.2f} > {alert['threshold']}")

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    def get_metrics_summary(self) -> Dict[str, Any]:

        current_time = datetime.now()

        recent_metrics = [
            m for m in self.raw_metrics
            if current_time - m.timestamp <= timedelta(minutes=5)
        ]

        summary = {
            "collection_status": {
                "is_running": self.is_running,
                "total_metrics_collected": len(self.raw_metrics),
                "recent_metrics_count": len(recent_metrics),
                "collection_rate_per_minute": len(recent_metrics) / 5.0
            },
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "aggregated_metrics": {},
            "system_health": self._get_system_health(),
            "alerts": self._get_recent_alerts()
        }

        for window_seconds in self.aggregation_windows:
            window_name = f"{window_seconds}s"
            summary["aggregated_metrics"][window_name] = {}

            for metric_key, aggregated_list in self.aggregated_metrics.items():
                if metric_key.endswith(f"_{window_seconds}s") and aggregated_list:
                    latest = aggregated_list[-1]
                    metric_base_name = latest.name

                    summary["aggregated_metrics"][window_name][metric_base_name] = {
                        "count": latest.count,
                        "avg": latest.avg_value,
                        "p50": latest.p50_value,
                        "p95": latest.p95_value,
                        "p99": latest.p99_value,
                        "min": latest.min_value,
                        "max": latest.max_value
                    }

        return summary

    def _get_system_health(self) -> Dict[str, Any]:

        health = {
            "status": "unknown",
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0
        }

        if PSUTIL_AVAILABLE:
            try:
                health["cpu_usage"] = self.gauges.get("system_cpu_usage_gauge", 0.0)
                health["memory_usage"] = self.gauges.get("system_memory_usage_gauge", 0.0)
                health["disk_usage"] = self.gauges.get("system_disk_usage_gauge", 0.0)

                if (health["cpu_usage"] < 80 and
                    health["memory_usage"] < 80 and
                    health["disk_usage"] < 90):
                    health["status"] = "healthy"
                elif (health["cpu_usage"] < 90 and
                      health["memory_usage"] < 90 and
                      health["disk_usage"] < 95):
                    health["status"] = "warning"
                else:
                    health["status"] = "critical"

            except Exception:
                health["status"] = "error"

        return health

    def _get_recent_alerts(self) -> List[Dict[str, Any]]:

        return []

    def export_metrics_prometheus(self) -> str:

        lines = []
        timestamp = int(time.time() * 1000)

        for name, value in self.counters.items():
            clean_name = name.replace("_count", "").replace("-", "_")
            lines.append(f"# TYPE {clean_name} counter")
            lines.append(f"{clean_name} {value} {timestamp}")

        for name, value in self.gauges.items():
            clean_name = name.replace("_gauge", "").replace("-", "_")
            lines.append(f"# TYPE {clean_name} gauge")
            lines.append(f"{clean_name} {value} {timestamp}")

        for name, values in self.histograms.items():
            if not values:
                continue

            clean_name = name.replace("_histogram", "").replace("-", "_")
            sorted_values = sorted(values)

            percentiles = [0.5, 0.95, 0.99]
            for p in percentiles:
                percentile_value = self._percentile(sorted_values, p)
                quantile_name = f"{clean_name}_quantile"
                lines.append(f"# TYPE {quantile_name} gauge")
                lines.append(f'{quantile_name}{{quantile="{p}"}} {percentile_value} {timestamp}')

        return "\n".join(lines)

    def health_check(self) -> Dict[str, Any]:

        system_health = self._get_system_health()

        is_healthy = (
            self.is_running and
            system_health["status"] in ["healthy", "warning"] and
            len(self.raw_metrics) > 0
        )

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "is_running": self.is_running,
            "metrics_collected": len(self.raw_metrics),
            "system_health": system_health,
            "collection_rate": len([
                m for m in self.raw_metrics
                if datetime.now() - m.timestamp <= timedelta(minutes=1)
            ]),
            "psutil_available": PSUTIL_AVAILABLE
        }

def create_metrics_collector(config: Optional[RealtimeConfig] = None) -> MetricsCollector:

    collector = MetricsCollector(config)

    if config:
        collector.alert_thresholds.update({
            "response_time_ms": config.max_response_time_ms * 1.5,
            "circuit_breaker_threshold": config.circuit_breaker_failure_threshold
        })

    return collector
