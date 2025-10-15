import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .config import ABTestConfig, DEFAULT_AB_TEST_CONFIG, ExperimentType, ExperimentStatus
from .experiment_manager import ExperimentManager
from .statistical_analyzer import StatisticalAnalyzer
from .metrics_tracker import MetricsTracker
from .experiment_service import ExperimentService
from .bandit_optimizer import BanditOptimizer

logger = logging.getLogger(__name__)

@dataclass
class ABTestingSystemStatus:
    total_experiments: int = 0
    active_experiments: int = 0
    completed_experiments: int = 0
    total_users_assigned: int = 0
    total_metrics_recorded: int = 0
    active_bandits: int = 0
    system_health: str = "healthy"
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class ABTestingSystem:
    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or DEFAULT_AB_TEST_CONFIG

        self.experiment_manager = ExperimentManager(self.config)
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        self.metrics_tracker = MetricsTracker(self.config)
        self.bandit_optimizer = BanditOptimizer(self.config)
        self.experiment_service = ExperimentService(self.config)

        self.is_initialized = False
        self.background_tasks = []

        logger.info("A/B Testing System initialized")

    async def initialize(self) -> bool:

        try:
            await self.experiment_manager.initialize()
            await self.metrics_tracker.initialize()

            await self._start_background_tasks()

            self.is_initialized = True
            logger.info("A/B Testing System successfully initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize A/B Testing System: {e}")
            return False

    async def shutdown(self):
        try:
            for task in self.background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            await self.metrics_tracker.cleanup()

            logger.info("A/B Testing System shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        experiment_type: ExperimentType = ExperimentType.AB_TEST,
        target_metric: str = "conversion_rate",
        sample_size_per_variant: Optional[int] = None,
        duration_days: int = 14,
        traffic_allocation: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:

        try:
            experiment_id = await self.experiment_service.create_experiment(
                name=name,
                description=description,
                variants=variants,
                experiment_type=experiment_type,
                target_metric=target_metric,
                sample_size_per_variant=sample_size_per_variant,
                duration_days=duration_days,
                traffic_allocation=traffic_allocation,
                metadata=metadata
            )

            if experiment_type == ExperimentType.BANDIT:
                variant_names = [v.get("name", f"variant_{i}") for i, v in enumerate(variants)]
                self.bandit_optimizer.create_bandit_experiment(
                    experiment_id, variant_names
                )

            logger.info(f"Created experiment: {experiment_id}")
            return experiment_id

        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            return None

    async def start_experiment(self, experiment_id: str) -> bool:

        return await self.experiment_service.start_experiment(experiment_id)

    async def stop_experiment(self, experiment_id: str, reason: str = "Manual stop") -> bool:

        return await self.experiment_service.stop_experiment(experiment_id, reason)

    async def assign_user(self, experiment_id: str, user_id: str) -> Optional[str]:

        experiment = await self.experiment_manager.get_experiment(experiment_id)
        if not experiment:
            return None

        if experiment.get("experiment_type") == ExperimentType.BANDIT.value:
            variant = self.bandit_optimizer.select_variant(experiment_id)
            if variant:
                await self.experiment_manager.assign_user(experiment_id, user_id, variant)
                return variant

        return await self.experiment_manager.assign_user(experiment_id, user_id)

    async def record_metric(
        self,
        experiment_id: str,
        user_id: str,
        metric_name: str,
        metric_value: float,
        variant: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:

        success = await self.experiment_service.record_metric(
            experiment_id, user_id, metric_name, metric_value, variant, metadata
        )

        if success and variant:
            experiment = await self.experiment_manager.get_experiment(experiment_id)
            if experiment and experiment.get("experiment_type") == ExperimentType.BANDIT.value:
                reward = min(1.0, max(0.0, metric_value))
                self.bandit_optimizer.update_bandit(experiment_id, variant, reward)

        return success

    async def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:

        return await self.experiment_service.get_experiment_results(experiment_id)

    async def get_statistical_analysis(
        self,
        experiment_id: str,
        metric_name: str,
        confidence_level: float = 0.95
    ) -> Optional[Dict[str, Any]]:

        return await self.experiment_service.get_statistical_analysis(
            experiment_id, metric_name, confidence_level
        )

    async def check_experiment_significance(
        self,
        experiment_id: str,
        metric_name: str = None
    ) -> Dict[str, Any]:

        return await self.experiment_service.check_experiment_significance(
            experiment_id, metric_name
        )

    async def get_experiment_recommendations(self, experiment_id: str) -> Dict[str, Any]:

        return await self.experiment_service.get_experiment_recommendations(experiment_id)

    def get_bandit_performance(self, experiment_id: str) -> Dict[str, Any]:

        return self.bandit_optimizer.get_bandit_performance(experiment_id)

    def get_traffic_allocation(self, experiment_id: str) -> Dict[str, float]:

        return self.bandit_optimizer.get_traffic_allocation(experiment_id)

    async def get_system_status(self) -> ABTestingSystemStatus:

        try:
            experiments = await self.experiment_manager.list_experiments()
            active_experiments = len([e for e in experiments if e.get("status") == ExperimentStatus.RUNNING.value])
            completed_experiments = len([e for e in experiments if e.get("status") == ExperimentStatus.COMPLETED.value])

            metrics_stats = await self.metrics_tracker.get_metrics_summary()
            bandit_metrics = self.bandit_optimizer.get_metrics()

            return ABTestingSystemStatus(
                total_experiments=len(experiments),
                active_experiments=active_experiments,
                completed_experiments=completed_experiments,
                total_users_assigned=sum(e.get("user_count", 0) for e in experiments),
                total_metrics_recorded=metrics_stats.get("total_events", 0),
                active_bandits=bandit_metrics.get("active_bandits", 0),
                system_health=await self._check_system_health(),
                last_updated=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return ABTestingSystemStatus(system_health="error")

    async def health_check(self) -> Dict[str, Any]:

        health_results = {}

        try:
            health_results["experiment_manager"] = await self.experiment_manager.health_check()
            health_results["statistical_analyzer"] = self.statistical_analyzer.health_check()
            health_results["metrics_tracker"] = await self.metrics_tracker.health_check()
            health_results["bandit_optimizer"] = self.bandit_optimizer.health_check()
            health_results["experiment_service"] = await self.experiment_service.health_check()

            all_healthy = all(
                result.get("status") == "healthy"
                for result in health_results.values()
            )

            health_results["system"] = {
                "status": "healthy" if all_healthy else "degraded",
                "initialized": self.is_initialized,
                "background_tasks": len(self.background_tasks),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_results["system"] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

        return health_results

    async def _start_background_tasks(self):
        metrics_task = asyncio.create_task(self._metrics_aggregation_task())
        self.background_tasks.append(metrics_task)

        monitoring_task = asyncio.create_task(self._experiment_monitoring_task())
        self.background_tasks.append(monitoring_task)

        health_task = asyncio.create_task(self._health_monitoring_task())
        self.background_tasks.append(health_task)

        logger.info(f"Started {len(self.background_tasks)} background tasks")

    async def _metrics_aggregation_task(self):
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self.metrics_tracker.aggregate_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics aggregation task: {e}")
                await asyncio.sleep(60)

    async def _experiment_monitoring_task(self):
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes

                experiments = await self.experiment_manager.list_experiments()
                for experiment in experiments:
                    if experiment.get("status") == ExperimentStatus.RUNNING.value:
                        await self._check_experiment_completion(experiment["id"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in experiment monitoring task: {e}")
                await asyncio.sleep(60)

    async def _health_monitoring_task(self):
        while True:
            try:
                await asyncio.sleep(900)  # Run every 15 minutes

                health = await self.health_check()
                if health.get("system", {}).get("status") != "healthy":
                    logger.warning(f"System health degraded: {health}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring task: {e}")
                await asyncio.sleep(60)

    async def _check_experiment_completion(self, experiment_id: str):
        try:
            recommendations = await self.get_experiment_recommendations(experiment_id)

            if recommendations.get("should_stop", False):
                reason = recommendations.get("stop_reason", "Automated completion")
                await self.stop_experiment(experiment_id, reason)
                logger.info(f"Auto-completed experiment {experiment_id}: {reason}")

        except Exception as e:
            logger.error(f"Error checking experiment completion: {e}")

    async def _check_system_health(self) -> str:

        try:
            health = await self.health_check()
            system_health = health.get("system", {}).get("status", "unknown")
            return system_health

        except Exception:
            return "error"

async def create_ab_testing_system(config: Optional[ABTestConfig] = None) -> ABTestingSystem:

    system = ABTestingSystem(config)
    await system.initialize()
    return system

def get_default_config() -> ABTestConfig:

    return DEFAULT_AB_TEST_CONFIG

__all__ = [
    "ABTestingSystem",
    "ABTestingSystemStatus",
    "ExperimentManager",
    "StatisticalAnalyzer",
    "MetricsTracker",
    "ExperimentService",
    "BanditOptimizer",
    "ABTestConfig",
    "ExperimentType",
    "ExperimentStatus",
    "create_ab_testing_system",
    "get_default_config"
]

from .experiment_manager import ExperimentManager
from .statistical_analyzer import StatisticalAnalyzer
from .metrics_tracker import MetricsTracker
from .experiment_service import ExperimentService
from .bandit_optimizer import BanditOptimizer
from .config import ABTestConfig

__version__ = "1.0.0"

__all__ = [
    "ExperimentManager",
    "StatisticalAnalyzer",
    "MetricsTracker",
    "ExperimentService",
    "BanditOptimizer",
    "ABTestConfig",
    "ABTestingSystem",
    "create_experiment_system"
]

class ABTestingSystem:
    def __init__(self, config=None):
        self.config = config
        self.experiment_manager = ExperimentManager(config)
        self.statistical_analyzer = StatisticalAnalyzer(config)
        self.metrics_tracker = MetricsTracker(config)
        self.experiment_service = ExperimentService(
            self.experiment_manager,
            self.statistical_analyzer,
            self.metrics_tracker,
            config
        )
        self.bandit_optimizer = BanditOptimizer(config)

    async def start_experiment(self, experiment_name: str, **kwargs):
        return await self.experiment_service.create_experiment(experiment_name, **kwargs)

    async def get_experiment_results(self, experiment_id: str):
        return await self.experiment_service.get_experiment_results(experiment_id)

    def health_check(self):
        return self.experiment_service.health_check()

def create_experiment_system(config=None):
    return ABTestingSystem(config)
