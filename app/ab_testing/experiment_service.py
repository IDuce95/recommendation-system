import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from .config import (
    ABTestConfig, ExperimentConfig, ExperimentType, ExperimentStatus,
    MetricType, DEFAULT_AB_TEST_CONFIG
)
from .experiment_manager import ExperimentManager, Experiment, ExperimentAssignment
from .statistical_analyzer import StatisticalAnalyzer, ExperimentAnalysis, StatisticalResult
from .metrics_tracker import MetricsTracker, ExperimentMetrics

logger = logging.getLogger(__name__)

class ExperimentService:
    def __init__(
        self,
        experiment_manager: Optional[ExperimentManager] = None,
        statistical_analyzer: Optional[StatisticalAnalyzer] = None,
        metrics_tracker: Optional[MetricsTracker] = None,
        config: Optional[ABTestConfig] = None
    ):
        self.config = config or DEFAULT_AB_TEST_CONFIG

        self.experiment_manager = experiment_manager or ExperimentManager(self.config)
        self.statistical_analyzer = statistical_analyzer or StatisticalAnalyzer(self.config)
        self.metrics_tracker = metrics_tracker or MetricsTracker(self.config)

        self.is_running = False

        logger.info("Experiment service initialized")

    async def start_service(self):
        if self.is_running:
            logger.warning("Experiment service already running")
            return

        self.is_running = True
        logger.info("Starting experiment service...")

        try:
            await asyncio.create_task(self.metrics_tracker.start_tracking())

        except Exception as e:
            logger.error(f"Error starting experiment service: {e}")
            self.is_running = False
            raise

    def stop_service(self):
        logger.info("Stopping experiment service...")
        self.is_running = False

        self.metrics_tracker.stop_tracking()

    async def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[str],
        traffic_allocation: Dict[str, float],
        target_metrics: List[MetricType],
        experiment_type: ExperimentType = ExperimentType.AB_TEST,
        minimum_sample_size: int = 1000,
        maximum_duration_days: int = 30,
        creator_id: str = "system"
    ) -> Experiment:
        """
        Create a new A/B test experiment.

        Args:
            name: Experiment name
            description: Experiment description
            variants: List of variant names (e.g., ["control", "treatment"])
            traffic_allocation: Percentage allocation per variant
            target_metrics: Primary metrics to track
            experiment_type: Type of experiment
            minimum_sample_size: Minimum sample size per variant
            maximum_duration_days: Maximum experiment duration
            creator_id: ID of experiment creator

        Returns:
            Created Experiment object
        """
        config = ExperimentConfig(
            name=name,
            description=description,
            experiment_type=experiment_type,
            variants=variants,
            traffic_allocation=traffic_allocation,
            target_metrics=target_metrics,
            minimum_sample_size=minimum_sample_size,
            maximum_duration_days=maximum_duration_days
        )

        experiment = await self.experiment_manager.create_experiment(config, creator_id)

        logger.info(f"Experiment created: {name} ({experiment.id})")
        return experiment

    async def start_experiment(self, experiment_id: str) -> bool:

        success = await self.experiment_manager.start_experiment(experiment_id)

        if success:
            logger.info(f"Experiment started: {experiment_id}")
        else:
            logger.error(f"Failed to start experiment: {experiment_id}")

        return success

    async def stop_experiment(self, experiment_id: str, reason: str = "completed") -> bool:

        success = await self.experiment_manager.stop_experiment(experiment_id, reason)

        if success:
            logger.info(f"Experiment stopped: {experiment_id} - {reason}")
        else:
            logger.error(f"Failed to stop experiment: {experiment_id}")

        return success

    async def assign_user_to_experiment(
        self,
        experiment_id: str,
        user_id: str,
        session_id: Optional[str] = None,
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:

        assignment = await self.experiment_manager.assign_user_to_experiment(
            experiment_id, user_id, session_id, user_attributes
        )

        if assignment:
            logger.debug(f"User {user_id} assigned to {assignment.variant} in {experiment_id}")
            return assignment.variant
        else:
            logger.debug(f"User {user_id} not eligible for experiment {experiment_id}")
            return None

    async def record_metric(
        self,
        experiment_id: str,
        variant: str,
        user_id: str,
        metric_type: MetricType,
        metric_value: float,
        session_id: Optional[str] = None
    ) -> bool:

        return await self.metrics_tracker.record_metric(
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            metric_type=metric_type,
            metric_value=metric_value,
            session_id=session_id
        )

    async def record_conversion(
        self,
        experiment_id: str,
        variant: str,
        user_id: str,
        converted: bool = True,
        conversion_value: float = 1.0,
        session_id: Optional[str] = None
    ) -> bool:

        return await self.metrics_tracker.record_conversion(
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            converted=converted,
            conversion_value=conversion_value,
            session_id=session_id
        )

    async def record_revenue(
        self,
        experiment_id: str,
        variant: str,
        user_id: str,
        revenue_amount: float,
        session_id: Optional[str] = None
    ) -> bool:

        return await self.metrics_tracker.record_revenue(
            experiment_id=experiment_id,
            variant=variant,
            user_id=user_id,
            revenue_amount=revenue_amount,
            session_id=session_id
        )

    async def get_experiment_results(
        self,
        experiment_id: str,
        include_analysis: bool = True,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:

        try:
            experiment_stats = await self.experiment_manager.get_experiment_stats(experiment_id)
            if not experiment_stats:
                return {"error": "Experiment not found"}

            experiment_metrics = await self.metrics_tracker.get_experiment_metrics(
                experiment_id, time_window_hours
            )

            results = {
                "experiment_info": experiment_stats,
                "metrics": experiment_metrics.to_dict() if experiment_metrics else {},
                "analysis_timestamp": datetime.now().isoformat()
            }

            if include_analysis and experiment_metrics:
                analysis = await self._perform_experiment_analysis(
                    experiment_id, experiment_metrics
                )
                results["statistical_analysis"] = analysis.to_dict() if analysis else {}

            return results

        except Exception as e:
            logger.error(f"Error getting experiment results: {e}")
            return {"error": str(e)}

    async def _perform_experiment_analysis(
        self,
        experiment_id: str,
        experiment_metrics: ExperimentMetrics
    ) -> Optional[ExperimentAnalysis]:

        try:
            variants = list(experiment_metrics.variant_metrics.keys())
            if len(variants) < 2:
                logger.warning(f"Not enough variants for analysis: {len(variants)}")
                return None

            control_variant = variants[0]
            treatment_variants = variants[1:]

            primary_results = {}
            secondary_results = {}

            for metric_type, variant_data in experiment_metrics.primary_metrics.items():
                if control_variant in variant_data:
                    control_summary = variant_data[control_variant]

                    for treatment_variant in treatment_variants:
                        if treatment_variant in variant_data:
                            treatment_summary = variant_data[treatment_variant]

                            control_data = await self._get_raw_metric_data(
                                experiment_id, control_variant, metric_type
                            )
                            treatment_data = await self._get_raw_metric_data(
                                experiment_id, treatment_variant, metric_type
                            )

                            if control_data and treatment_data:
                                result = await self.statistical_analyzer.analyze_experiment(
                                    experiment_id=experiment_id,
                                    control_data=control_data,
                                    treatment_data=treatment_data,
                                    metric_name=metric_type.value,
                                    control_variant=control_variant,
                                    treatment_variant=treatment_variant
                                )

                                primary_results[f"{control_variant}_vs_{treatment_variant}"] = result

            for metric_type, variant_data in experiment_metrics.secondary_metrics.items():
                if control_variant in variant_data:
                    for treatment_variant in treatment_variants:
                        if treatment_variant in variant_data:
                            control_data = await self._get_raw_metric_data(
                                experiment_id, control_variant, metric_type
                            )
                            treatment_data = await self._get_raw_metric_data(
                                experiment_id, treatment_variant, metric_type
                            )

                            if control_data and treatment_data:
                                result = await self.statistical_analyzer.analyze_experiment(
                                    experiment_id=experiment_id,
                                    control_data=control_data,
                                    treatment_data=treatment_data,
                                    metric_name=metric_type.value,
                                    control_variant=control_variant,
                                    treatment_variant=treatment_variant
                                )

                                secondary_results[f"{control_variant}_vs_{treatment_variant}"] = result

            recommendation = self._generate_experiment_recommendation(primary_results)

            statistical_power = self._calculate_statistical_power(primary_results)

            sample_size_adequate = self._check_sample_size_adequacy(
                experiment_id, experiment_metrics
            )

            early_stopping = self._generate_early_stopping_recommendation(
                primary_results, experiment_metrics
            )

            return ExperimentAnalysis(
                experiment_id=experiment_id,
                variant_results=primary_results,
                primary_metric_results=primary_results,
                secondary_metric_results=secondary_results,
                overall_recommendation=recommendation,
                statistical_power=statistical_power,
                sample_size_adequate=sample_size_adequate,
                early_stopping_recommendation=early_stopping,
                analysis_timestamp=datetime.now(),
                confidence_level=self.config.default_confidence_level
            )

        except Exception as e:
            logger.error(f"Error performing experiment analysis: {e}")
            return None

    async def _get_raw_metric_data(
        self,
        experiment_id: str,
        variant: str,
        metric_type: MetricType
    ) -> List[float]:

        key = f"{experiment_id}:{variant}:{metric_type.value}"

        if hasattr(self.metrics_tracker, 'metric_events'):
            events = self.metrics_tracker.metric_events.get(key, [])
            return [event.metric_value for event in events]

        return []

    def _generate_experiment_recommendation(
        self,
        primary_results: Dict[str, StatisticalResult]
    ) -> str:

        if not primary_results:
            return "insufficient_data"

        significant_results = [
            result for result in primary_results.values()
            if result.is_significant
        ]

        if not significant_results:
            return "no_significant_difference"

        positive_effects = [
            result for result in significant_results
            if result.mean_treatment > result.mean_control
        ]

        if len(positive_effects) == len(significant_results):
            return "treatment_wins"
        elif len(positive_effects) == 0:
            return "control_wins"
        else:
            return "mixed_results"

    def _calculate_statistical_power(
        self,
        primary_results: Dict[str, StatisticalResult]
    ) -> float:

        if not primary_results:
            return 0.0

        powers = []
        for result in primary_results.values():
            power = self.statistical_analyzer.calculate_statistical_power(
                control_data=[result.mean_control] * result.sample_size_control,
                treatment_data=[result.mean_treatment] * result.sample_size_treatment,
                effect_size=result.effect_size
            )
            powers.append(power)

        return sum(powers) / len(powers) if powers else 0.0

    def _check_sample_size_adequacy(
        self,
        experiment_id: str,
        experiment_metrics: ExperimentMetrics
    ) -> bool:

        min_sample_size = self.config.minimum_sample_size_per_variant

        for variant_metrics in experiment_metrics.variant_metrics.values():
            for metric_summary in variant_metrics.values():
                if metric_summary.count < min_sample_size:
                    return False

        return True

    def _generate_early_stopping_recommendation(
        self,
        primary_results: Dict[str, StatisticalResult],
        experiment_metrics: ExperimentMetrics
    ) -> str:

        if not primary_results:
            return "continue"

        very_significant = [
            result for result in primary_results.values()
            if result.p_value < 0.001
        ]

        if very_significant:
            return "stop_early_significant"

        small_effects = [
            result for result in primary_results.values()
            if abs(result.effect_size) < 0.01  # Less than 1% effect
        ]

        if len(small_effects) == len(primary_results):
            return "stop_early_futile"

        return "continue"

    async def get_user_experiments(self, user_id: str) -> List[Dict[str, Any]]:

        assignments = await self.experiment_manager.get_user_experiments(user_id)

        return [
            {
                "experiment_id": assignment.experiment_id,
                "variant": assignment.variant,
                "timestamp": assignment.timestamp.isoformat(),
                "session_id": assignment.session_id
            }
            for assignment in assignments
        ]

    async def list_experiments(
        self,
        status_filter: Optional[ExperimentStatus] = None
    ) -> List[Dict[str, Any]]:

        experiments = []

        for experiment in self.experiment_manager.active_experiments.values():
            if status_filter is None or experiment.status == status_filter:
                experiment_info = await self.experiment_manager.get_experiment_stats(experiment.id)
                experiments.append(experiment_info)

        return experiments

    def health_check(self) -> Dict[str, Any]:

        experiment_manager_health = self.experiment_manager.health_check()
        metrics_tracker_health = self.metrics_tracker.health_check()
        statistical_analyzer_health = self.statistical_analyzer.health_check()

        all_healthy = all([
            experiment_manager_health["status"] == "healthy",
            metrics_tracker_health["status"] == "healthy",
            statistical_analyzer_health["status"] == "healthy"
        ])

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "service_running": self.is_running,
            "components": {
                "experiment_manager": experiment_manager_health,
                "metrics_tracker": metrics_tracker_health,
                "statistical_analyzer": statistical_analyzer_health
            }
        }

    def get_service_metrics(self) -> Dict[str, Any]:

        return {
            "experiment_manager": self.experiment_manager.get_metrics(),
            "metrics_tracker": self.metrics_tracker.get_metrics(),
            "service_status": {
                "is_running": self.is_running,
                "config_loaded": self.config is not None
            }
        }
