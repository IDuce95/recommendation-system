from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

class ExperimentType(Enum):
    AB_TEST = "ab_test"  # Classic A/B test
    MULTIVARIATE = "multivariate"  # Multiple variants
    MULTI_ARMED_BANDIT = "multi_armed_bandit"  # Dynamic allocation
    SEQUENTIAL = "sequential"  # Sequential testing
    SPLIT_URL = "split_url"  # URL-based splitting

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class MetricType(Enum):
    CONVERSION_RATE = "conversion_rate"
    CLICK_THROUGH_RATE = "ctr"
    REVENUE_PER_USER = "revenue_per_user"
    ENGAGEMENT_TIME = "engagement_time"
    RECOMMENDATION_ACCURACY = "recommendation_accuracy"
    USER_SATISFACTION = "user_satisfaction"
    CHURN_RATE = "churn_rate"
    LIFETIME_VALUE = "lifetime_value"

class StatisticalTest(Enum):
    T_TEST = "t_test"
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    BAYESIAN = "bayesian"

@dataclass
class ExperimentConfig:
    name: str
    description: str
    experiment_type: ExperimentType
    variants: List[str]
    traffic_allocation: Dict[str, float]  # variant -> percentage
    target_metrics: List[MetricType]
    minimum_sample_size: int = 1000
    maximum_duration_days: int = 30
    confidence_level: float = 0.95
    statistical_power: float = 0.8
    minimum_detectable_effect: float = 0.05  # 5% relative change
    early_stopping_enabled: bool = True
    stratification_keys: Optional[List[str]] = None
    exclusion_criteria: Optional[Dict[str, Any]] = None

@dataclass
class ABTestConfig:
    database_url: str = "sqlite:///ab_testing.db"
    redis_url: str = "redis://localhost:6379"

    default_confidence_level: float = 0.95
    default_statistical_power: float = 0.8
    default_minimum_detectable_effect: float = 0.05
    multiple_testing_correction: str = "bonferroni"  # bonferroni, benjamini_hochberg

    default_experiment_duration_days: int = 14
    minimum_sample_size_per_variant: int = 1000
    maximum_concurrent_experiments: int = 10
    traffic_allocation_precision: float = 0.01  # 1% precision

    primary_metrics: List[MetricType] = field(default_factory=lambda: [
        MetricType.CONVERSION_RATE,
        MetricType.CLICK_THROUGH_RATE,
        MetricType.REVENUE_PER_USER
    ])

    secondary_metrics: List[MetricType] = field(default_factory=lambda: [
        MetricType.ENGAGEMENT_TIME,
        MetricType.USER_SATISFACTION,
        MetricType.RECOMMENDATION_ACCURACY
    ])

    bandit_algorithm: str = "thompson_sampling"  # epsilon_greedy, ucb1, thompson_sampling
    bandit_exploration_rate: float = 0.1
    bandit_update_frequency_minutes: int = 60
    bandit_minimum_samples_per_arm: int = 100

    monitoring_enabled: bool = True
    monitoring_interval_minutes: int = 15
    alert_on_significant_changes: bool = True
    alert_email_recipients: List[str] = field(default_factory=list)

    batch_size: int = 1000
    cache_ttl_seconds: int = 300
    max_concurrent_analyses: int = 5

    kafka_enabled: bool = True
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_experiment_topic: str = "ab_test_events"

    experiment_data_retention_days: int = 365
    metrics_data_retention_days: int = 90

    enable_sequential_testing: bool = True
    enable_bayesian_analysis: bool = True
    enable_causal_inference: bool = False
    enable_uplift_modeling: bool = False

DEFAULT_AB_TEST_CONFIG = ABTestConfig()

@dataclass
class VariantConfig:
    name: str
    description: str
    traffic_percentage: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_control: bool = False

    def __post_init__(self):
        if not 0 <= self.traffic_percentage <= 100:
            raise ValueError("Traffic percentage must be between 0 and 100")

@dataclass
class MetricConfig:
    metric_type: MetricType
    name: str
    description: str
    calculation_method: str  # SQL query or Python function
    aggregation_period: str = "daily"  # hourly, daily, weekly
    is_primary: bool = False
    higher_is_better: bool = True
    minimum_sample_size: int = 100
    outlier_detection_enabled: bool = True
    outlier_threshold_std: float = 3.0

@dataclass
class AnalysisConfig:
    statistical_tests: List[StatisticalTest] = field(default_factory=lambda: [
        StatisticalTest.T_TEST,
        StatisticalTest.MANN_WHITNEY_U
    ])

    confidence_levels: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])

    sequential_testing_enabled: bool = True
    sequential_alpha_spending_function: str = "obrien_fleming"
    sequential_max_analyses: int = 5

    bayesian_prior_type: str = "uninformative"  # uninformative, informative, empirical
    bayesian_credible_interval: float = 0.95
    bayesian_rope_threshold: float = 0.01  # Region of Practical Equivalence

    correction_method: str = "benjamini_hochberg"
    family_wise_error_rate: float = 0.05

    bootstrap_samples: int = 10000
    bootstrap_confidence_level: float = 0.95

    calculate_effect_sizes: bool = True
    effect_size_measures: List[str] = field(default_factory=lambda: [
        "cohen_d", "hedges_g", "cliff_delta"
    ])

@dataclass
class MonitoringConfig:
    real_time_monitoring: bool = True
    monitoring_frequency_minutes: int = 15

    sample_ratio_mismatch_threshold: float = 0.05  # 5% deviation
    conversion_rate_anomaly_threshold: float = 0.2  # 20% change
    p_value_threshold: float = 0.001  # Very significant results

    data_quality_checks: bool = True
    missing_data_threshold: float = 0.1  # 10% missing data triggers alert
    duplicate_user_threshold: float = 0.01  # 1% duplicate users

    analysis_time_threshold_seconds: float = 30.0
    memory_usage_threshold_mb: float = 1000.0

    slack_webhook_url: Optional[str] = None
    email_notifications: bool = True
    sms_notifications: bool = False

def validate_traffic_allocation(variants: List[VariantConfig]) -> bool:

    total_traffic = sum(variant.traffic_percentage for variant in variants)
    return abs(total_traffic - 100.0) < 0.01

def validate_experiment_config(config: ExperimentConfig) -> List[str]:

    errors = []

    total_allocation = sum(config.traffic_allocation.values())
    if abs(total_allocation - 100.0) > 0.01:
        errors.append(f"Traffic allocation sums to {total_allocation}%, must be 100%")

    for variant in config.variants:
        if variant not in config.traffic_allocation:
            errors.append(f"Variant '{variant}' missing from traffic allocation")

    if config.minimum_sample_size < 100:
        errors.append("Minimum sample size must be at least 100")

    if not 0.5 <= config.confidence_level <= 0.99:
        errors.append("Confidence level must be between 0.5 and 0.99")

    if not 0.5 <= config.statistical_power <= 0.99:
        errors.append("Statistical power must be between 0.5 and 0.99")

    return errors

def get_sample_size_estimate(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8
) -> int:

    import math
    from scipy import stats

    effect_size = baseline_rate * minimum_detectable_effect

    p_pooled = baseline_rate
    pooled_std = math.sqrt(2 * p_pooled * (1 - p_pooled))

    d = effect_size / pooled_std

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = ((z_alpha + z_beta) / d) ** 2

    return max(100, int(math.ceil(n)))  # Minimum 100 samples
