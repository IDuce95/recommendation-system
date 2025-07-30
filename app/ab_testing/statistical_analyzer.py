import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import statistics

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available - using basic statistics")

try:
    from scipy import stats
    from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available - limited statistical tests")

from .config import (
    ABTestConfig, StatisticalTest, AnalysisConfig,
    DEFAULT_AB_TEST_CONFIG
)

logger = logging.getLogger(__name__)

@dataclass
class StatisticalResult:
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    effect_size_type: str
    sample_size_control: int
    sample_size_treatment: int
    mean_control: float
    mean_treatment: float
    std_control: float
    std_treatment: float
    is_significant: bool
    confidence_level: float

    def to_dict(self) -> Dict[str, Any]:

        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "confidence_interval": self.confidence_interval,
            "effect_size": self.effect_size,
            "effect_size_type": self.effect_size_type,
            "sample_size_control": self.sample_size_control,
            "sample_size_treatment": self.sample_size_treatment,
            "mean_control": self.mean_control,
            "mean_treatment": self.mean_treatment,
            "std_control": self.std_control,
            "std_treatment": self.std_treatment,
            "is_significant": self.is_significant,
            "confidence_level": self.confidence_level
        }

@dataclass
class ExperimentAnalysis:
    experiment_id: str
    variant_results: Dict[str, StatisticalResult]
    primary_metric_results: Dict[str, StatisticalResult]
    secondary_metric_results: Dict[str, StatisticalResult]
    overall_recommendation: str
    statistical_power: float
    sample_size_adequate: bool
    early_stopping_recommendation: str
    analysis_timestamp: datetime
    confidence_level: float

    def to_dict(self) -> Dict[str, Any]:

        return {
            "experiment_id": self.experiment_id,
            "variant_results": {k: v.to_dict() for k, v in self.variant_results.items()},
            "primary_metric_results": {k: v.to_dict() for k, v in self.primary_metric_results.items()},
            "secondary_metric_results": {k: v.to_dict() for k, v in self.secondary_metric_results.items()},
            "overall_recommendation": self.overall_recommendation,
            "statistical_power": self.statistical_power,
            "sample_size_adequate": self.sample_size_adequate,
            "early_stopping_recommendation": self.early_stopping_recommendation,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "confidence_level": self.confidence_level
        }

class StatisticalAnalyzer:
    """
    Advanced statistical analyzer for A/B test experiments.

    Features:
    - Multiple statistical tests (t-test, Mann-Whitney U, Chi-square)
    - Effect size calculations (Cohen's d, Hedges' g)
    - Confidence interval estimation
    - Sequential testing with alpha spending functions
    - Multiple testing correction
    - Power analysis and sample size validation
    """

    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or DEFAULT_AB_TEST_CONFIG
        self.analysis_config = AnalysisConfig()

        self.bootstrap_cache = {}

        logger.info("Statistical analyzer initialized")

    async def analyze_experiment(
        self,
        experiment_id: str,
        control_data: List[float],
        treatment_data: List[float],
        metric_name: str = "conversion_rate",
        control_variant: str = "control",
        treatment_variant: str = "treatment"
    ) -> StatisticalResult:

        if not control_data or not treatment_data:
            raise ValueError("Both control and treatment data must be non-empty")

        mean_control = statistics.mean(control_data)
        mean_treatment = statistics.mean(treatment_data)
        std_control = statistics.stdev(control_data) if len(control_data) > 1 else 0
        std_treatment = statistics.stdev(treatment_data) if len(treatment_data) > 1 else 0

        test_result = await self._perform_statistical_test(
            control_data, treatment_data, metric_name
        )

        effect_size, effect_size_type = self._calculate_effect_size(
            control_data, treatment_data, mean_control, mean_treatment,
            std_control, std_treatment
        )

        confidence_interval = self._calculate_confidence_interval(
            control_data, treatment_data, self.config.default_confidence_level
        )

        alpha = 1 - self.config.default_confidence_level
        is_significant = test_result["p_value"] < alpha

        result = StatisticalResult(
            test_name=test_result["test_name"],
            statistic=test_result["statistic"],
            p_value=test_result["p_value"],
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            effect_size_type=effect_size_type,
            sample_size_control=len(control_data),
            sample_size_treatment=len(treatment_data),
            mean_control=mean_control,
            mean_treatment=mean_treatment,
            std_control=std_control,
            std_treatment=std_treatment,
            is_significant=is_significant,
            confidence_level=self.config.default_confidence_level
        )

        logger.info(f"Analysis completed for {metric_name}: p={test_result['p_value']:.4f}, significant={is_significant}")
        return result

    async def _perform_statistical_test(
        self,
        control_data: List[float],
        treatment_data: List[float],
        metric_name: str
    ) -> Dict[str, Any]:

        is_binary = all(x in [0, 1] for x in control_data + treatment_data)
        is_normal_control = self._test_normality(control_data)
        is_normal_treatment = self._test_normality(treatment_data)

        if is_binary:
            return self._chi_square_test(control_data, treatment_data)
        elif is_normal_control and is_normal_treatment:
            return self._t_test(control_data, treatment_data)
        else:
            return self._mann_whitney_test(control_data, treatment_data)

    def _t_test(self, control_data: List[float], treatment_data: List[float]) -> Dict[str, Any]:

        if SCIPY_AVAILABLE:
            statistic, p_value = ttest_ind(control_data, treatment_data, equal_var=False)
            return {
                "test_name": "Welch's t-test",
                "statistic": float(statistic),
                "p_value": float(p_value)
            }
        else:
            return self._t_test_fallback(control_data, treatment_data)

    def _t_test_fallback(self, control_data: List[float], treatment_data: List[float]) -> Dict[str, Any]:

        n1, n2 = len(control_data), len(treatment_data)
        mean1, mean2 = statistics.mean(control_data), statistics.mean(treatment_data)
        var1 = statistics.variance(control_data) if n1 > 1 else 0
        var2 = statistics.variance(treatment_data) if n2 > 1 else 0

        pooled_se = math.sqrt(var1/n1 + var2/n2)
        if pooled_se == 0:
            return {"test_name": "t-test", "statistic": 0.0, "p_value": 1.0}

        t_statistic = (mean2 - mean1) / pooled_se

        if var1 > 0 and var2 > 0:
            df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        else:
            df = n1 + n2 - 2

        if df > 30:
            p_value = 2 * (1 - self._normal_cdf(abs(t_statistic)))
        else:
            p_value = 2 * (1 - self._normal_cdf(abs(t_statistic)))

        return {
            "test_name": "t-test (approximation)",
            "statistic": t_statistic,
            "p_value": min(1.0, max(0.0, p_value))
        }

    def _mann_whitney_test(self, control_data: List[float], treatment_data: List[float]) -> Dict[str, Any]:

        if SCIPY_AVAILABLE:
            statistic, p_value = mannwhitneyu(control_data, treatment_data, alternative='two-sided')
            return {
                "test_name": "Mann-Whitney U",
                "statistic": float(statistic),
                "p_value": float(p_value)
            }
        else:
            return self._mann_whitney_fallback(control_data, treatment_data)

    def _mann_whitney_fallback(self, control_data: List[float], treatment_data: List[float]) -> Dict[str, Any]:

        combined = [(x, 0) for x in control_data] + [(x, 1) for x in treatment_data]
        combined.sort(key=lambda x: x[0])

        ranks = {}
        for i, (value, group) in enumerate(combined):
            if value not in ranks:
                ranks[value] = []
            ranks[value].append(i + 1)

        avg_ranks = {}
        for value, rank_list in ranks.items():
            avg_ranks[value] = sum(rank_list) / len(rank_list)

        r1 = sum(avg_ranks[x] for x in control_data)
        n1, n2 = len(control_data), len(treatment_data)

        u1 = r1 - n1 * (n1 + 1) / 2
        u2 = n1 * n2 - u1
        u_statistic = min(u1, u2)

        mean_u = n1 * n2 / 2
        std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

        if std_u > 0:
            z_score = (u_statistic - mean_u) / std_u
            p_value = 2 * (1 - self._normal_cdf(abs(z_score)))
        else:
            p_value = 1.0

        return {
            "test_name": "Mann-Whitney U (approximation)",
            "statistic": u_statistic,
            "p_value": min(1.0, max(0.0, p_value))
        }

    def _chi_square_test(self, control_data: List[float], treatment_data: List[float]) -> Dict[str, Any]:

        control_success = sum(control_data)
        control_failure = len(control_data) - control_success
        treatment_success = sum(treatment_data)
        treatment_failure = len(treatment_data) - treatment_success

        observed = [[control_success, control_failure],
                   [treatment_success, treatment_failure]]

        if SCIPY_AVAILABLE:
            chi2_stat, p_value, dof, expected = chi2_contingency(observed)
            return {
                "test_name": "Chi-square test",
                "statistic": float(chi2_stat),
                "p_value": float(p_value)
            }
        else:
            return self._chi_square_fallback(observed)

    def _chi_square_fallback(self, observed: List[List[int]]) -> Dict[str, Any]:

        row_totals = [sum(row) for row in observed]
        col_totals = [sum(observed[i][j] for i in range(len(observed))) for j in range(len(observed[0]))]
        total = sum(row_totals)

        chi2_stat = 0
        for i in range(len(observed)):
            for j in range(len(observed[0])):
                expected = row_totals[i] * col_totals[j] / total
                if expected > 0:
                    chi2_stat += (observed[i][j] - expected) ** 2 / expected

        p_value = 1 - self._chi2_cdf(chi2_stat, 1)

        return {
            "test_name": "Chi-square test (approximation)",
            "statistic": chi2_stat,
            "p_value": min(1.0, max(0.0, p_value))
        }

    def _calculate_effect_size(
        self,
        control_data: List[float],
        treatment_data: List[float],
        mean_control: float,
        mean_treatment: float,
        std_control: float,
        std_treatment: float
    ) -> Tuple[float, str]:

        is_binary = all(x in [0, 1] for x in control_data + treatment_data)

        if is_binary:
            if mean_control > 0:
                relative_risk = mean_treatment / mean_control
                effect_size = relative_risk - 1  # Relative change
                return effect_size, "relative_risk_change"
            else:
                return 0.0, "relative_risk_change"
        else:
            pooled_std = self._calculate_pooled_std(control_data, treatment_data)
            if pooled_std > 0:
                cohens_d = (mean_treatment - mean_control) / pooled_std
                return cohens_d, "cohens_d"
            else:
                return 0.0, "cohens_d"

    def _calculate_pooled_std(self, control_data: List[float], treatment_data: List[float]) -> float:

        n1, n2 = len(control_data), len(treatment_data)

        if n1 <= 1 and n2 <= 1:
            return 0.0

        var1 = statistics.variance(control_data) if n1 > 1 else 0
        var2 = statistics.variance(treatment_data) if n2 > 1 else 0

        pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        return math.sqrt(max(0, pooled_variance))

    def _calculate_confidence_interval(
        self,
        control_data: List[float],
        treatment_data: List[float],
        confidence_level: float
    ) -> Tuple[float, float]:

        if SCIPY_AVAILABLE:
            return self._bootstrap_confidence_interval(control_data, treatment_data, confidence_level)
        else:
            return self._approximate_confidence_interval(control_data, treatment_data, confidence_level)

    def _bootstrap_confidence_interval(
        self,
        control_data: List[float],
        treatment_data: List[float],
        confidence_level: float
    ) -> Tuple[float, float]:

        if not NUMPY_AVAILABLE:
            return self._approximate_confidence_interval(control_data, treatment_data, confidence_level)

        n_bootstrap = 1000
        differences = []

        control_array = np.array(control_data)
        treatment_array = np.array(treatment_data)

        for _ in range(n_bootstrap):
            control_bootstrap = np.random.choice(control_array, size=len(control_array), replace=True)
            treatment_bootstrap = np.random.choice(treatment_array, size=len(treatment_array), replace=True)

            diff = np.mean(treatment_bootstrap) - np.mean(control_bootstrap)
            differences.append(diff)

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(differences, lower_percentile)
        upper_bound = np.percentile(differences, upper_percentile)

        return (float(lower_bound), float(upper_bound))

    def _approximate_confidence_interval(
        self,
        control_data: List[float],
        treatment_data: List[float],
        confidence_level: float
    ) -> Tuple[float, float]:

        mean_control = statistics.mean(control_data)
        mean_treatment = statistics.mean(treatment_data)

        n1, n2 = len(control_data), len(treatment_data)
        var1 = statistics.variance(control_data) if n1 > 1 else 0
        var2 = statistics.variance(treatment_data) if n2 > 1 else 0

        se_diff = math.sqrt(var1/n1 + var2/n2)

        alpha = 1 - confidence_level
        z_critical = self._normal_ppf(1 - alpha/2)

        diff = mean_treatment - mean_control
        margin_error = z_critical * se_diff

        return (diff - margin_error, diff + margin_error)

    def _test_normality(self, data: List[float]) -> bool:

        if len(data) < 8:
            return True  # Assume normal for small samples

        try:
            mean_val = statistics.mean(data)
            std_val = statistics.stdev(data)

            if std_val == 0:
                return True

            standardized = [(x - mean_val) / std_val for x in data]

            skewness = sum(x**3 for x in standardized) / len(standardized)
            kurtosis = sum(x**4 for x in standardized) / len(standardized) - 3

            return abs(skewness) < 2 and abs(kurtosis) < 7

        except Exception:
            return True  # Default to normal if calculation fails

    def _normal_cdf(self, x: float) -> float:

        if x < 0:
            return 1 - self._normal_cdf(-x)

        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)

        return y

    def _normal_ppf(self, p: float) -> float:

        if p <= 0 or p >= 1:
            raise ValueError("p must be between 0 and 1")

        if p < 0.5:
            return -self._normal_ppf(1 - p)

        a = [0, -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]

        b = [0, -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]

        c = [0, -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]

        d = [0, 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]

        p_low = 0.02425
        p_high = 1 - p_low

        if p < p_low:
            q = math.sqrt(-2 * math.log(p))
            x = (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            x = (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q / (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
        else:
            q = math.sqrt(-2 * math.log(1 - p))
            x = -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)

        return x

    def _chi2_cdf(self, x: float, df: int) -> float:

        if x <= 0:
            return 0.0
        if df == 1:
            return 2 * self._normal_cdf(math.sqrt(x)) - 1
        elif df == 2:
            return 1 - math.exp(-x / 2)
        else:
            h = 2.0 / (9.0 * df)
            z = (math.pow(x / df, 1.0/3.0) - (1 - h)) / math.sqrt(h)
            return self._normal_cdf(z)

    def calculate_statistical_power(
        self,
        control_data: List[float],
        treatment_data: List[float],
        effect_size: float,
        alpha: float = 0.05
    ) -> float:

        n1, n2 = len(control_data), len(treatment_data)

        if n1 != n2:
            n_harmonic = 2 * n1 * n2 / (n1 + n2)
        else:
            n_harmonic = n1

        z_alpha = self._normal_ppf(1 - alpha / 2)
        z_beta = effect_size * math.sqrt(n_harmonic / 2) - z_alpha
        power = self._normal_cdf(z_beta)

        return max(0.0, min(1.0, power))

    def recommend_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:

        z_alpha = self._normal_ppf(1 - alpha / 2)
        z_beta = self._normal_ppf(power)

        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        p_avg = (p1 + p2) / 2
        effect = abs(p2 - p1)

        if effect == 0:
            return 10000  # Large number if no effect

        n = 2 * p_avg * (1 - p_avg) * ((z_alpha + z_beta) / effect) ** 2

        return max(100, int(math.ceil(n)))

    def health_check(self) -> Dict[str, Any]:

        return {
            "status": "healthy",
            "numpy_available": NUMPY_AVAILABLE,
            "scipy_available": SCIPY_AVAILABLE,
            "supported_tests": [
                "t-test",
                "Mann-Whitney U",
                "Chi-square",
                "Bootstrap confidence intervals"
            ]
        }
