import math
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available - using basic implementations")

from .config import ABTestConfig, DEFAULT_AB_TEST_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class BanditArm:
    name: str
    total_pulls: int = 0
    total_reward: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    estimated_reward: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def average_reward(self) -> float:

        return self.total_reward / max(1, self.total_pulls)

    @property
    def success_rate(self) -> float:

        total_trials = self.success_count + self.failure_count
        return self.success_count / max(1, total_trials)

@dataclass
class BanditConfiguration:
    algorithm: str = "thompson_sampling"
    exploration_rate: float = 0.1
    update_frequency_minutes: int = 60
    minimum_samples_per_arm: int = 100
    confidence_level: float = 0.95
    temperature: float = 1.0  # For softmax allocation
    decay_factor: float = 0.99  # For UCB algorithms

class BanditAlgorithm(ABC):
    def __init__(self, config: BanditConfiguration):
        self.config = config
        self.arms: Dict[str, BanditArm] = {}

    @abstractmethod
    def select_arm(self) -> str:

        pass

    @abstractmethod
    def update_arm(self, arm_name: str, reward: float):
        pass

    def add_arm(self, arm_name: str):
        if arm_name not in self.arms:
            self.arms[arm_name] = BanditArm(name=arm_name)

    def get_arm_statistics(self) -> Dict[str, Dict[str, Any]]:

        return {
            name: {
                "total_pulls": arm.total_pulls,
                "average_reward": arm.average_reward,
                "success_rate": arm.success_rate,
                "estimated_reward": arm.estimated_reward,
                "confidence_interval": arm.confidence_interval
            }
            for name, arm in self.arms.items()
        }

class EpsilonGreedyBandit(BanditAlgorithm):
    def select_arm(self) -> str:

        if not self.arms:
            raise ValueError("No arms available")

        if random.random() < self.config.exploration_rate:
            return random.choice(list(self.arms.keys()))

        best_arm = max(self.arms.items(), key=lambda x: x[1].average_reward)
        return best_arm[0]

    def update_arm(self, arm_name: str, reward: float):
        if arm_name not in self.arms:
            self.add_arm(arm_name)

        arm = self.arms[arm_name]
        arm.total_pulls += 1
        arm.total_reward += reward
        arm.last_updated = datetime.now()

        if reward == 0.0:
            arm.failure_count += 1
        elif reward == 1.0:
            arm.success_count += 1

class UCB1Bandit(BanditAlgorithm):
    def select_arm(self) -> str:

        if not self.arms:
            raise ValueError("No arms available")

        total_pulls = sum(arm.total_pulls for arm in self.arms.values())

        for name, arm in self.arms.items():
            if arm.total_pulls == 0:
                return name

        ucb_values = {}
        for name, arm in self.arms.items():
            confidence_bonus = math.sqrt(
                (2 * math.log(total_pulls)) / arm.total_pulls
            )
            ucb_values[name] = arm.average_reward + confidence_bonus

        best_arm = max(ucb_values.items(), key=lambda x: x[1])
        return best_arm[0]

    def update_arm(self, arm_name: str, reward: float):
        if arm_name not in self.arms:
            self.add_arm(arm_name)

        arm = self.arms[arm_name]
        arm.total_pulls += 1
        arm.total_reward += reward
        arm.last_updated = datetime.now()

        if reward == 0.0:
            arm.failure_count += 1
        elif reward == 1.0:
            arm.success_count += 1

class ThompsonSamplingBandit(BanditAlgorithm):
    def select_arm(self) -> str:

        if not self.arms:
            raise ValueError("No arms available")

        samples = {}
        for name, arm in self.arms.items():
            alpha = arm.success_count + 1
            beta = arm.failure_count + 1

            if NUMPY_AVAILABLE:
                sample = np.random.beta(alpha, beta)
            else:
                sample = self._sample_beta_approximation(alpha, beta)

            samples[name] = sample

            arm.estimated_reward = alpha / (alpha + beta)
            arm.confidence_interval = self._calculate_beta_ci(alpha, beta)

        best_arm = max(samples.items(), key=lambda x: x[1])
        return best_arm[0]

    def update_arm(self, arm_name: str, reward: float):
        if arm_name not in self.arms:
            self.add_arm(arm_name)

        arm = self.arms[arm_name]
        arm.total_pulls += 1
        arm.total_reward += reward
        arm.last_updated = datetime.now()

        if reward == 1.0:
            arm.success_count += 1
        elif reward == 0.0:
            arm.failure_count += 1
        else:
            if random.random() < reward:
                arm.success_count += 1
            else:
                arm.failure_count += 1

    def _sample_beta_approximation(self, alpha: float, beta: float) -> float:

        if alpha > 10 and beta > 10:
            mean = alpha / (alpha + beta)
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            std = math.sqrt(variance)

            sample = random.gauss(mean, std)
            return max(0.0, min(1.0, sample))
        else:
            return random.random()

    def _calculate_beta_ci(self, alpha: float, beta: float) -> Tuple[float, float]:

        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std = math.sqrt(variance)

        margin = 1.96 * std
        lower = max(0.0, mean - margin)
        upper = min(1.0, mean + margin)

        return (lower, upper)

class BanditOptimizer:
    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or DEFAULT_AB_TEST_CONFIG
        self.bandit_config = BanditConfiguration(
            algorithm=self.config.bandit_algorithm,
            exploration_rate=self.config.bandit_exploration_rate,
            update_frequency_minutes=self.config.bandit_update_frequency_minutes,
            minimum_samples_per_arm=self.config.bandit_minimum_samples_per_arm
        )

        self.bandits: Dict[str, BanditAlgorithm] = {}

        self.metrics = {
            "bandits_created": 0,
            "arms_pulled": 0,
            "total_regret": 0.0,
            "allocation_updates": 0
        }

        logger.info(f"Bandit optimizer initialized with {self.bandit_config.algorithm}")

    def create_bandit_experiment(
        self,
        experiment_id: str,
        variant_names: List[str],
        algorithm: Optional[str] = None
    ) -> bool:

        try:
            bandit_algorithm = algorithm or self.bandit_config.algorithm

            if bandit_algorithm == "epsilon_greedy":
                bandit = EpsilonGreedyBandit(self.bandit_config)
            elif bandit_algorithm == "ucb1":
                bandit = UCB1Bandit(self.bandit_config)
            elif bandit_algorithm == "thompson_sampling":
                bandit = ThompsonSamplingBandit(self.bandit_config)
            else:
                raise ValueError(f"Unknown bandit algorithm: {bandit_algorithm}")

            for variant_name in variant_names:
                bandit.add_arm(variant_name)

            self.bandits[experiment_id] = bandit
            self.metrics["bandits_created"] += 1

            logger.info(f"Bandit experiment created: {experiment_id} with {len(variant_names)} arms")
            return True

        except Exception as e:
            logger.error(f"Error creating bandit experiment: {e}")
            return False

    def select_variant(self, experiment_id: str) -> Optional[str]:

        if experiment_id not in self.bandits:
            logger.warning(f"Bandit not found for experiment: {experiment_id}")
            return None

        try:
            bandit = self.bandits[experiment_id]
            selected_arm = bandit.select_arm()

            self.metrics["arms_pulled"] += 1
            logger.debug(f"Bandit selected arm: {selected_arm} for experiment {experiment_id}")

            return selected_arm

        except Exception as e:
            logger.error(f"Error selecting variant: {e}")
            return None

    def update_bandit(
        self,
        experiment_id: str,
        variant: str,
        reward: float
    ) -> bool:

        if experiment_id not in self.bandits:
            logger.warning(f"Bandit not found for experiment: {experiment_id}")
            return False

        try:
            bandit = self.bandits[experiment_id]
            bandit.update_arm(variant, reward)

            best_arm = max(bandit.arms.values(), key=lambda x: x.average_reward)
            current_regret = best_arm.average_reward - bandit.arms[variant].average_reward
            self.metrics["total_regret"] += max(0, current_regret)

            logger.debug(f"Bandit updated: {variant} with reward {reward}")
            return True

        except Exception as e:
            logger.error(f"Error updating bandit: {e}")
            return False

    def get_traffic_allocation(self, experiment_id: str) -> Dict[str, float]:

        if experiment_id not in self.bandits:
            return {}

        try:
            bandit = self.bandits[experiment_id]

            if self.bandit_config.algorithm == "thompson_sampling":
                return self._thompson_allocation(bandit)
            elif self.bandit_config.algorithm == "ucb1":
                return self._ucb_allocation(bandit)
            else:
                return self._epsilon_greedy_allocation(bandit)

        except Exception as e:
            logger.error(f"Error calculating traffic allocation: {e}")
            return {}

    def _thompson_allocation(self, bandit: BanditAlgorithm) -> Dict[str, float]:

        num_samples = 1000
        selection_counts = {name: 0 for name in bandit.arms.keys()}

        for _ in range(num_samples):
            if isinstance(bandit, ThompsonSamplingBandit):
                selected = bandit.select_arm()
                selection_counts[selected] += 1

        total_selections = sum(selection_counts.values())
        return {
            name: (count / total_selections) * 100
            for name, count in selection_counts.items()
        }

    def _ucb_allocation(self, bandit: BanditAlgorithm) -> Dict[str, float]:

        if not isinstance(bandit, UCB1Bandit):
            return self._equal_allocation(bandit)

        total_pulls = sum(arm.total_pulls for arm in bandit.arms.values())

        ucb_values = {}
        for name, arm in bandit.arms.items():
            if arm.total_pulls == 0:
                ucb_values[name] = float('inf')
            else:
                confidence_bonus = math.sqrt((2 * math.log(total_pulls)) / arm.total_pulls)
                ucb_values[name] = arm.average_reward + confidence_bonus

        return self._softmax_allocation(ucb_values)

    def _epsilon_greedy_allocation(self, bandit: BanditAlgorithm) -> Dict[str, float]:

        if not bandit.arms:
            return {}

        num_arms = len(bandit.arms)
        exploration_per_arm = (self.bandit_config.exploration_rate / num_arms) * 100

        best_arm = max(bandit.arms.items(), key=lambda x: x[1].average_reward)
        best_arm_name = best_arm[0]

        allocation = {}
        for name in bandit.arms.keys():
            if name == best_arm_name:
                allocation[name] = (1 - self.bandit_config.exploration_rate) * 100 + exploration_per_arm
            else:
                allocation[name] = exploration_per_arm

        return allocation

    def _softmax_allocation(self, values: Dict[str, float]) -> Dict[str, float]:

        finite_values = {k: v for k, v in values.items() if v != float('inf')}
        infinite_keys = [k for k, v in values.items() if v == float('inf')]

        if infinite_keys:
            equal_share = 100.0 / len(infinite_keys)
            allocation = {k: 0.0 for k in finite_values.keys()}
            allocation.update({k: equal_share for k in infinite_keys})
            return allocation

        temperature = self.bandit_config.temperature
        exp_values = {k: math.exp(v / temperature) for k, v in finite_values.items()}
        sum_exp = sum(exp_values.values())

        return {k: (exp_v / sum_exp) * 100 for k, exp_v in exp_values.items()}

    def _equal_allocation(self, bandit: BanditAlgorithm) -> Dict[str, float]:

        if not bandit.arms:
            return {}

        equal_share = 100.0 / len(bandit.arms)
        return {name: equal_share for name in bandit.arms.keys()}

    def get_bandit_performance(self, experiment_id: str) -> Dict[str, Any]:

        if experiment_id not in self.bandits:
            return {}

        bandit = self.bandits[experiment_id]
        arm_stats = bandit.get_arm_statistics()

        total_pulls = sum(arm.total_pulls for arm in bandit.arms.values())
        total_reward = sum(arm.total_reward for arm in bandit.arms.values())

        best_arm = max(bandit.arms.items(), key=lambda x: x[1].average_reward)

        return {
            "experiment_id": experiment_id,
            "algorithm": self.bandit_config.algorithm,
            "total_pulls": total_pulls,
            "average_reward": total_reward / max(1, total_pulls),
            "best_arm": {
                "name": best_arm[0],
                "average_reward": best_arm[1].average_reward,
                "confidence_interval": best_arm[1].confidence_interval
            },
            "arm_statistics": arm_stats,
            "traffic_allocation": self.get_traffic_allocation(experiment_id)
        }

    def get_metrics(self) -> Dict[str, Any]:

        return {
            "bandits_created": self.metrics["bandits_created"],
            "arms_pulled": self.metrics["arms_pulled"],
            "total_regret": self.metrics["total_regret"],
            "allocation_updates": self.metrics["allocation_updates"],
            "active_bandits": len(self.bandits),
            "algorithms_supported": [
                "epsilon_greedy",
                "ucb1",
                "thompson_sampling"
            ]
        }

    def health_check(self) -> Dict[str, Any]:

        return {
            "status": "healthy",
            "numpy_available": NUMPY_AVAILABLE,
            "active_bandits": len(self.bandits),
            "metrics": self.get_metrics()
        }
