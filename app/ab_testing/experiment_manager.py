import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib
import random

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using in-memory cache")

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    logging.warning("SQLite not available")

from .config import (
    ABTestConfig, ExperimentConfig, VariantConfig, ExperimentType,
    ExperimentStatus, DEFAULT_AB_TEST_CONFIG, validate_experiment_config
)

logger = logging.getLogger(__name__)

@dataclass
class ExperimentAssignment:
    experiment_id: str
    user_id: str
    variant: str
    timestamp: datetime
    session_id: Optional[str] = None
    user_attributes: Optional[Dict[str, Any]] = None

@dataclass
class Experiment:
    id: str
    config: ExperimentConfig
    status: ExperimentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    creator_id: str = "system"
    total_users: int = 0
    variant_assignments: Dict[str, int] = None

    def __post_init__(self):
        if self.variant_assignments is None:
            self.variant_assignments = {variant: 0 for variant in self.config.variants}

class ExperimentManager:
    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or DEFAULT_AB_TEST_CONFIG

        self.redis_client = None
        self.sqlite_conn = None
        self._initialize_storage()

        self.active_experiments: Dict[str, Experiment] = {}
        self.user_assignments: Dict[str, Dict[str, ExperimentAssignment]] = {}

        self.metrics = {
            "assignments_made": 0,
            "experiments_created": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "assignment_time_ms": []
        }

        asyncio.create_task(self._load_active_experiments())

        logger.info("Experiment manager initialized")

    def _initialize_storage(self):
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(
                    self.config.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None

        if SQLITE_AVAILABLE:
            try:
                self.sqlite_conn = sqlite3.connect(
                    self.config.database_url.replace("sqlite:///", ""),
                    check_same_thread=False
                )
                self._create_tables()
                logger.info("SQLite database initialized")
            except Exception as e:
                logger.warning(f"SQLite initialization failed: {e}")
                self.sqlite_conn = None

    def _create_tables(self):
        if not self.sqlite_conn:
            return

        cursor = self.sqlite_conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                config TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                started_at TIMESTAMP,
                ended_at TIMESTAMP,
                creator_id TEXT,
                total_users INTEGER DEFAULT 0
            )

            CREATE TABLE IF NOT EXISTS user_assignments (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                variant TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                session_id TEXT,
                user_attributes TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )

            CREATE TABLE IF NOT EXISTS experiment_metrics (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                variant TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                user_id TEXT,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_assignments_user_id ON user_assignments(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_assignments_experiment_id ON user_assignments(experiment_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiment_metrics_experiment_id ON experiment_metrics(experiment_id)")

        self.sqlite_conn.commit()
        logger.info("Database tables created successfully")

    async def create_experiment(
        self,
        config: ExperimentConfig,
        creator_id: str = "system"
    ) -> Experiment:

        validation_errors = validate_experiment_config(config)
        if validation_errors:
            raise ValueError(f"Invalid experiment config: {validation_errors}")

        active_count = len([e for e in self.active_experiments.values()
                           if e.status == ExperimentStatus.RUNNING])
        if active_count >= self.config.maximum_concurrent_experiments:
            raise ValueError(f"Maximum concurrent experiments reached ({active_count})")

        experiment_id = str(uuid.uuid4())
        experiment = Experiment(
            id=experiment_id,
            config=config,
            status=ExperimentStatus.DRAFT,
            created_at=datetime.now(),
            creator_id=creator_id
        )

        await self._store_experiment(experiment)

        self.active_experiments[experiment_id] = experiment

        self.metrics["experiments_created"] += 1
        logger.info(f"Experiment created: {config.name} ({experiment_id})")

        return experiment

    async def start_experiment(self, experiment_id: str) -> bool:

        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            logger.error(f"Experiment not found: {experiment_id}")
            return False

        if experiment.status != ExperimentStatus.DRAFT:
            logger.error(f"Cannot start experiment in status: {experiment.status}")
            return False

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now()

        await self._store_experiment(experiment)

        if self.redis_client:
            try:
                experiment_data = {
                    "config": asdict(experiment.config),
                    "status": experiment.status.value,
                    "started_at": experiment.started_at.isoformat()
                }
                self.redis_client.setex(
                    f"experiment:{experiment_id}",
                    3600,  # 1 hour TTL
                    json.dumps(experiment_data)
                )
            except Exception as e:
                logger.warning(f"Failed to cache experiment in Redis: {e}")

        logger.info(f"Experiment started: {experiment.config.name} ({experiment_id})")
        return True

    async def stop_experiment(self, experiment_id: str, reason: str = "completed") -> bool:

        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            logger.error(f"Experiment not found: {experiment_id}")
            return False

        if experiment.status != ExperimentStatus.RUNNING:
            logger.error(f"Cannot stop experiment in status: {experiment.status}")
            return False

        experiment.status = ExperimentStatus.COMPLETED
        experiment.ended_at = datetime.now()

        await self._store_experiment(experiment)

        if self.redis_client:
            try:
                self.redis_client.delete(f"experiment:{experiment_id}")
            except Exception as e:
                logger.warning(f"Failed to remove experiment from Redis: {e}")

        logger.info(f"Experiment stopped: {experiment.config.name} ({experiment_id}) - {reason}")
        return True

    async def assign_user_to_experiment(
        self,
        experiment_id: str,
        user_id: str,
        session_id: Optional[str] = None,
        user_attributes: Optional[Dict[str, Any]] = None
    ) -> Optional[ExperimentAssignment]:

        start_time = time.time()

        try:
            if experiment_id in self.user_assignments.get(user_id, {}):
                assignment = self.user_assignments[user_id][experiment_id]
                self.metrics["cache_hits"] += 1
                return assignment

            experiment = self.active_experiments.get(experiment_id)
            if not experiment or experiment.status != ExperimentStatus.RUNNING:
                return None

            if not self._check_user_eligibility(user_id, experiment.config, user_attributes):
                return None

            variant = self._assign_variant(user_id, experiment.config)

            assignment = ExperimentAssignment(
                experiment_id=experiment_id,
                user_id=user_id,
                variant=variant,
                timestamp=datetime.now(),
                session_id=session_id,
                user_attributes=user_attributes
            )

            await self._store_assignment(assignment)

            if user_id not in self.user_assignments:
                self.user_assignments[user_id] = {}
            self.user_assignments[user_id][experiment_id] = assignment

            experiment.total_users += 1
            experiment.variant_assignments[variant] += 1

            self.metrics["assignments_made"] += 1
            self.metrics["cache_misses"] += 1

            processing_time = (time.time() - start_time) * 1000
            self.metrics["assignment_time_ms"].append(processing_time)

            logger.debug(f"User {user_id} assigned to {variant} in experiment {experiment_id}")
            return assignment

        except Exception as e:
            logger.error(f"Error assigning user to experiment: {e}")
            return None

    def _assign_variant(self, user_id: str, config: ExperimentConfig) -> str:

        hash_input = f"{user_id}:{config.name}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        percentage = (hash_value % 10000) / 100.0

        cumulative_percentage = 0.0
        for variant, allocation in config.traffic_allocation.items():
            cumulative_percentage += allocation
            if percentage < cumulative_percentage:
                return variant

        return config.variants[0]

    def _check_user_eligibility(
        self,
        user_id: str,
        config: ExperimentConfig,
        user_attributes: Optional[Dict[str, Any]]
    ) -> bool:

        if not config.exclusion_criteria:
            return True

        if not user_attributes:
            return True

        for key, excluded_values in config.exclusion_criteria.items():
            if key in user_attributes:
                user_value = user_attributes[key]
                if isinstance(excluded_values, list):
                    if user_value in excluded_values:
                        return False
                else:
                    if user_value == excluded_values:
                        return False

        return True

    async def get_user_experiments(self, user_id: str) -> List[ExperimentAssignment]:

        if user_id in self.user_assignments:
            return list(self.user_assignments[user_id].values())

        if self.sqlite_conn:
            try:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("""
                    SELECT experiment_id, variant, timestamp, session_id, user_attributes
                    FROM user_assignments
                    WHERE user_id = ?
                """, (user_id,))

                assignments = []
                for row in cursor.fetchall():
                    assignment = ExperimentAssignment(
                        experiment_id=row[0],
                        user_id=user_id,
                        variant=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        session_id=row[3],
                        user_attributes=json.loads(row[4]) if row[4] else None
                    )
                    assignments.append(assignment)

                if assignments:
                    self.user_assignments[user_id] = {
                        a.experiment_id: a for a in assignments
                    }

                return assignments

            except Exception as e:
                logger.error(f"Error loading user assignments: {e}")

        return []

    async def get_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:

        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            return {}

        runtime_hours = 0
        if experiment.started_at:
            runtime = datetime.now() - experiment.started_at
            runtime_hours = runtime.total_seconds() / 3600

        total_users = experiment.total_users
        traffic_distribution = {}
        for variant, count in experiment.variant_assignments.items():
            percentage = (count / max(1, total_users)) * 100
            traffic_distribution[variant] = {
                "count": count,
                "percentage": percentage,
                "expected_percentage": experiment.config.traffic_allocation.get(variant, 0)
            }

        return {
            "experiment_id": experiment_id,
            "name": experiment.config.name,
            "status": experiment.status.value,
            "runtime_hours": runtime_hours,
            "total_users": total_users,
            "traffic_distribution": traffic_distribution,
            "variants": experiment.config.variants,
            "target_metrics": [m.value for m in experiment.config.target_metrics],
            "created_at": experiment.created_at.isoformat(),
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "ended_at": experiment.ended_at.isoformat() if experiment.ended_at else None
        }

    async def _store_experiment(self, experiment: Experiment):
        if not self.sqlite_conn:
            return

        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO experiments
                (id, name, config, status, created_at, started_at, ended_at, creator_id, total_users)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.id,
                experiment.config.name,
                json.dumps(asdict(experiment.config)),
                experiment.status.value,
                experiment.created_at.isoformat(),
                experiment.started_at.isoformat() if experiment.started_at else None,
                experiment.ended_at.isoformat() if experiment.ended_at else None,
                experiment.creator_id,
                experiment.total_users
            ))
            self.sqlite_conn.commit()

        except Exception as e:
            logger.error(f"Error storing experiment: {e}")

    async def _store_assignment(self, assignment: ExperimentAssignment):
        if not self.sqlite_conn:
            return

        try:
            cursor = self.sqlite_conn.cursor()
            assignment_id = f"{assignment.experiment_id}:{assignment.user_id}"

            cursor.execute("""
                INSERT OR REPLACE INTO user_assignments
                (id, experiment_id, user_id, variant, timestamp, session_id, user_attributes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                assignment_id,
                assignment.experiment_id,
                assignment.user_id,
                assignment.variant,
                assignment.timestamp.isoformat(),
                assignment.session_id,
                json.dumps(assignment.user_attributes) if assignment.user_attributes else None
            ))
            self.sqlite_conn.commit()

            if self.redis_client:
                self.redis_client.setex(
                    f"assignment:{assignment.user_id}:{assignment.experiment_id}",
                    86400,  # 24 hours TTL
                    json.dumps(asdict(assignment), default=str)
                )

        except Exception as e:
            logger.error(f"Error storing assignment: {e}")

    async def _load_active_experiments(self):
        if not self.sqlite_conn:
            return

        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT id, name, config, status, created_at, started_at, ended_at, creator_id, total_users
                FROM experiments
                WHERE status IN ('running', 'paused')
            """)

            for row in cursor.fetchall():
                config_dict = json.loads(row[2])
                config_dict['experiment_type'] = ExperimentType(config_dict['experiment_type'])
                config_dict['target_metrics'] = [getattr(__import__('app.ab_testing.config', fromlist=['MetricType']), 'MetricType')(m) for m in config_dict['target_metrics']]

                experiment_config = ExperimentConfig(**config_dict)

                experiment = Experiment(
                    id=row[0],
                    config=experiment_config,
                    status=ExperimentStatus(row[3]),
                    created_at=datetime.fromisoformat(row[4]),
                    started_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    ended_at=datetime.fromisoformat(row[6]) if row[6] else None,
                    creator_id=row[7],
                    total_users=row[8]
                )

                self.active_experiments[experiment.id] = experiment

            logger.info(f"Loaded {len(self.active_experiments)} active experiments")

        except Exception as e:
            logger.error(f"Error loading active experiments: {e}")

    def get_metrics(self) -> Dict[str, Any]:

        avg_assignment_time = 0
        if self.metrics["assignment_time_ms"]:
            avg_assignment_time = sum(self.metrics["assignment_time_ms"]) / len(self.metrics["assignment_time_ms"])

        cache_hit_rate = 0
        total_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total_requests > 0:
            cache_hit_rate = self.metrics["cache_hits"] / total_requests

        return {
            "experiments_created": self.metrics["experiments_created"],
            "assignments_made": self.metrics["assignments_made"],
            "active_experiments": len(self.active_experiments),
            "cache_hit_rate": cache_hit_rate,
            "avg_assignment_time_ms": avg_assignment_time,
            "storage_backend": {
                "redis_connected": self.redis_client is not None,
                "sqlite_connected": self.sqlite_conn is not None
            }
        }

    def health_check(self) -> Dict[str, Any]:

        redis_healthy = True
        sqlite_healthy = True

        if self.redis_client:
            try:
                self.redis_client.ping()
            except Exception:
                redis_healthy = False

        if self.sqlite_conn:
            try:
                self.sqlite_conn.execute("SELECT 1")
            except Exception:
                sqlite_healthy = False

        overall_healthy = (redis_healthy or not REDIS_AVAILABLE) and (sqlite_healthy or not SQLITE_AVAILABLE)

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "redis_healthy": redis_healthy,
            "sqlite_healthy": sqlite_healthy,
            "active_experiments": len(self.active_experiments),
            "metrics": self.get_metrics()
        }
