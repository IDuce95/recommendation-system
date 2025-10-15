import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.feature_store import FeatureStore
from app.recommendation.recommender import RecommendationEngine
from app.ab_testing import ABTestingSystem
from config.ml_system_config import MLSystemConfig

logger = logging.getLogger(__name__)

class MLSystemOrchestrator:
    def __init__(self, config: MLSystemConfig):
        self.config = config
        self.feature_store: Optional[FeatureStore] = None
        self.recommendation_engine: Optional[RecommendationEngine] = None
        self.ab_testing_system: Optional[ABTestingSystem] = None
        self.is_running = False
        self.background_tasks: List[asyncio.Task] = []

    async def initialize(self) -> bool:
        try:
            logger.info("Initializing ML System Orchestrator...")
            
            await self._initialize_feature_store()
            await self._initialize_recommendation_engine()
            await self._initialize_ab_testing()
            
            self.is_running = True
            logger.info("ML System Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            return False

    async def _initialize_feature_store(self):
        logger.info("Initializing Feature Store...")
        self.feature_store = FeatureStore(self.config.feature_store)
        await self.feature_store.initialize()

    async def _initialize_recommendation_engine(self):
        logger.info("Initializing Recommendation Engine...")
        self.recommendation_engine = RecommendationEngine(
            feature_store=self.feature_store,
            config=self.config.recommendation
        )
        await self.recommendation_engine.initialize()

    async def _initialize_ab_testing(self):
        logger.info("Initializing A/B Testing System...")
        self.ab_testing_system = ABTestingSystem(self.config.ab_testing)
        await self.ab_testing_system.initialize()

    async def start_background_tasks(self):
        health_task = asyncio.create_task(self._health_monitoring())
        self.background_tasks.append(health_task)
        
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self.background_tasks.append(cleanup_task)

    async def _health_monitoring(self):
        while self.is_running:
            try:
                await asyncio.sleep(300)
                await self._check_system_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    async def _periodic_cleanup(self):
        while self.is_running:
            try:
                await asyncio.sleep(3600)
                if self.feature_store:
                    await self.feature_store.cleanup_expired_features()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _check_system_health(self):
        health_status = {}
        
        if self.feature_store:
            health_status["feature_store"] = await self.feature_store.health_check()
        
        if self.recommendation_engine:
            health_status["recommendation_engine"] = await self.recommendation_engine.health_check()
        
        if self.ab_testing_system:
            health_status["ab_testing"] = await self.ab_testing_system.health_check()
        
        return health_status

    async def shutdown(self):
        logger.info("Shutting down ML System Orchestrator...")
        self.is_running = False
        
        for task in self.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        if self.ab_testing_system:
            await self.ab_testing_system.shutdown()
        
        if self.feature_store:
            await self.feature_store.cleanup()
        
        logger.info("ML System Orchestrator shutdown complete")
