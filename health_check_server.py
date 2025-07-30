from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from ml_system import MLRecommendationSystem
from config.ml_system_config import MLSystemConfig

logger = logging.getLogger(__name__)

ml_system: MLRecommendationSystem = None

def create_health_app() -> FastAPI:

    app = FastAPI(
        title="ML Recommendation System - Health Check",
        description="Health checking and system monitoring for the ML recommendation system",
        version="1.0.0"
    )

    @app.on_event("startup")
    async def startup_event():
        global ml_system
        try:
            config = MLSystemConfig.from_environment()
            ml_system = MLRecommendationSystem()
            await ml_system.initialize()
            logger.info("ML Recommendation System initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ML system: {e}")
            raise

    @app.on_event("shutdown")
    async def shutdown_event():
        global ml_system
        if ml_system:
            await ml_system.shutdown()
            logger.info("ML Recommendation System shutdown complete")

    @app.get("/health", response_model=Dict[str, Any])
    async def health_check():
        try:
            if not ml_system or not ml_system.is_initialized:
                raise HTTPException(
                    status_code=503,
                    detail="ML Recommendation System not initialized"
                )

            health_result = await ml_system.health_check()
            overall_status = health_result.get("system", {}).get("status", "unknown")

            status_code = 200 if overall_status == "healthy" else 503

            return JSONResponse(
                status_code=status_code,
                content={
                    "status": overall_status,
                    "timestamp": datetime.now().isoformat(),
                    "components": health_result
                }
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Health check error: {str(e)}"
            )

    @app.get("/status", response_model=Dict[str, Any])
    async def system_status():
        try:
            if not ml_system or not ml_system.is_initialized:
                raise HTTPException(
                    status_code=503,
                    detail="ML Recommendation System not initialized"
                )

            status = await ml_system.get_system_status()

            return {
                "system_status": {
                    "overall_healthy": status.overall_healthy,
                    "uptime_seconds": status.system_uptime,
                    "total_requests": status.total_requests,
                    "active_experiments": status.active_experiments,
                    "last_updated": status.last_updated.isoformat()
                },
                "component_health": {
                    "feature_store": status.feature_store_healthy,
                    "recommendation_api": status.recommendation_api_healthy,
                    "ab_testing": status.ab_testing_healthy,
                    "kafka": status.kafka_healthy,
                    "redis": status.redis_healthy
                },
                "metrics": ml_system.metrics,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Status check error: {str(e)}"
            )

    @app.get("/metrics", response_model=Dict[str, Any])
    async def get_metrics():
        try:
            if not ml_system or not ml_system.is_initialized:
                raise HTTPException(
                    status_code=503,
                    detail="ML Recommendation System not initialized"
                )

            return {
                "system_metrics": ml_system.metrics,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Metrics error: {str(e)}"
            )

    @app.post("/admin/restart", response_model=Dict[str, str])
    async def restart_system():
        try:
            global ml_system

            if ml_system:
                await ml_system.shutdown()

            config = MLSystemConfig.from_environment()
            ml_system = MLRecommendationSystem()
            success = await ml_system.initialize()

            if success:
                return {
                    "status": "success",
                    "message": "ML Recommendation System restarted successfully",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to restart ML Recommendation System"
                )

        except Exception as e:
            logger.error(f"System restart failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Restart error: {str(e)}"
            )

    @app.get("/admin/config", response_model=Dict[str, Any])
    async def get_system_config():
        try:
            if not ml_system:
                raise HTTPException(
                    status_code=503,
                    detail="ML Recommendation System not available"
                )

            config_dict = ml_system.config_manager.get_config()

            safe_config = config_dict.copy()
            if "redis" in safe_config:
                safe_config["redis"].pop("password", None)
            if "database" in safe_config:
                safe_config["database"]["url"] = "***HIDDEN***"

            return {
                "configuration": safe_config,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Config retrieval failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Config error: {str(e)}"
            )

    return app

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    health_app = create_health_app()

    uvicorn.run(
        health_app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
