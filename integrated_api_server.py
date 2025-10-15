from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from contextlib import asynccontextmanager

from app.feature_store import FeatureStore
from app.recommendation.recommender import RecommendationEngine
from app.ab_testing import ABTestingSystem
from config.ml_system_config import create_system_config

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = create_system_config()
    
    feature_store = FeatureStore(config.feature_store)
    await feature_store.initialize()
    
    recommendation_engine = RecommendationEngine(
        feature_store=feature_store,
        config=config.recommendation
    )
    await recommendation_engine.initialize()
    
    ab_testing_system = ABTestingSystem(config.ab_testing)
    await ab_testing_system.initialize()
    
    app.state.feature_store = feature_store
    app.state.recommendation_engine = recommendation_engine
    app.state.ab_testing_system = ab_testing_system
    
    yield
    
    await ab_testing_system.shutdown()
    await feature_store.cleanup()

app = FastAPI(
    title="ML Recommendation System",
    description="Integrated API for ML recommendation system",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "integrated-api"}

@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str, num_recommendations: int = 10):
    try:
        recommendations = await app.state.recommendation_engine.get_recommendations(
            user_id=user_id,
            num_recommendations=num_recommendations
        )
        return {"user_id": user_id, "recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/experiments")
async def create_experiment(experiment_data: dict):
    try:
        experiment_id = await app.state.ab_testing_system.create_experiment(
            name=experiment_data["name"],
            description=experiment_data["description"],
            variants=experiment_data["variants"]
        )
        return {"experiment_id": experiment_id}
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "integrated_api_server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
