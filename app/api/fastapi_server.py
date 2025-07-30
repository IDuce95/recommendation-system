import os
import sys
from contextlib import asynccontextmanager

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.base import BaseHTTPMiddleware

from app.data_processing.data_loader import DataLoader
from app.data_processing.data_preprocessor import DataPreprocessor
from app.recommendation.recommender import Recommender
from config.config_manager import get_config, setup_logging
from ai import ModelManager, RecommendationAgent
from app.api.routers.recommendations import router as recommendations_router, setup_router_dependencies
from app.api.routers.rag import router as rag_router
from app.api.config import DEFAULT_VALUES
from app.api.prometheus_metrics import get_metrics, metrics_middleware
from app.feature_store.integration import get_feature_store

os.environ['CURL_CA_BUNDLE'] = ""

config = get_config()
setup_logging()
logger = logging.getLogger(__name__)

data_loader = None
product_data = None
data_preprocessor = None
recommender = None
model_manager = None
recommendation_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_loader, product_data, data_preprocessor, recommender, model_manager, recommendation_agent

    logger.info("Starting API initialization...")

    data_loader = DataLoader()
    product_data = data_loader.load_product_data()

    if product_data is None:
        logger.error("Failed to load product data during API initialization.")
        exit(1)

    data_preprocessor = DataPreprocessor()
    data_preprocessor.generate_embeddings(product_data)

    image_embeddings = data_preprocessor.image_embeddings
    text_embeddings = data_preprocessor.text_embeddings

    recommender = Recommender(
        product_data=product_data,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
    )

    model_config = config.get_model_config()
    model_manager = ModelManager(model_config)
    model_manager.initialize_model()

    recommendation_agent = RecommendationAgent(
        model_manager=model_manager,
        default_values=DEFAULT_VALUES
    )
    recommendation_agent.initialize()

    setup_router_dependencies(recommender, recommendation_agent)

    logger.info("API initialization complete.")
    yield
    logger.info("API closed.")

app = FastAPI(
    title="Product Recommendation API",
    description="API for product recommendations with AI-powered suggestions",
    version="0.1.0",
    lifespan=lifespan,
)

app.middleware("http")(metrics_middleware)

app.include_router(recommendations_router)
app.include_router(rag_router)

@app.get("/metrics")
async def metrics():
    return get_metrics()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Product Recommendation API",
        "version": "0.1.0"
    }

@app.get("/feature-store/health")
async def feature_store_health():
    feature_store = get_feature_store()
    return feature_store.health_check()

@app.get("/feature-store/stats")
async def feature_store_stats():
    feature_store = get_feature_store()
    return feature_store.get_feature_statistics()

@app.get("/")
async def root():
    return {
        "message": "Welcome to Product Recommendation API",
        "docs": "/docs",
        "health": "/health",
        "feature-store": "/feature-store/health",
        "version": "0.1.0"
    }

if __name__ == "__main__":
    api_config = config.get_api_config()
    logger.info(f"Starting FastAPI server on {api_config['host']}:{api_config['port']}")
    uvicorn.run(
        "app.api.fastapi_server:app",
        host=api_config["host"],
        port=api_config["port"],
        reload=api_config.get("reload", False)
    )
