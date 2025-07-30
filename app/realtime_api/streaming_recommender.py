import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .config import RealtimeConfig, EventType, DEFAULT_REALTIME_CONFIG
from .kafka_producer import KafkaEventProducer

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class RecommendationRequest:
    request_id: str
    user_id: str
    num_recommendations: int = 10
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Recommendation:
    product_id: str
    score: float
    reason: str
    confidence: float
    features: Optional[Dict[str, Any]] = None

@dataclass
class RecommendationResponse:
    request_id: str
    user_id: str
    recommendations: List[Recommendation]
    processing_time_ms: float
    model_version: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class StreamingRecommender:
    def __init__(
        self,
        config: Optional[RealtimeConfig] = None,
        kafka_producer: Optional[KafkaEventProducer] = None
    ):
        self.config = config or DEFAULT_REALTIME_CONFIG
        self.kafka_producer = kafka_producer

        self.user_embeddings_cache = {}
        self.product_embeddings_cache = {}
        self.user_features_cache = {}
        self.product_features_cache = {}

        self.cache_ttl_seconds = 300  # 5 minutes
        self.last_cache_cleanup = time.time()

        self.metrics = {
            "requests_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_inference_time": 0.0,
            "total_feature_fetch_time": 0.0,
            "fallback_recommendations": 0
        }

        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)

        self._initialize_mock_data()

    def _initialize_mock_data(self):
        self.mock_user_embeddings = {}
        for i in range(1000):
            user_id = f"user_{i}"
            np.random.seed(hash(user_id) % 2**32)
            embedding = np.random.normal(0, 1, 100).astype(np.float32)
            self.mock_user_embeddings[user_id] = embedding

        self.mock_product_embeddings = {}
        for i in range(100):
            product_id = f"product_{i}"
            np.random.seed(hash(product_id) % 2**32)
            embedding = np.random.normal(0, 1, 100).astype(np.float32)
            self.mock_product_embeddings[product_id] = embedding

        self.mock_popularity = {
            f"product_{i}": np.random.beta(2, 5) for i in range(100)
        }

        logger.info("Mock data initialized for testing")

    async def generate_recommendations(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:

        start_time = time.time()

        try:
            await self._cleanup_cache_if_needed()

            user_features = await self._get_user_features(request.user_id)
            user_embedding = await self._get_user_embedding(request.user_id)

            candidate_products = await self._get_candidate_products(
                request.user_id, request.context
            )

            scored_candidates = await self._score_candidates(
                user_embedding, user_features, candidate_products, request.context
            )

            recommendations = await self._rank_and_select(
                scored_candidates, request.num_recommendations
            )

            processing_time = (time.time() - start_time) * 1000
            response = RecommendationResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                recommendations=recommendations,
                processing_time_ms=processing_time,
                model_version="streaming_v1.0",
                timestamp=datetime.now(),
                metadata={
                    "candidates_evaluated": len(candidate_products),
                    "cache_hit_rate": self._get_cache_hit_rate()
                }
            )

            self.metrics["requests_processed"] += 1
            self.metrics["total_inference_time"] += processing_time / 1000

            if self.kafka_producer:
                await self._send_recommendation_response(response)

            return response

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return await self._generate_fallback_recommendations(request)

    async def _get_user_features(self, user_id: str) -> Dict[str, Any]:

        cache_key = f"user_features:{user_id}"

        if cache_key in self.user_features_cache:
            cache_entry = self.user_features_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl_seconds:
                self.metrics["cache_hits"] += 1
                return cache_entry["data"]

        self.metrics["cache_misses"] += 1
        start_time = time.time()

        features = await self._fetch_user_features_mock(user_id)

        self.user_features_cache[cache_key] = {
            "data": features,
            "timestamp": time.time()
        }

        self.metrics["total_feature_fetch_time"] += time.time() - start_time
        return features

    async def _fetch_user_features_mock(self, user_id: str) -> Dict[str, Any]:

        await asyncio.sleep(0.001)

        user_hash = hash(user_id) % 2**32
        np.random.seed(user_hash)

        return {
            "age_group": np.random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]),
            "total_purchases": np.random.poisson(5),
            "avg_rating": np.random.uniform(3.0, 5.0),
            "categories_purchased": np.random.choice(
                ["electronics", "books", "clothing", "home"],
                size=np.random.randint(1, 4),
                replace=False
            ).tolist(),
            "last_purchase_days_ago": np.random.exponential(30),
            "preferred_price_range": np.random.choice(["low", "medium", "high"]),
            "engagement_score": np.random.beta(2, 3)
        }

    async def _get_user_embedding(self, user_id: str) -> np.ndarray:

        cache_key = f"user_embedding:{user_id}"

        if cache_key in self.user_embeddings_cache:
            cache_entry = self.user_embeddings_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl_seconds:
                self.metrics["cache_hits"] += 1
                return cache_entry["data"]

        self.metrics["cache_misses"] += 1

        embedding = self.mock_user_embeddings.get(
            user_id,
            np.random.normal(0, 1, 100).astype(np.float32)
        )

        self.user_embeddings_cache[cache_key] = {
            "data": embedding,
            "timestamp": time.time()
        }

        return embedding

    async def _get_candidate_products(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:

        all_products = list(self.mock_product_embeddings.keys())

        if context and "category" in context:
            category_products = [p for p in all_products if context["category"] in p]
            if category_products:
                return category_products[:50]  # Limit candidates

        popular_products = sorted(
            all_products,
            key=lambda p: self.mock_popularity.get(p, 0),
            reverse=True
        )

        return popular_products[:50]  # Limit to top 50 candidates

    async def _score_candidates(
        self,
        user_embedding: np.ndarray,
        user_features: Dict[str, Any],
        candidate_products: List[str],
        context: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:

        scored_candidates = []

        for product_id in candidate_products:
            try:
                product_embedding = self.mock_product_embeddings.get(
                    product_id,
                    np.random.normal(0, 1, 100).astype(np.float32)
                )

                if SKLEARN_AVAILABLE:
                    similarity = cosine_similarity(
                        user_embedding.reshape(1, -1),
                        product_embedding.reshape(1, -1)
                    )[0][0]
                else:
                    similarity = np.dot(user_embedding, product_embedding) / (
                        np.linalg.norm(user_embedding) * np.linalg.norm(product_embedding)
                    )

                popularity = self.mock_popularity.get(product_id, 0.1)

                final_score = 0.6 * similarity + 0.4 * popularity

                if context:
                    if context.get("boost_recent"):
                        final_score *= 1.1  # 10% boost for recent items

                    if context.get("preferred_category") in product_id:
                        final_score *= 1.2  # 20% boost for preferred category

                scored_candidates.append((product_id, float(final_score)))

            except Exception as e:
                logger.warning(f"Error scoring product {product_id}: {e}")
                continue

        return scored_candidates

    async def _rank_and_select(
        self,
        scored_candidates: List[Tuple[str, float]],
        num_recommendations: int
    ) -> List[Recommendation]:

        ranked_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)

        recommendations = []
        for i, (product_id, score) in enumerate(ranked_candidates[:num_recommendations]):
            recommendation = Recommendation(
                product_id=product_id,
                score=score,
                reason=self._generate_reason(product_id, score, i),
                confidence=min(score, 1.0),  # Cap confidence at 1.0
                features={
                    "rank": i + 1,
                    "popularity": self.mock_popularity.get(product_id, 0.1),
                    "category": self._extract_category(product_id)
                }
            )
            recommendations.append(recommendation)

        return recommendations

    def _generate_reason(self, product_id: str, score: float, rank: int) -> str:

        if rank == 0:
            return "Top match based on your preferences"
        elif score > 0.8:
            return "Highly recommended for you"
        elif score > 0.6:
            return "Popular with similar users"
        elif score > 0.4:
            return "Based on your browsing history"
        else:
            return "Trending in your area of interest"

    def _extract_category(self, product_id: str) -> str:

        if "laptop" in product_id:
            return "electronics"
        elif "book" in product_id:
            return "books"
        elif "clothing" in product_id:
            return "clothing"
        else:
            return "general"

    async def _send_recommendation_response(self, response: RecommendationResponse):
        try:
            event_data = {
                "event_type": EventType.RECOMMENDATION_RESPONSE.value,
                "request_id": response.request_id,
                "user_id": response.user_id,
                "recommendations": [
                    {
                        "product_id": rec.product_id,
                        "score": rec.score,
                        "reason": rec.reason,
                        "confidence": rec.confidence
                    }
                    for rec in response.recommendations
                ],
                "processing_time_ms": response.processing_time_ms,
                "model_version": response.model_version,
                "timestamp": response.timestamp.isoformat()
            }

            await self.kafka_producer.send_event(event_data)

        except Exception as e:
            logger.error(f"Failed to send recommendation response: {e}")

    async def _generate_fallback_recommendations(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:

        self.metrics["fallback_recommendations"] += 1

        popular_products = sorted(
            self.mock_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:request.num_recommendations]

        recommendations = [
            Recommendation(
                product_id=product_id,
                score=popularity * 0.5,  # Lower confidence for fallback
                reason="Popular recommendation (fallback)",
                confidence=0.3,
                features={"fallback": True}
            )
            for product_id, popularity in popular_products
        ]

        return RecommendationResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            recommendations=recommendations,
            processing_time_ms=50.0,  # Fast fallback
            model_version="fallback_v1.0",
            timestamp=datetime.now(),
            metadata={"fallback": True}
        )

    async def _cleanup_cache_if_needed(self):
        current_time = time.time()

        if current_time - self.last_cache_cleanup > 60:  # Cleanup every minute
            expired_keys = [
                key for key, value in self.user_features_cache.items()
                if current_time - value["timestamp"] > self.cache_ttl_seconds
            ]
            for key in expired_keys:
                del self.user_features_cache[key]

            expired_keys = [
                key for key, value in self.user_embeddings_cache.items()
                if current_time - value["timestamp"] > self.cache_ttl_seconds
            ]
            for key in expired_keys:
                del self.user_embeddings_cache[key]

            self.last_cache_cleanup = current_time

            if expired_keys:
                logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")

    def _get_cache_hit_rate(self) -> float:

        total_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total_requests == 0:
            return 0.0
        return self.metrics["cache_hits"] / total_requests

    def get_metrics(self) -> Dict[str, Any]:

        total_requests = self.metrics["requests_processed"]
        avg_inference_time = (
            self.metrics["total_inference_time"] / max(1, total_requests)
        )
        avg_feature_time = (
            self.metrics["total_feature_fetch_time"] / max(1, total_requests)
        )

        return {
            "requests_processed": total_requests,
            "cache_hit_rate": self._get_cache_hit_rate(),
            "avg_inference_time_ms": avg_inference_time * 1000,
            "avg_feature_fetch_time_ms": avg_feature_time * 1000,
            "fallback_rate": self.metrics["fallback_recommendations"] / max(1, total_requests),
            "cache_sizes": {
                "user_features": len(self.user_features_cache),
                "user_embeddings": len(self.user_embeddings_cache),
                "product_features": len(self.product_features_cache),
                "product_embeddings": len(self.product_embeddings_cache)
            }
        }

    def health_check(self) -> Dict[str, Any]:

        metrics = self.get_metrics()

        is_healthy = (
            metrics["avg_inference_time_ms"] < self.config.max_response_time_ms and
            metrics["fallback_rate"] < 0.1  # Less than 10% fallback rate
        )

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "ml_libraries_available": {
                "torch": TORCH_AVAILABLE,
                "sklearn": SKLEARN_AVAILABLE
            },
            "metrics": metrics
        }
