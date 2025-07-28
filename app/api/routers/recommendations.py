import logging
from typing import Dict

import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from app.api.config import (
    DEFAULT_VALUES, RESPONSE_MESSAGES, HTTP_STATUS, RESPONSE_KEYS,
    ENDPOINTS
)
from app.api.models import RecommendationResponse, ImageInfo

logger = logging.getLogger(__name__)

recommender = None
recommendation_agent = None


def setup_router_dependencies(recommender_instance, agent_instance):
    global recommender, recommendation_agent
    recommender = recommender_instance
    recommendation_agent = agent_instance


router = APIRouter(prefix="/api", tags=["recommendations"])


def _get_product_recommendations(
    product_id: int,
    top_n: int,
    use_text_embeddings: bool,
    use_image_embeddings: bool,
    text_weight: float,
    image_weight: float,
) -> Dict:
    return recommender.recommend_products(
        product_id=product_id,
        top_n=top_n,
        use_text_embeddings=use_text_embeddings,
        use_image_embeddings=use_image_embeddings,
        text_weight=text_weight,
        image_weight=image_weight
    )


def _create_system_message(selected_product: Dict) -> str:
    category = selected_product.get('category', DEFAULT_VALUES["unknown_category"])
    name = selected_product.get('name', DEFAULT_VALUES["unknown_product"])
    description = selected_product.get('description', '').replace('\n', ', ')
    return f"Generate recommendations for product:  \n{category} {name} ({description})"


def _handle_empty_recommendations(system_message: str) -> RecommendationResponse:
    return RecommendationResponse(
        success=False,
        system_message=system_message,
        recommendations_text=RESPONSE_MESSAGES["no_recommendations"],
        images=[],
        product_count=0
    )


def _generate_ai_response(recommended_products: pd.DataFrame, original_product: Dict) -> Dict:
    return recommendation_agent.generate_recommendations(
        recommended_products, [], original_product
    )


def _build_success_response(system_message: str, response_data: Dict) -> RecommendationResponse:
    images = []
    for img in response_data.get("images", []):
        images.append(ImageInfo(
            product_name=img.get("product_name", ""),
            image_path=img.get("image_path", ""),
            category=img.get("category", "")
        ))

    return RecommendationResponse(
        success=True,
        system_message=system_message,
        recommendations_text=response_data.get("text", ""),
        images=images,
        product_count=response_data.get("product_count", 0)
    )


@router.post(ENDPOINTS["recommendations"], response_model=RecommendationResponse)
async def get_recommendations_endpoint(
    product_id: int = Body(..., title="Product ID"),
    top_n: int = Body(..., title="Number of Recommendations"),
    use_text_embeddings: bool = Body(DEFAULT_VALUES["use_text_embeddings"], title="Use Text Embeddings"),
    use_image_embeddings: bool = Body(DEFAULT_VALUES["use_image_embeddings"], title="Use Image Embeddings"),
    text_weight: float = Body(DEFAULT_VALUES["text_weight"], title="Text Weight"),
    image_weight: float = Body(DEFAULT_VALUES["image_weight"], title="Image Weight"),
) -> RecommendationResponse:
    try:
        recommender_response = _get_product_recommendations(
            product_id, top_n, use_text_embeddings, use_image_embeddings, text_weight, image_weight
        )

        recommended_products = recommender_response.get(RESPONSE_KEYS["recommended_products"], pd.DataFrame())
        selected_product = recommender_response.get(RESPONSE_KEYS["chosen_product"], {})
        system_message = _create_system_message(selected_product)

        if recommended_products.empty:
            return _handle_empty_recommendations(system_message)

        response_data = _generate_ai_response(recommended_products, selected_product)
        return _build_success_response(system_message, response_data)

    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {e}")
        raise HTTPException(status_code=HTTP_STATUS["internal_server_error"], detail=str(e))
