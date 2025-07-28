import logging
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.recommendation.config import (
    LOGGING_CONFIG, WEIGHT_CONFIG, ERROR_MESSAGES, RESPONSE_KEYS
)

logging.basicConfig(level=getattr(logging, LOGGING_CONFIG["level"]), format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


class Recommender:
    def __init__(
        self,
        product_data: pd.DataFrame,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> None:
        self.product_data = product_data
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings

    def _validate_parameters(
        self,
        product_id: int,
        top_n: int,
    ) -> None:
        if product_id is None:
            raise ValueError("product_id parameter is required and must be specified")
        if top_n is None:
            raise ValueError("top_n parameter is required and must be specified")

    def _find_product_index(
        self,
        product_id: int,
    ) -> int:
        product_indices = self.product_data[self.product_data['id'] == product_id].index
        if len(product_indices) == 0:
            raise IndexError(ERROR_MESSAGES["product_not_found"].format(product_id=product_id))
        return product_indices[0]

    def _calculate_text_similarity_scores(
        self,
        product_index: int,
        text_weight: float,
    ) -> np.ndarray:
        product_embedding = self.text_embeddings[product_index]
        text_similarity_scores = cosine_similarity(
            self.text_embeddings,
            product_embedding.reshape(1, -1)
        ).flatten()
        return text_similarity_scores * text_weight

    def _calculate_image_similarity_scores(
        self,
        product_index: int,
        image_weight: float,
    ) -> np.ndarray:
        product_image_embedding = self.image_embeddings[product_index]
        image_similarity_scores = np.dot(
            self.image_embeddings,
            product_image_embedding
        ) / (
            np.linalg.norm(self.image_embeddings, axis=1) * np.linalg.norm(product_image_embedding) + 1e-8
        )
        return image_similarity_scores * image_weight

    def _get_top_similar_products(
        self,
        combined_similarity_scores: np.ndarray,
        product_id: int,
        top_n: int,
    ) -> pd.DataFrame:
        similar_product_indices = np.argsort(combined_similarity_scores)[::-1]

        top_product_id = self.product_data.iloc[similar_product_indices[0]]['id']
        if top_product_id == product_id:
            similar_product_indices = np.delete(similar_product_indices, 0)

        top_similar_indices = similar_product_indices[:top_n]
        return self.product_data.iloc[top_similar_indices]

    def _log_recommendation_info(
        self,
        product_id: int,
        top_n: int,
        use_text_embeddings: bool,
        use_image_embeddings: bool,
        text_weight: float,
        image_weight: float,
    ) -> None:
        log_message = f"Top {top_n} similar products recommended for product ID: {product_id}"
        log_message += f" (text embedding: {use_text_embeddings} (weight: {text_weight}), "
        log_message += f"image embedding: {use_image_embeddings} (weight: {image_weight}))"
        logger.info(log_message)

    def recommend_products(
        self,
        product_id: int,
        top_n: int,
        use_text_embeddings: bool = True,
        use_image_embeddings: bool = True,
        text_weight: float = WEIGHT_CONFIG["default_text_weight"],
        image_weight: float = WEIGHT_CONFIG["default_image_weight"],
    ) -> Dict[str, Union[Dict, pd.DataFrame]]:
        try:
            self._validate_parameters(product_id, top_n)
            product_index = self._find_product_index(product_id)

            combined_similarity_scores = np.zeros(len(self.product_data))

            if use_text_embeddings:
                text_scores = self._calculate_text_similarity_scores(product_index, text_weight)
                combined_similarity_scores += text_scores

            if use_image_embeddings:
                image_scores = self._calculate_image_similarity_scores(product_index, image_weight)
                combined_similarity_scores += image_scores

            recommended_products = self._get_top_similar_products(
                combined_similarity_scores, product_id, top_n
            )

            self._log_recommendation_info(
                product_id, top_n, use_text_embeddings, use_image_embeddings, text_weight, image_weight
            )

            return {
                RESPONSE_KEYS["chosen_product"]: self.product_data.iloc[product_index].to_dict(),
                RESPONSE_KEYS["recommended_products"]: recommended_products
            }

        except IndexError as e:
            logger.error(ERROR_MESSAGES["recommendation_error"].format(error=str(e)))
            return pd.DataFrame()
