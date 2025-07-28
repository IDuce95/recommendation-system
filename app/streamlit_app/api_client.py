import logging
from typing import Dict, List, Optional, Union

import requests
import streamlit as st

logger = logging.getLogger(__name__)


class RecommendationAPIClient:

    def __init__(self, base_url: str, endpoint: str):
        self.base_url = base_url
        self.endpoint = endpoint
        self.api_url = f"{base_url}{endpoint}"

    def get_recommendations(
        self,
        selected_product_info: Dict[str, Union[int, str]],
        use_text_embeddings: bool,
        use_image_embeddings: bool,
        top_n: int,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
    ) -> Optional[Dict[str, Union[bool, str, List[Dict[str, str]]]]]:
        params = {
            'product_id': int(selected_product_info.get('id', 0)),
            'top_n': top_n,
            'use_text_embeddings': use_text_embeddings,
            'use_image_embeddings': use_image_embeddings,
            'text_weight': text_weight,
            'image_weight': image_weight
        }

        try:
            response = requests.post(self.api_url, json=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            st.error("Error connecting to recommendation service. Please try again.")
            return None
