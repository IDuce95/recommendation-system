import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

from .config import FeatureStoreConfig, COMPUTED_FEATURES

logger = logging.getLogger(__name__)

class FeatureComputer:
    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.computed_features_config = COMPUTED_FEATURES.copy()

        self.computation_stats = {
            "features_computed": 0,
            "computation_time": 0.0,
            "errors": 0
        }

        logger.info("Feature Computer initialized")

    def compute_all_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:

        start_time = time.time()
        computed_features = {}

        try:
            basic_features = self._extract_basic_features(raw_data)
            computed_features.update(basic_features)

            derived_features = self._compute_derived_features(raw_data, computed_features)
            computed_features.update(derived_features)

            embedding_features = self._compute_embeddings(raw_data)
            computed_features.update(embedding_features)

            stats_features = self._compute_statistical_features(raw_data, computed_features)
            computed_features.update(stats_features)

            computed_features["computed_at"] = datetime.now().isoformat()
            computed_features["computation_version"] = "1.0"

            self.computation_stats["features_computed"] += len(computed_features)
            self.computation_stats["computation_time"] += time.time() - start_time

            logger.info(f"Computed {len(computed_features)} features")
            return computed_features

        except Exception as e:
            self.computation_stats["errors"] += 1
            logger.error(f"Error computing features: {e}")
            return {}

    def _extract_basic_features(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:

        basic_features = {}

        feature_mappings = {
            "product_id": ["id", "product_id"],
            "name": ["name", "title", "product_name"],
            "category": ["category", "product_category"],
            "description": ["description", "product_description"],
            "brand": ["brand", "manufacturer"],
            "price": ["price", "cost", "amount"]
        }

        for feature_name, possible_keys in feature_mappings.items():
            for key in possible_keys:
                if key in raw_data and raw_data[key] is not None:
                    basic_features[feature_name] = raw_data[key]
                    break

        if "price" in basic_features:
            try:
                basic_features["price"] = float(basic_features["price"])
            except (ValueError, TypeError):
                basic_features["price"] = None

        if "product_id" in basic_features:
            try:
                basic_features["product_id"] = int(basic_features["product_id"])
            except (ValueError, TypeError):
                pass

        return basic_features

    def _compute_derived_features(
        self,
        raw_data: Dict[str, Any],
        basic_features: Dict[str, Any]
    ) -> Dict[str, Any]:

        derived_features = {}

        try:
            if "description" in basic_features:
                description = basic_features["description"]
                if description:
                    derived_features["description_length"] = len(str(description))
                    derived_features["description_word_count"] = len(str(description).split())
                    derived_features["has_description"] = True
                else:
                    derived_features["has_description"] = False

            if "name" in basic_features:
                name = basic_features["name"]
                if name:
                    derived_features["name_length"] = len(str(name))
                    derived_features["name_word_count"] = len(str(name).split())

            if "category" in basic_features:
                category = basic_features["category"]
                if category:
                    derived_features["category_normalized"] = str(category).lower().strip()

            if "price" in basic_features and basic_features["price"] is not None:
                price = basic_features["price"]
                derived_features["price_range"] = self._categorize_price(price)
                derived_features["is_premium"] = price > 1000  # Threshold can be configurable

            text_content = []
            for field in ["name", "description", "brand", "category"]:
                if field in basic_features and basic_features[field]:
                    text_content.append(str(basic_features[field]))

            if text_content:
                combined_text = " ".join(text_content)
                derived_features["combined_text"] = combined_text
                derived_features["combined_text_length"] = len(combined_text)
                derived_features["combined_word_count"] = len(combined_text.split())

        except Exception as e:
            logger.warning(f"Error computing derived features: {e}")

        return derived_features

    def _compute_embeddings(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:

        embedding_features = {}

        try:

            if "text_embedding" in raw_data:
                embedding_features["text_embedding"] = raw_data["text_embedding"]

            if "image_embedding" in raw_data:
                embedding_features["image_embedding"] = raw_data["image_embedding"]

        except Exception as e:
            logger.warning(f"Error computing embeddings: {e}")

        return embedding_features

    def _compute_statistical_features(
        self,
        raw_data: Dict[str, Any],
        computed_features: Dict[str, Any]
    ) -> Dict[str, Any]:

        stats_features = {}

        try:
            total_features = len(computed_features)
            non_null_features = sum(1 for v in computed_features.values() if v is not None)

            if total_features > 0:
                stats_features["feature_completeness"] = non_null_features / total_features

            text_quality_score = 0.0
            text_features = 0

            if "description_length" in computed_features:
                desc_len = computed_features["description_length"]
                if desc_len > 0:
                    text_quality_score += min(desc_len / 500, 1.0)  # Normalize to 0-1
                    text_features += 1

            if "name_length" in computed_features:
                name_len = computed_features["name_length"]
                if 5 <= name_len <= 100:  # Reasonable range
                    text_quality_score += 1.0
                    text_features += 1

            if text_features > 0:
                stats_features["text_quality_score"] = text_quality_score / text_features

            stats_features["is_fresh"] = True  # Placeholder - would use actual timestamp logic

        except Exception as e:
            logger.warning(f"Error computing statistical features: {e}")

        return stats_features

    def compute_popularity_score(self, interaction_data: Dict[str, Any]) -> float:

        try:
            view_count = interaction_data.get("view_count", 0)
            interaction_count = interaction_data.get("interaction_count", 0)
            rating_avg = interaction_data.get("rating_avg", 0)

            popularity = (
                view_count * 0.3 +
                interaction_count * 0.5 +
                rating_avg * 20 * 0.2  # Scale rating to similar range
            )

            return min(popularity / 100, 1.0)  # Normalize to 0-1

        except Exception as e:
            logger.error(f"Error computing popularity score: {e}")
            return 0.0

    def compute_category_similarity(self, product_data: Dict[str, Any]) -> float:

        try:

            return 0.75  # Placeholder

        except Exception as e:
            logger.error(f"Error computing category similarity: {e}")
            return 0.0

    def compute_embedding_freshness(self, embedding_data: Dict[str, Any]) -> float:

        try:
            if "last_updated" not in embedding_data:
                return 0.0

            last_updated = datetime.fromisoformat(embedding_data["last_updated"])
            age_hours = (datetime.now() - last_updated).total_seconds() / 3600

            if age_hours < 1:
                return 1.0
            elif age_hours < 24:
                return 0.8
            elif age_hours < 168:  # 1 week
                return 0.5
            else:
                return 0.2

        except Exception as e:
            logger.error(f"Error computing embedding freshness: {e}")
            return 0.0

    def _categorize_price(self, price: float) -> str:

        if price < 50:
            return "budget"
        elif price < 200:
            return "mid_range"
        elif price < 1000:
            return "premium"
        else:
            return "luxury"

    def get_computation_stats(self) -> Dict[str, Any]:

        return self.computation_stats.copy()

    def reset_stats(self) -> None:

        self.computation_stats = {
            "features_computed": 0,
            "computation_time": 0.0,
            "errors": 0
        }

    def register_custom_computation(
        self,
        feature_name: str,
        computation_function: callable,
        dependencies: List[str]
    ) -> bool:

        try:
            self.computed_features_config[feature_name] = {
                "depends_on": dependencies,
                "computation_function": computation_function.__name__,
                "custom": True,
                "registered_at": datetime.now().isoformat()
            }

            setattr(self, f"compute_{feature_name}", computation_function)

            logger.info(f"Registered custom computation for feature: {feature_name}")
            return True

        except Exception as e:
            logger.error(f"Error registering custom computation: {e}")
            return False
