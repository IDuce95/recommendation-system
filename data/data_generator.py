import logging
import os
import random
from typing import Dict, Union

import pandas as pd
from config import (BATTERY_RANGE, BRANDS, CATEGORY, DATA_DIR, DESCRIPTION,
                    DESCRIPTION_TEMPLATE, FILENAME, ID, IMAGE_PATH, IMAGES_DIR,
                    MEMORY_RANGE, MODELS, NAME, NUM_PRODUCTS,
                    OPERATING_SYSTEMS, PRODUCT_CATEGORIES, SCREEN_RANGE)

logger = logging.getLogger(__name__)


class DataGenerator:
    def __init__(
        self,
        num_products: int = 100,
    ) -> None:
        self.num_products = num_products

    @staticmethod
    def generate_product_description(
        feature_values: Dict[str, Union[str, int, float]],
    ) -> str:
        description = DESCRIPTION_TEMPLATE
        for feature, value in feature_values.items():
            if feature in description:
                description = description.replace(f"[{feature}]", str(value))
        return description

    @staticmethod
    def generate_product_features(
    ) -> Dict[str, Union[str, int, float]]:
        features = {}
        features['category'] = random.choice(PRODUCT_CATEGORIES)
        features['brand'] = random.choice(BRANDS)
        features['model'] = random.choice(MODELS)
        features['screen'] = round(random.uniform(min(SCREEN_RANGE), max(SCREEN_RANGE)), 1)
        features['memory'] = random.randint(min(MEMORY_RANGE), max(MEMORY_RANGE))
        features['battery'] = random.randint(min(BATTERY_RANGE), max(BATTERY_RANGE))
        features['operating_system'] = random.choice(OPERATING_SYSTEMS)
        return features

    @staticmethod
    def generate_image_path(
        category: str,
    ) -> str:
        images_dir = f"{DATA_DIR}/{IMAGES_DIR}"
        image_files = [
            f for f in os.listdir(images_dir)
            if f.endswith(".jpg") and category.lower() in f
        ]
        random_image_file = random.choice(image_files)
        return os.path.join(images_dir, random_image_file)

    def _create_single_product(
        self,
        product_id: int,
    ) -> Dict[str, Union[str, int]]:
        features = self.generate_product_features()
        description = self.generate_product_description(features)
        image_path = self.generate_image_path(features.get('category', 'default'))
        product_name = f"{features.get('brand', 'Unknown')} {features.get('model', 'Unknown')}"

        return {
            ID: product_id,
            CATEGORY: features.get('category', 'Unknown').title(),
            NAME: product_name,
            DESCRIPTION: description,
            IMAGE_PATH: image_path,
        }

    def generate_products_dataframe(
        self,
    ) -> pd.DataFrame:
        products = []

        for i in range(self.num_products):
            product = self._create_single_product(i + 1)
            products.append(product)

        return pd.DataFrame(products)


if __name__ == "__main__":
    data_generator = DataGenerator(num_products=NUM_PRODUCTS)
    df_products = data_generator.generate_products_dataframe()
    df_products.to_csv(f"{DATA_DIR}/{FILENAME}", index=False)

    logger.info(f"Generated dataset: /{DATA_DIR}/{FILENAME}")
