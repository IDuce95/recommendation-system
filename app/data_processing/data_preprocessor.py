import os
import sys
from typing import List

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
import string

import numpy as np
import pandas as pd
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from PIL import Image
from torchvision import models, transforms

from app.data_processing.data_loader import DataLoader
from app.recommendation.recommender import Recommender
from app.recommendation.similarity_calculator import CosineSimilarityCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(
        self,
    ) -> None:
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.embeddings_model = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
        self.text_embeddings = None
        self.image_embeddings = None

        self.image_model = models.resnet50(weights="DEFAULT")
        self.image_model = torch.nn.Sequential(*list(self.image_model.children())[:-1])
        self.image_model.eval()

        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_text(
        self,
        text: str,
    ) -> str:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens_without_stopwords = [token for token in tokens if token not in self.stop_words]
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens_without_stopwords]
        return " ".join(lemmatized_tokens)

    def generate_text_embeddings(
        self,
        descriptions: List[str],
    ) -> np.ndarray:
        processed_descriptions = [self.preprocess_text(desc) for desc in descriptions]
        embedding_list = self.embeddings_model.embed_documents(processed_descriptions)
        self.text_embeddings = np.array(embedding_list)
        logger.info("Text embeddings generated for product descriptions using Langchain.")
        return self.text_embeddings

    def get_text_embeddings(
        self,
    ) -> np.ndarray:
        if self.text_embeddings is None:
            raise ValueError("Text embeddings not generated yet. Call generate_text_embeddings first.")
        return self.text_embeddings

    def preprocess_descriptions(
        self,
        descriptions_series: pd.Series,
    ) -> pd.Series:
        return descriptions_series.apply(self.preprocess_text)

    def _process_single_image(
        self,
        image_path: str,
    ) -> np.ndarray:
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}. Skipping image embedding.")
            return None

        try:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
            image = image.unsqueeze(0)

            with torch.no_grad():
                embedding_vector = self.image_model(image)
            embedding_vector = embedding_vector.flatten().numpy()

            return embedding_vector
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}. Skipping image embedding.")
            return None

    def _filter_valid_embeddings(
        self,
        image_embedding_list: List,
    ) -> np.ndarray:
        valid_embeddings = [emb for emb in image_embedding_list if emb is not None]
        return np.array(valid_embeddings)

    def generate_image_embeddings(
        self,
        image_paths: List[str],
    ) -> np.ndarray:
        image_embedding_list = []
        for image_path in image_paths:
            embedding = self._process_single_image(image_path)
            image_embedding_list.append(embedding)

        self.image_embeddings = self._filter_valid_embeddings(image_embedding_list)
        logger.info("Image embeddings generated for product images using ResNet50.")
        return self.image_embeddings

    def generate_embeddings(
        self,
        products: pd.DataFrame,
    ) -> None:
        description_embeddings = self.generate_text_embeddings(products['description'])
        image_embeddings = self.generate_image_embeddings(products['image_path'])

        self.text_embeddings = description_embeddings
        self.image_embeddings = image_embeddings

        logger.info("Data preprocessing completed: Text and image embeddings generated.")

    def get_text_similarity_matrix(
        self,
    ) -> np.ndarray:
        if self.text_embeddings is None:
            raise ValueError("Text embeddings not generated yet. Call generate_text_embeddings first.")
        similarity_calculator = CosineSimilarityCalculator()
        text_similarity_matrix = similarity_calculator.calculate_similarity_matrix(self.text_embeddings)
        logger.info("Text similarity matrix generated using Cosine Similarity.")
        return text_similarity_matrix

if __name__ == "__main__":
    loader = DataLoader()
    product_data = loader.load_product_data()

    if product_data is not None:
        preprocessor = DataPreprocessor()
        product_data['processed_description_text'] = product_data['description'].apply(preprocessor.preprocess_text)
        text_embeddings = preprocessor.generate_text_embeddings(product_data['processed_description_text'])

        similarity_calculator = CosineSimilarityCalculator()
        similarity_matrix = similarity_calculator.calculate_similarity_matrix(text_embeddings)

        recommender = Recommender(
            product_data=product_data,
            image_embeddings=preprocessor.image_embeddings,
            text_embeddings=text_embeddings
        )
        product_id_to_recommend = 1
        recommendation_result = recommender.recommend_products(product_id=product_id_to_recommend, top_n=3)
        top_recommendations = recommendation_result.get('recommended_products', pd.DataFrame())

        logger.info("Sample of original and processed descriptions (lemmatized text strings):")
        sample_df = product_data[['description', 'processed_description_text']].sample(5)
        logger.info(sample_df)

        logger.info("\nShape of Text Embedding matrix (Langchain):")
        logger.info(text_embeddings.shape)
        logger.info("\nSample Text Embedding (Langchain, first 5 values of the first embedding):")
        logger.info(text_embeddings[0][:5])

        if not top_recommendations.empty:
            original_product_name = product_data.loc[product_data['id'] == product_id_to_recommend, 'name'].iloc[0]
            original_product_category = product_data.loc[product_data['id'] == product_id_to_recommend, 'category'].iloc[0]

            logger.info(f"\nRecommendations for product ID {product_id_to_recommend} (Langchain):")
            logger.info(f"  Original Product: {original_product_name} (Category: {original_product_category})")
            logger.info("  Top 5 Recommended Products:")
            for index, product in top_recommendations.iterrows():
                logger.info(f"    ID: {product.get('id', 'Unknown')}, Name: {product.get('name', 'Unknown')}, Category: {product.get('category', 'Unknown')}")
        else:
            logger.warning(f"No recommendations found for product ID {product_id_to_recommend} or product not found.")

    else:
        logger.error("Data loading failed, preprocessing and Text Embeddings (Langchain) cannot be demonstrated.")
