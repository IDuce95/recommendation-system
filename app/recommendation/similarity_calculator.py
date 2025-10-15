import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.recommendation.config import LOGGING_CONFIG, SUCCESS_MESSAGES

logging.basicConfig(level=getattr(logging, LOGGING_CONFIG["level"]), format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

class CosineSimilarityCalculator:
    def calculate_similarity_matrix(
        self,
        embedding_matrix: np.ndarray,
    ) -> np.ndarray:
        logger.info("Calculating cosine similarity matrix...")
        similarity_matrix = cosine_similarity(embedding_matrix)
        logger.info(SUCCESS_MESSAGES["similarity_calculated"])
        return similarity_matrix
