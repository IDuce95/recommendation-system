
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s"
}

SIMILARITY_CONFIG = {
    "default_metric": "cosine",
    "available_metrics": ["cosine", "euclidean", "manhattan", "dot_product"],
    "default_top_n": 5
}

WEIGHT_CONFIG = {
    "default_text_weight": 0.5,
    "default_image_weight": 0.5,
    "min_weight": 0.0,
    "max_weight": 1.0
}

PATHS = {
    "text_embeddings": "embeddings/text_embeddings.pkl",
    "image_embeddings": "embeddings/image_embeddings.pkl",
    "embeddings_dir": "embeddings/"
}

DEFAULT_VALUES = {
    "unknown_product": "Unknown Product",
    "empty_recommendations": "No recommendations available",
    "invalid_product_id": "Invalid product ID provided"
}

ERROR_MESSAGES = {
    "product_not_found": "Product with ID {product_id} not found.",
    "embeddings_not_found": "Embeddings file not found: {file_path}",
    "invalid_weights": "Text and image weights must sum to 1.0. Got text_weight={text_weight}, image_weight={image_weight}",
    "no_embeddings_selected": "At least one embedding type must be selected.",
    "similarity_calculation_error": "Error calculating similarity: {error}",
    "recommendation_error": "Error generating recommendations: {error}"
}

SUCCESS_MESSAGES = {
    "embeddings_loaded": "Successfully loaded {embedding_type} embeddings.",
    "recommendations_generated": "Successfully generated {count} recommendations for product ID {product_id}.",
    "similarity_calculated": "Similarity calculation completed successfully."
}

RESPONSE_KEYS = {
    "recommended_products": "recommended_products",
    "chosen_product": "chosen_product",
    "similarity_scores": "similarity_scores",
    "success": "success",
    "error": "error",
    "message": "message"
}
