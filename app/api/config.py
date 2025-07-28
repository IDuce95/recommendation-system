
SERVER_CONFIG = {
    "host": "127.0.0.1",
    "port": 5000,
    "app_module": "app.api.fastapi_server:app"
}

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s"
}

ENDPOINTS = {
    "recommendations": "/get_recommendations/"
}

PRODUCT_FORMAT_TEMPLATE = "- {name}\nCategory: {category}\nDescription: {description}\n\n"

DEFAULT_VALUES = {
    "unknown_product": "Unknown Product",
    "unknown_category": "Unknown",
    "text_weight": 0.5,
    "image_weight": 0.5,
    "use_text_embeddings": True,
    "use_image_embeddings": True
}

RESPONSE_MESSAGES = {
    "no_recommendations": "Unable to find recommendations",
    "no_recommendations_available": "No recommendations available",
    "system_message_template": "Selected product: {name} (Category: {category})"
}

HTTP_STATUS = {
    "internal_server_error": 500
}

RESPONSE_KEYS = {
    "success": "success",
    "system_message": "system_message",
    "recommendations_text": "recommendations_text",
    "recommended_products": "recommended_products",
    "chosen_product": "chosen_product"
}
