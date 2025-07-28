
API_BASE_URL = "http://127.0.0.1:5000"
API_RECOMMENDATIONS_ENDPOINT = "/get_recommendations/"

PAGE_TITLE = "Product Recommendation System"
PAGE_LAYOUT = "wide"

MAIN_COLUMNS_SPEC = [0.4, 0.6]

SLIDER_DEFAULTS = {
    "text_weight_percent": {
        "min_value": 0,
        "max_value": 100,
        "default_value": 50,
        "step": 5,
        "help_text": "Higher values prioritize text-based similarity"
    },
    "top_n_recommendations": {
        "min_value": 1,
        "max_value": 5,
        "default_value": 3,
        "step": 1,
        "help_text": "Select the number of similar products to recommend"
    }
}

DEFAULT_WEIGHTS = {
    "text_only": {"text": 1.0, "image": 0.0},
    "image_only": {"text": 0.0, "image": 1.0},
    "balanced": {"text": 0.5, "image": 0.5}
}

UI_LABELS = {
    "category_select": r"$\textsf{\Large Select product category}$",
    "product_select": r"$\textsf{\Large Select product}$",
    "settings_header": "## Settings",
    "recommendations_header": "## Product Recommendations",
    "selected_product_header": "### Selected product details",
    "embeddings_header": "### Embeddings",
    "recommendations_count_header": "### Number of recommendations",
    "weights_header": "### Weights balance",
    "text_embeddings_checkbox": "Use text embeddings",
    "image_embeddings_checkbox": "Use visual embeddings",
    "generate_button": "Generate",
    "text_weight_slider": "Text similarity weight (%)",
    "text_weight_metric": "Text weight ",
    "image_weight_metric": "Image weight ",
    "recommendations_count_slider": "How many products to recommend?"
}

STATUS_MESSAGES = {
    "success": "✅ Recommendations generated!",
    "error_api": "An error occurred while getting recommendations from API.",
    "error_db": "Error while connecting to database: {error}",
    "warning_no_product": "Please select a product to get recommendations.",
    "warning_no_embeddings": "You must select at least one type of embeddings. Defaulting to text embeddings.",
    "info_text_only": "Using only text embeddings (100% text weight)",
    "info_image_only": "Using only image embeddings (100% image weight)",
    "loading_recommendations": "Generating recommendations...",
    "weight_description": "Adjust the balance between text and image similarity (total = 100%)",
    "weight_distribution": "Weight distribution:"
}

IMAGE_CONFIG = {
    "width": 400
}

DEFAULT_VALUES = {
    "use_text_embeddings": True,
    "use_image_embeddings": True
}
