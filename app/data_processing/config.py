
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s"
}

DB_MESSAGES = {
    "connected": "Connected to PostgreSQL database.",
    "disconnected": "Disconnected from PostgreSQL database.",
    "connection_error": "Error connecting to PostgreSQL database: {error}",
    "query_error": "Error executing query: {error}",
    "load_success": "Successfully loaded {count} products from database.",
    "load_error": "Error loading product data: {error}",
    "no_connection": "No database connection available."
}

QUERIES = {
    "load_all_products": "SELECT * FROM products;"
}

TABLES = {
    "products": "products"
}

COLUMNS = {
    "id": "id",
    "name": "name",
    "category": "category",
    "description": "description",
    "image_path": "image_path"
}

DEFAULT_VALUES = {
    "empty_dataframe_message": "No data available",
    "unknown_error": "Unknown error occurred"
}
