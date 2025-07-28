from typing import Dict, Union

NUM_PRODUCTS = 200

DATA_DIR = "data"
IMAGES_DIR = "images"
FILENAME = "dataset.csv"

SMARTPHONE = "smartphone"
LAPTOP = "laptop"
SMARTWATCH = "smartwatch"
TABLET = "tablet"
PRODUCT_CATEGORIES = [SMARTPHONE, LAPTOP, SMARTWATCH, TABLET]

ID = "id"
NAME = "name"
CATEGORY = "category"
DESCRIPTION = "description"
IMAGE_PATH = "image_path"

BRANDS = ['Samsung', 'Apple', 'Huawei', 'Xiaomi', 'Lenovo', 'Dell', 'Sony']
MODELS = ['Pro', 'Max', 'Ultra', 'Galaxy', 'X', 'Elite', 'Titan']
SCREEN_RANGE = [5.0, 15.6]
MEMORY_RANGE = [4, 32]
BATTERY_RANGE = [2000, 10000]
OPERATING_SYSTEMS = ['Android', 'iOS', 'Windows 11', 'macOS']

DESCRIPTION_TEMPLATE = "screen [screen] inch\nmemory [memory] GB\nbattery [battery] mAh\noperating system [operating_system]"

DB_CONFIG: Dict[str, Union[str, int]] = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'password',
    'database': 'recommendation_system'
}

DB_LOADING_CONFIG = {
    "batch_size": 100,
    "expected_columns": ['id', 'category', 'name', 'description', 'image_path']
}

DB_LOADING_MESSAGES = {
    "reading_csv": "Wczytywanie danych z pliku: {file_path}",
    "csv_loaded": "Wczytano {count} rekordów z pliku CSV",
    "csv_columns": "Kolumny w pliku CSV: {columns}",
    "missing_columns_error": "Brakujące kolumny w pliku CSV: {missing_columns}",
    "missing_columns_exception": "Brakujące kolumny: {missing_columns}",
    "connecting_db": "Łączenie z bazą danych PostgreSQL...",
    "connected_db": "Połączono z bazą danych PostgreSQL",
    "connection_error": "Błąd podczas łączenia z bazą danych: {error}",
    "table_created": "Tabela 'products' została utworzona",
    "table_exists": "Tabela 'products' już istnieje",
    "table_error": "Błąd podczas tworzenia tabeli: {error}",
    "inserting_data": "Wstawianie danych do tabeli 'products'...",
    "data_inserted": "Pomyślnie wstawiono {count} rekordów do tabeli 'products'",
    "insert_error": "Błąd podczas wstawiania danych: {error}",
    "disconnected": "Rozłączono z bazą danych"
}
