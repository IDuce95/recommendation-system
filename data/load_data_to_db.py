
import logging
import os
import sys

import pandas as pd
import psycopg2
from config import DB_LOADING_CONFIG, DB_LOADING_MESSAGES
from psycopg2.extras import execute_values

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config_manager import get_config

config = get_config()
db_config = config.get_database_config()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _read_and_validate_csv(
    csv_file_path: str,
) -> pd.DataFrame:
    logger.info(DB_LOADING_MESSAGES["reading_csv"].format(file_path=csv_file_path))

    df = pd.read_csv(csv_file_path, quotechar='"', skipinitialspace=True)
    logger.info(DB_LOADING_MESSAGES["csv_loaded"].format(count=len(df)))
    logger.info(DB_LOADING_MESSAGES["csv_columns"].format(columns=list(df.columns)))

    expected_columns = DB_LOADING_CONFIG["expected_columns"]
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        logger.error(DB_LOADING_MESSAGES["missing_columns_error"].format(missing_columns=missing_columns))
        raise ValueError(DB_LOADING_MESSAGES["missing_columns_exception"].format(missing_columns=missing_columns))

    return df


def _clean_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.dropna(subset=['id', 'name'])
    df = df.fillna('')
    logger.info(f"Po czyszczeniu: {len(df)} rekordów gotowych do załadowania")
    return df


def _establish_database_connection():
    logger.info("Łączenie z bazą danych PostgreSQL...")
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    return conn, cursor


def _truncate_table_if_needed(
    cursor,
    conn,
    table_name: str,
    truncate_first: bool,
) -> None:
    if truncate_first:
        logger.info(f"Czyszczenie tabeli {table_name}...")
        cursor.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE;")
        conn.commit()
        logger.info("Tabela wyczyszczona")


def _insert_data_to_table(
    cursor,
    conn,
    df: pd.DataFrame,
    table_name: str,
) -> None:
    columns = df.columns.tolist()
    values = df.values.tolist()


    logger.info(f"Wstawianie {len(values)} rekordów do tabeli {table_name}...")
    execute_values(
        cursor,
        insert_query,
        values,
        template=None,
        page_size=100
    )

    conn.commit()
    logger.info("Dane zostały pomyślnie załadowane do bazy danych")


def _verify_insertion(
    cursor,
    table_name: str,
) -> None:
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    count = cursor.fetchone()[0]
    logger.info(f"Liczba rekordów w tabeli {table_name}: {count}")


def _close_database_connection(
    cursor,
    conn,
) -> None:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    logger.info("Połączenie z bazą danych zamknięte")


def load_csv_to_database(
    csv_file_path: str,
    table_name: str = 'products',
    truncate_first: bool = True,
) -> None:
    conn = None
    cursor = None

    try:
        df = _read_and_validate_csv(csv_file_path)
        df = _clean_dataframe(df)

        conn, cursor = _establish_database_connection()
        _truncate_table_if_needed(cursor, conn, table_name, truncate_first)
        _insert_data_to_table(cursor, conn, df, table_name)
        _verify_insertion(cursor, table_name)

    except Exception as e:
        logger.error(f"Błąd podczas ładowania danych: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        _close_database_connection(cursor, conn)


def verify_data_load(
    table_name: str = 'products',
) -> None:
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
        rows = cursor.fetchall()

        logger.info("Przykładowe dane z tabeli:")
        for row in rows:
            logger.info(f"ID: {row[0]}, Name: {row[1]}, Description: {row[2][:50]}...")

    except Exception as e:
        logger.error(f"Błąd podczas weryfikacji: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    csv_file = "data/dataset.csv"

    if not os.path.exists(csv_file):
        logger.error(f"Plik {csv_file} nie istnieje!")
        sys.exit(1)

    try:
        load_csv_to_database(csv_file, truncate_first=True)

        verify_data_load()

        logger.info("Proces ładowania danych zakończony pomyślnie!")

    except Exception as e:
        logger.error(f"Proces ładowania danych zakończony błędem: {e}")
        sys.exit(1)
