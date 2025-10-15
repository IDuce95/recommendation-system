import os
import sys
from typing import Optional

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
import warnings

import pandas as pd
import psycopg2

warnings.filterwarnings('ignore')

from config.config_manager import get_config
from app.data_processing.config import LOGGING_CONFIG, DB_MESSAGES, QUERIES

config = get_config()
logging.basicConfig(level=getattr(logging, LOGGING_CONFIG["level"]), format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(
        self,
    ) -> None:
        self.db_config = config.get_database_config()
        self.conn = None

    def _connect_to_db(
        self,
    ) -> None:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info(DB_MESSAGES["connected"])
        except Exception as e:
            logger.error(DB_MESSAGES["connection_error"].format(error=e))
            self.conn = None
            raise

    def _disconnect_from_db(
        self,
    ) -> None:
        if self.conn:
            self.conn.close()
            logger.info(DB_MESSAGES["disconnected"])
            self.conn = None

    def load_product_data(
        self,
    ) -> Optional[pd.DataFrame]:
        try:
            self._connect_to_db()
            if self.conn:
                query = QUERIES["load_all_products"]
                product_data = pd.read_sql_query(query, self.conn)
                logger.info(DB_MESSAGES["load_success"].format(count=len(product_data)))
                return product_data
            else:
                logger.error(DB_MESSAGES["no_connection"])
                return None
        except Exception as e:
            logger.error(DB_MESSAGES["load_error"].format(error=e))
            return None
        finally:
            self._disconnect_from_db()

if __name__ == "__main__":
    data_loader = DataLoader()
    product_data = data_loader.load_product_data()

    if product_data is not None:
        logger.info("\nSample of product data loaded from PostgreSQL:")
        logger.info(product_data.sample(5))
        logger.info("\nShape of product data:")
        logger.info(product_data.shape)
    else:
        logger.error("Data loading from PostgreSQL failed.")
