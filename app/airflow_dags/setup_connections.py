from airflow import settings
from airflow.models import Connection
from sqlalchemy.orm import sessionmaker

def create_connections():
    session = settings.Session()

    postgres_conn = Connection(
        conn_id='postgres_default',
        conn_type='postgres',
        host='postgres',
        login='postgres',
        password='password',
        schema='recommendation_system',
        port=5432
    )

    mlflow_conn = Connection(
        conn_id='mlflow_default',
        conn_type='http',
        host='mlflow',
        port=5000
    )

    existing_postgres = session.query(Connection).filter(Connection.conn_id == 'postgres_default').first()
    if not existing_postgres:
        session.add(postgres_conn)

    existing_mlflow = session.query(Connection).filter(Connection.conn_id == 'mlflow_default').first()
    if not existing_mlflow:
        session.add(mlflow_conn)

    session.commit()
    session.close()

if __name__ == "__main__":
    create_connections()
