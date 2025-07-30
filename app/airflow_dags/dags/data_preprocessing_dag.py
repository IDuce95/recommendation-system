from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import sys
import os

sys.path.append('/opt/airflow/app')

from app.data_processing.data_loader import DataLoader
from app.data_processing.data_preprocessor import DataPreprocessor

default_args = {
    'owner': 'recommendation-system',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'data_preprocessing_pipeline',
    default_args=default_args,
    description='Daily data preprocessing pipeline for recommendation system',
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
    tags=['data', 'preprocessing', 'etl']
)

def extract_raw_data(**context):
    loader = DataLoader()
    raw_data = loader.load_from_database()
    output_path = '/tmp/raw_data.csv'
    raw_data.to_csv(output_path, index=False)
    return output_path

def validate_data_quality(**context):
    input_path = context['task_instance'].xcom_pull(task_ids='extract_data')
    df = pd.read_csv(input_path)
    quality_issues = []
    if df.isnull().sum().sum() > len(df) * 0.1:
        quality_issues.append("Too many null values")
    if len(df) < 100:
        quality_issues.append("Insufficient data volume")
    if 'product_id' not in df.columns:
        quality_issues.append("Missing required columns")
    if quality_issues:
        raise ValueError(f"Data quality issues: {quality_issues}")
    return True

def preprocess_data(**context):
    input_path = context['task_instance'].xcom_pull(task_ids='extract_data')
    df = pd.read_csv(input_path)
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process(df)
    output_path = '/tmp/processed_data.csv'
    processed_data.to_csv(output_path, index=False)
    return output_path

def generate_embeddings(**context):
    input_path = context['task_instance'].xcom_pull(task_ids='preprocess_data')
    df = pd.read_csv(input_path)
    embeddings = []
    for _, row in df.iterrows():
        text = f"{row.get('name', '')} {row.get('description', '')} {row.get('category', '')}"
        embedding = [0.1] * 384
        embeddings.append(embedding)
    df['text_embedding'] = embeddings
    output_path = '/tmp/data_with_embeddings.csv'
    df.to_csv(output_path, index=False)
    return output_path

def load_to_staging(**context):
    input_path = context['task_instance'].xcom_pull(task_ids='generate_embeddings')
    df = pd.read_csv(input_path)
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    df.to_sql(
        'products_staging',
        postgres_hook.get_sqlalchemy_engine(),
        if_exists='replace',
        index=False
    )
    return len(df)

def data_quality_checks(**context):
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    staging_count = postgres_hook.get_first(
        "SELECT COUNT(*) FROM products_staging"
    )[0]
    production_count = postgres_hook.get_first(
        "SELECT COUNT(*) FROM products"
    )[0] or 0
    if staging_count < production_count * 0.9:
        raise ValueError(f"Staging data volume too low: {staging_count} vs {production_count}")
    return True

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_raw_data,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data_quality,
    dag=dag
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

embeddings_task = PythonOperator(
    task_id='generate_embeddings',
    python_callable=generate_embeddings,
    dag=dag
)

staging_task = PythonOperator(
    task_id='load_to_staging',
    python_callable=load_to_staging,
    dag=dag
)

quality_check_task = PythonOperator(
    task_id='final_quality_checks',
    python_callable=data_quality_checks,
    dag=dag
)

promote_task = PostgresOperator(
    task_id='promote_to_production',
    postgres_conn_id='postgres_default',
    sql="""
        BEGIN;

        CREATE TABLE IF NOT EXISTS products_backup AS
        SELECT * FROM products;

        TRUNCATE TABLE products;

        INSERT INTO products
        SELECT * FROM products_staging
        WHERE product_id IS NOT NULL;

        COMMIT;
    """,
    dag=dag
)

cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command='rm -f /tmp/raw_data.csv /tmp/processed_data.csv /tmp/data_with_embeddings.csv',
    dag=dag
)

extract_task >> validate_task >> preprocess_task >> embeddings_task >> staging_task >> quality_check_task >> promote_task >> cleanup_task
