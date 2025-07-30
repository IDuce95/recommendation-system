from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import sys

sys.path.append('/opt/airflow/app')

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
    'embeddings_generation_pipeline',
    default_args=default_args,
    description='Generate and update text embeddings for products',
    schedule_interval=timedelta(hours=6),
    catchup=False,
    max_active_runs=1,
    tags=['embeddings', 'ml', 'nlp']
)

def get_products_needing_embeddings(**context):
    print("Finding products that need embeddings...")

    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

    query = """
        SELECT product_id, name, description, category, brand
        FROM products
        WHERE text_embedding IS NULL
           OR updated_at > embedding_generated_at
           OR embedding_generated_at IS NULL
        ORDER BY updated_at DESC
        LIMIT 1000
    """

    df = postgres_hook.get_pandas_df(query)

    print(f"Found {len(df)} products needing embeddings")

    if len(df) == 0:
        print("No products need embeddings, skipping pipeline")
        return None

    output_path = '/tmp/products_for_embeddings.csv'
    df.to_csv(output_path, index=False)

    return output_path

def generate_text_embeddings(**context):
    print("Generating text embeddings...")

    input_path = context['task_instance'].xcom_pull(task_ids='get_products')

    if input_path is None:
        print("No products to process, skipping")
        return None

    df = pd.read_csv(input_path)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings_data = []

    for _, row in df.iterrows():
        text_content = f"{row.get('name', '')} {row.get('description', '')} {row.get('category', '')} {row.get('brand', '')}"

        text_content = text_content.strip()
        if not text_content:
            embedding = np.zeros(384).tolist()
        else:
            embedding = model.encode(text_content).tolist()

        embeddings_data.append({
            'product_id': row['product_id'],
            'text_embedding': embedding,
            'embedding_model': 'all-MiniLM-L6-v2',
            'embedding_version': '1.0'
        })

        if len(embeddings_data) % 100 == 0:
            print(f"Processed {len(embeddings_data)} embeddings...")

    embeddings_df = pd.DataFrame(embeddings_data)

    output_path = '/tmp/generated_embeddings.csv'
    embeddings_df.to_csv(output_path, index=False)

    print(f"Generated {len(embeddings_df)} embeddings")
    return output_path

def validate_embeddings(**context):
    print("Validating generated embeddings...")

    input_path = context['task_instance'].xcom_pull(task_ids='generate_embeddings')

    if input_path is None:
        print("No embeddings to validate, skipping")
        return True

    df = pd.read_csv(input_path)

    issues = []

    if len(df) == 0:
        issues.append("No embeddings generated")

    null_embeddings = df['text_embedding'].isna().sum()
    if null_embeddings > 0:
        issues.append(f"{null_embeddings} null embeddings found")

    zero_embeddings = 0
    for _, row in df.iterrows():
        try:
            embedding = eval(row['text_embedding'])
            if all(x == 0 for x in embedding):
                zero_embeddings += 1
        except:
            issues.append(f"Invalid embedding format for product {row['product_id']}")

    if zero_embeddings > len(df) * 0.1:
        issues.append(f"Too many zero embeddings: {zero_embeddings}")

    if issues:
        raise ValueError(f"Embedding validation failed: {issues}")

    print(f"Validation passed for {len(df)} embeddings")
    return True

def update_embeddings_in_db(**context):
    print("Updating embeddings in database...")

    input_path = context['task_instance'].xcom_pull(task_ids='generate_embeddings')

    if input_path is None:
        print("No embeddings to update, skipping")
        return 0

    df = pd.read_csv(input_path)

    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

    updated_count = 0

    for _, row in df.iterrows():
        postgres_hook.run("""
            UPDATE products
            SET
                text_embedding = %s,
                embedding_model = %s,
                embedding_version = %s,
                embedding_generated_at = %s
            WHERE product_id = %s
        """, parameters=(
            row['text_embedding'],
            row['embedding_model'],
            row['embedding_version'],
            datetime.now(),
            row['product_id']
        ))

        updated_count += 1

        if updated_count % 100 == 0:
            print(f"Updated {updated_count} products...")

    print(f"Updated embeddings for {updated_count} products")
    return updated_count

def calculate_similarity_matrix(**context):
    print("Calculating product similarity matrix...")

    updated_count = context['task_instance'].xcom_pull(task_ids='update_embeddings')

    if updated_count == 0:
        print("No embeddings updated, skipping similarity calculation")
        return 0

    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

    query = """
        SELECT product_id, text_embedding
        FROM products
        WHERE text_embedding IS NOT NULL
        LIMIT 5000
    """

    df = postgres_hook.get_pandas_df(query)

    if len(df) < 2:
        print("Not enough products with embeddings for similarity calculation")
        return 0

    embeddings = []
    product_ids = []

    for _, row in df.iterrows():
        try:
            embedding = eval(row['text_embedding'])
            embeddings.append(embedding)
            product_ids.append(row['product_id'])
        except:
            continue

    embeddings_matrix = np.array(embeddings)

    similarity_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)

    similarities_data = []

    for i, product_1 in enumerate(product_ids):
        similarities = similarity_matrix[i]
        top_indices = np.argsort(similarities)[-11:][::-1][1:]

        for j, idx in enumerate(top_indices):
            if similarities[idx] > 0.5:
                similarities_data.append({
                    'product_1': product_1,
                    'product_2': product_ids[idx],
                    'similarity_score': float(similarities[idx]),
                    'rank': j + 1
                })

    if similarities_data:
        similarities_df = pd.DataFrame(similarities_data)

        postgres_hook.run("DELETE FROM product_similarities")

        for _, row in similarities_df.iterrows():
            postgres_hook.run("""
                INSERT INTO product_similarities
                (product_1, product_2, similarity_score, rank, updated_at)
                VALUES (%s, %s, %s, %s, %s)
            """, parameters=(
                row['product_1'],
                row['product_2'],
                row['similarity_score'],
                row['rank'],
                datetime.now()
            ))

    print(f"Calculated {len(similarities_data)} similarity relationships")
    return len(similarities_data)

def generate_embedding_report(**context):
    print("Generating embedding pipeline report...")

    updated_count = context['task_instance'].xcom_pull(task_ids='update_embeddings')
    similarities_count = context['task_instance'].xcom_pull(task_ids='calculate_similarities')

    report = {
        'embeddings_updated': updated_count or 0,
        'similarities_calculated': similarities_count or 0,
        'timestamp': datetime.now().isoformat(),
        'model_used': 'all-MiniLM-L6-v2'
    }

    print(f"Embedding pipeline completed:")
    print(f"- Updated embeddings: {report['embeddings_updated']}")
    print(f"- Calculated similarities: {report['similarities_calculated']}")

    return report

get_products_task = PythonOperator(
    task_id='get_products',
    python_callable=get_products_needing_embeddings,
    dag=dag
)

generate_embeddings_task = PythonOperator(
    task_id='generate_embeddings',
    python_callable=generate_text_embeddings,
    dag=dag
)

validate_embeddings_task = PythonOperator(
    task_id='validate_embeddings',
    python_callable=validate_embeddings,
    dag=dag
)

update_embeddings_task = PythonOperator(
    task_id='update_embeddings',
    python_callable=update_embeddings_in_db,
    dag=dag
)

calculate_similarities_task = PythonOperator(
    task_id='calculate_similarities',
    python_callable=calculate_similarity_matrix,
    dag=dag
)

report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_embedding_report,
    dag=dag
)

get_products_task >> generate_embeddings_task >> validate_embeddings_task >> update_embeddings_task >> calculate_similarities_task >> report_task
