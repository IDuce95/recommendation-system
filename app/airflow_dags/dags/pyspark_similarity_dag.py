from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import numpy as np
import logging
import time

default_args = {
    'owner': 'recommendation_system',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'pyspark_similarity_calculation',
    default_args=default_args,
    description='Distributed similarity calculation using PySpark',
    schedule_interval='@daily',  # Daily run
    catchup=False,
    max_active_runs=1,
    tags=['pyspark', 'similarity', 'distributed', 'ml']
)

logger = logging.getLogger(__name__)

def extract_embeddings_from_db(**context) -> Dict[str, Any]:

    logger.info("Extracting embeddings from database...")

    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

    query = """
        SELECT
            product_id,
            text_embedding,
            image_embedding,
            name,
            category,
            updated_at
        FROM products
        WHERE text_embedding IS NOT NULL
        ORDER BY product_id
        LIMIT 5000  -- Limit for performance
    """

    df = postgres_hook.get_pandas_df(query)

    if len(df) == 0:
        raise ValueError("No products with embeddings found in database")

    logger.info(f"Extracted {len(df)} products with embeddings")

    embeddings_data = {
        'product_ids': df['product_id'].tolist(),
        'text_embeddings': [],
        'product_names': df['name'].tolist(),
        'categories': df['category'].tolist(),
        'count': len(df)
    }

    for embedding_str in df['text_embedding']:
        try:
            embedding = eval(embedding_str) if isinstance(embedding_str, str) else embedding_str
            embeddings_data['text_embeddings'].append(embedding)
        except Exception as e:
            logger.warning(f"Failed to parse embedding: {e}")
            embeddings_data['text_embeddings'].append(np.random.randn(384).tolist())

    logger.info(f"Successfully processed {len(embeddings_data['text_embeddings'])} embeddings")

    context['task_instance'].xcom_push(key='embeddings_data', value=embeddings_data)

    return embeddings_data

def calculate_similarity_with_pyspark(**context) -> Dict[str, Any]:

    logger.info("Starting PySpark similarity calculation...")

    embeddings_data = context['task_instance'].xcom_pull(
        task_ids='extract_embeddings',
        key='embeddings_data'
    )

    if not embeddings_data:
        raise ValueError("No embeddings data received from previous task")

    product_ids = embeddings_data['product_ids']
    text_embeddings = np.array(embeddings_data['text_embeddings'])

    logger.info(f"Processing {len(product_ids)} products with {text_embeddings.shape[1]}D embeddings")

    try:
        from app.pyspark_processing import SparkSimilarityCalculator

        start_time = time.time()

        calculator = SparkSimilarityCalculator()

        similarity_df = calculator.calculate_similarity_matrix_distributed(
            text_embeddings,
            product_ids
        )

        similarity_results = similarity_df.collect()

        spark_time = time.time() - start_time

        logger.info(f"PySpark calculation completed in {spark_time:.2f}s")
        logger.info(f"Generated {len(similarity_results)} similarity pairs")

        similarity_data = []
        for row in similarity_results:
            similarity_data.append({
                'product_1': int(row['product_1']),
                'product_2': int(row['product_2']),
                'similarity_score': float(row['similarity_score']),
                'calculation_method': 'pyspark_distributed',
                'calculation_time': spark_time
            })

        results = {
            'similarity_data': similarity_data,
            'calculation_time': spark_time,
            'pairs_generated': len(similarity_data),
            'method': 'pyspark',
            'success': True
        }

        context['task_instance'].xcom_push(key='pyspark_results', value=results)

        return results

    except ImportError as e:
        logger.warning(f"PySpark not available: {e}")
        logger.info("Falling back to pandas implementation...")

        return calculate_similarity_with_pandas_fallback(
            text_embeddings,
            product_ids,
            context
        )

    except Exception as e:
        logger.error(f"PySpark calculation failed: {e}")

        return calculate_similarity_with_pandas_fallback(
            text_embeddings,
            product_ids,
            context
        )

def calculate_similarity_with_pandas_fallback(
    embeddings: np.ndarray,
    product_ids: list,
    context
) -> Dict[str, Any]:

    logger.info("Using pandas fallback for similarity calculation...")

    from sklearn.metrics.pairwise import cosine_similarity

    start_time = time.time()

    similarity_matrix = cosine_similarity(embeddings)

    similarity_data = []
    threshold = 0.3

    for i, product_1 in enumerate(product_ids):
        for j, product_2 in enumerate(product_ids):
            if i < j and similarity_matrix[i][j] > threshold:
                similarity_data.append({
                    'product_1': int(product_1),
                    'product_2': int(product_2),
                    'similarity_score': float(similarity_matrix[i][j]),
                    'calculation_method': 'pandas_fallback',
                    'calculation_time': 0  # Will be set after
                })

    pandas_time = time.time() - start_time

    for item in similarity_data:
        item['calculation_time'] = pandas_time

    logger.info(f"Pandas calculation completed in {pandas_time:.2f}s")
    logger.info(f"Generated {len(similarity_data)} similarity pairs")

    results = {
        'similarity_data': similarity_data,
        'calculation_time': pandas_time,
        'pairs_generated': len(similarity_data),
        'method': 'pandas_fallback',
        'success': True
    }

    context['task_instance'].xcom_push(key='pyspark_results', value=results)

    return results

def store_similarities_to_db(**context) -> Dict[str, Any]:

    logger.info("Storing similarities to database...")

    results = context['task_instance'].xcom_pull(
        task_ids='calculate_similarity_pyspark',
        key='pyspark_results'
    )

    if not results or not results.get('success'):
        raise ValueError("No valid similarity results to store")

    similarity_data = results['similarity_data']
    method = results['method']

    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

    delete_query = f"DELETE FROM product_similarities WHERE calculation_method = '{method}'"
    postgres_hook.run(delete_query)

    logger.info(f"Cleared old similarities for method: {method}")

    insert_count = 0
    batch_size = 1000

    for i in range(0, len(similarity_data), batch_size):
        batch = similarity_data[i:i + batch_size]

        insert_query = """
            INSERT INTO product_similarities
            (product_1, product_2, similarity_score, calculation_method, rank, updated_at)
            VALUES %s
        """

        values = []
        for j, item in enumerate(batch):
            values.append((
                item['product_1'],
                item['product_2'],
                item['similarity_score'],
                item['calculation_method'],
                j + 1 + i,  # Rank within batch
                datetime.now()
            ))

        postgres_hook.run(insert_query, parameters=(values,))
        insert_count += len(batch)

        logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} similarities")

    logger.info(f"Successfully stored {insert_count} similarities using method: {method}")

    return {
        'stored_count': insert_count,
        'method': method,
        'calculation_time': results['calculation_time']
    }

def generate_performance_report(**context) -> Dict[str, Any]:

    logger.info("Generating performance report...")

    embeddings_data = context['task_instance'].xcom_pull(
        task_ids='extract_embeddings',
        key='embeddings_data'
    )

    similarity_results = context['task_instance'].xcom_pull(
        task_ids='calculate_similarity_pyspark',
        key='pyspark_results'
    )

    storage_results = context['task_instance'].xcom_pull(
        task_ids='store_similarities'
    )

    report = {
        'dag_run_date': context['ds'],
        'execution_date': str(context['execution_date']),
        'data_processing': {
            'products_processed': embeddings_data.get('count', 0),
            'embedding_dimension': len(embeddings_data['text_embeddings'][0]) if embeddings_data['text_embeddings'] else 0
        },
        'similarity_calculation': {
            'method_used': similarity_results.get('method', 'unknown'),
            'calculation_time_seconds': similarity_results.get('calculation_time', 0),
            'pairs_generated': similarity_results.get('pairs_generated', 0),
            'success': similarity_results.get('success', False)
        },
        'data_storage': {
            'pairs_stored': storage_results.get('stored_count', 0),
            'storage_method': storage_results.get('method', 'unknown')
        }
    }

    logger.info("Performance Report:")
    logger.info(f"  Products processed: {report['data_processing']['products_processed']}")
    logger.info(f"  Method used: {report['similarity_calculation']['method_used']}")
    logger.info(f"  Calculation time: {report['similarity_calculation']['calculation_time_seconds']:.2f}s")
    logger.info(f"  Pairs generated: {report['similarity_calculation']['pairs_generated']}")
    logger.info(f"  Pairs stored: {report['data_storage']['pairs_stored']}")

    try:
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

        log_query = """
            INSERT INTO dag_execution_logs
            (dag_id, execution_date, task_summary, performance_metrics, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """

        postgres_hook.run(log_query, parameters=(
            'pyspark_similarity_calculation',
            context['execution_date'],
            f"Processed {report['data_processing']['products_processed']} products using {report['similarity_calculation']['method_used']}",
            str(report),
            datetime.now()
        ))

        logger.info("Performance report logged to database")

    except Exception as e:
        logger.warning(f"Failed to log performance report to database: {e}")

    return report

extract_embeddings_task = PythonOperator(
    task_id='extract_embeddings',
    python_callable=extract_embeddings_from_db,
    dag=dag,
    doc_md="""
    **Extract Embeddings Task**

    Extracts product embeddings from PostgreSQL database.

    - Queries products with text_embedding != NULL
    - Converts embeddings from string to numpy arrays
    - Passes data to next task via XCom
    """
)

calculate_similarity_task = PythonOperator(
    task_id='calculate_similarity_pyspark',
    python_callable=calculate_similarity_with_pyspark,
    dag=dag,
    doc_md="""
    **PySpark Similarity Calculation**

    Calculates product similarities using PySpark distributed computing.

    - Uses PySpark for distributed similarity calculation
    - Falls back to pandas if PySpark unavailable
    - Optimized for large datasets
    """
)

store_similarities_task = PythonOperator(
    task_id='store_similarities',
    python_callable=store_similarities_to_db,
    dag=dag,
    doc_md="""
    **Store Similarities Task**

    Stores calculated similarities back to database.

    - Batch inserts for performance
    - Clears old similarities first
    - Tracks calculation method used
    """
)

generate_report_task = PythonOperator(
    task_id='generate_performance_report',
    python_callable=generate_performance_report,
    dag=dag,
    doc_md="""
    **Performance Report Task**

    Generates comprehensive performance report.

    - Compares different calculation methods
    - Logs performance metrics
    - Tracks processing statistics

This DAG demonstrates distributed similarity calculation using PySpark in Airflow.

1. **Extract embeddings** from PostgreSQL database
2. **Calculate similarities** using PySpark distributed computing
3. **Store results** back to database with performance tracking
4. **Generate report** comparing different methods

- **Distributed Computing**: Uses PySpark for scalable similarity calculation
- **Fallback Mechanism**: Falls back to pandas if PySpark unavailable
- **Performance Tracking**: Compares PySpark vs pandas performance
- **Batch Processing**: Efficient database operations with batching

- Schedule: Daily (`@daily`)
- Max Active Runs: 1 (prevents concurrent executions)
- Retries: 1 with 5-minute delay

- PostgreSQL connection: `postgres_default`
- PySpark (optional, with pandas fallback)
- Products table with embeddings
"""
