from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys

sys.path.append('/opt/airflow/app')

from ai.model_manager import ModelManager
from ai.recommendation_agent import RecommendationAgent

default_args = {
    'owner': 'recommendation-system',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Model training and retraining pipeline',
    schedule_interval=timedelta(days=7),
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'training', 'model']
)

def prepare_training_data(**context):
    print("Preparing training data...")

    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

    query = """
        SELECT
            product_id,
            category,
            price,
            rating,
            text_embedding,
            CASE
                WHEN rating >= 4.0 THEN 1
                ELSE 0
            END as high_quality
        FROM products
        WHERE rating IS NOT NULL
        AND text_embedding IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 10000
    """

    df = postgres_hook.get_pandas_df(query)

    print(f"Prepared {len(df)} training samples")

    output_path = '/tmp/training_data.csv'
    df.to_csv(output_path, index=False)

    return output_path

def train_recommendation_model(**context):
    print("Training recommendation model...")

    input_path = context['task_instance'].xcom_pull(task_ids='prepare_data')
    df = pd.read_csv(input_path)

    mlflow.set_experiment("recommendation_model_training")

    with mlflow.start_run():
        X = df[['price', 'rating']].fillna(0)
        y = df['high_quality']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("training_samples", len(X_train))

        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained with accuracy: {accuracy:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        model_uri = mlflow.get_artifact_uri("model")

        return {
            'model_uri': model_uri,
            'accuracy': accuracy,
            'run_id': mlflow.active_run().info.run_id
        }

def evaluate_model_performance(**context):
    print("Evaluating model performance...")

    model_info = context['task_instance'].xcom_pull(task_ids='train_model')

    current_accuracy = model_info['accuracy']

    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

    previous_accuracy_result = postgres_hook.get_first("""
        SELECT accuracy
        FROM model_performance
        ORDER BY created_at DESC
        LIMIT 1

        INSERT INTO model_performance (run_id, accuracy, improvement, created_at)
        VALUES (%s, %s, %s, %s)
    """, parameters=(
        model_info['run_id'],
        current_accuracy,
        improvement,
        datetime.now()
    ))

    if current_accuracy < 0.6:
        raise ValueError(f"Model performance too low: {current_accuracy:.3f}")

    print(f"Model evaluation completed. Accuracy: {current_accuracy:.3f}, Improvement: {improvement:.3f}")

    return {
        'approved': current_accuracy >= 0.6,
        'accuracy': current_accuracy,
        'improvement': improvement
    }

def deploy_model(**context):
    print("Deploying model to production...")

    model_info = context['task_instance'].xcom_pull(task_ids='train_model')
    evaluation = context['task_instance'].xcom_pull(task_ids='evaluate_model')

    if not evaluation['approved']:
        print("Model not approved for deployment")
        return False

    model_manager = ModelManager()

    success = model_manager.deploy_model(
        model_uri=model_info['model_uri'],
        model_name="recommendation_classifier",
        version="latest"
    )

    if success:
        print(f"Model deployed successfully. Run ID: {model_info['run_id']}")

        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        postgres_hook.run("""
            UPDATE model_performance
            SET deployed = TRUE, deployed_at = %s
            WHERE run_id = %s
        """, parameters=(datetime.now(), model_info['run_id']))

    return success

def update_recommendation_agent(**context):
    print("Updating recommendation agent...")

    deployment_success = context['task_instance'].xcom_pull(task_ids='deploy_model')

    if not deployment_success:
        print("Skipping agent update due to deployment failure")
        return False

    agent = RecommendationAgent()

    success = agent.reload_models()

    if success:
        print("Recommendation agent updated successfully")
    else:
        print("Failed to update recommendation agent")

    return success

def generate_model_report(**context):
    print("Generating model training report...")

    model_info = context['task_instance'].xcom_pull(task_ids='train_model')
    evaluation = context['task_instance'].xcom_pull(task_ids='evaluate_model')
    deployment_success = context['task_instance'].xcom_pull(task_ids='deploy_model')

    report = {
        'run_id': model_info['run_id'],
        'accuracy': evaluation['accuracy'],
        'improvement': evaluation['improvement'],
        'deployed': deployment_success,
        'timestamp': datetime.now().isoformat()
    }

    report_path = f"/tmp/model_report_{model_info['run_id']}.json"

    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Model report generated: {report_path}")
    return report_path

prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_training_data,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_recommendation_model,
    dag=dag
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_performance,
    dag=dag
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

update_agent_task = PythonOperator(
    task_id='update_agent',
    python_callable=update_recommendation_agent,
    dag=dag
)

report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_model_report,
    dag=dag
)

cleanup_task = BashOperator(
    task_id='cleanup',
    bash_command='rm -f /tmp/training_data.csv /tmp/model_report_*.json',
    dag=dag
)

prepare_data_task >> train_model_task >> evaluate_model_task >> deploy_model_task >> update_agent_task >> report_task >> cleanup_task
