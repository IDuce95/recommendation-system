import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'airflow_dags', 'dags'))

class TestDataPreprocessingDAG:

    @patch('data_preprocessing_dag.PostgresHook')
    def test_check_new_data(self, mock_postgres_hook):
        from data_preprocessing_dag import check_new_data

        mock_hook = MagicMock()
        mock_postgres_hook.return_value = mock_hook
        mock_hook.get_pandas_df.return_value = pd.DataFrame({
            'product_id': [1, 2, 3],
            'name': ['Product 1', 'Product 2', 'Product 3']
        })

        context = {'task_instance': MagicMock()}
        result = check_new_data(**context)

        assert result is not None
        assert 'products_to_process.csv' in result
        mock_hook.get_pandas_df.assert_called_once()

    def test_validate_data_quality_pass(self):
        from data_preprocessing_dag import validate_data_quality

        test_df = pd.DataFrame({
            'product_id': [1, 2, 3],
            'name': ['Product 1', 'Product 2', 'Product 3'],
            'price': [10.0, 20.0, 30.0],
            'category': ['A', 'B', 'C']
        })

        test_df.to_csv('/tmp/test_products.csv', index=False)

        context = {
            'task_instance': MagicMock()
        }
        context['task_instance'].xcom_pull.return_value = '/tmp/test_products.csv'

        result = validate_data_quality(**context)
        assert result is True

    def test_validate_data_quality_fail(self):
        from data_preprocessing_dag import validate_data_quality

        test_df = pd.DataFrame({
            'product_id': [1, None, 3],
            'name': ['Product 1', '', 'Product 3'],
            'price': [10.0, 20.0, None],
            'category': ['A', 'B', 'C']
        })

        test_df.to_csv('/tmp/test_products_bad.csv', index=False)

        context = {
            'task_instance': MagicMock()
        }
        context['task_instance'].xcom_pull.return_value = '/tmp/test_products_bad.csv'

        with pytest.raises(ValueError):
            validate_data_quality(**context)

class TestModelTrainingDAG:

    @patch('model_training_dag.PostgresHook')
    def test_prepare_training_data(self, mock_postgres_hook):
        from model_training_dag import prepare_training_data

        mock_hook = MagicMock()
        mock_postgres_hook.return_value = mock_hook
        mock_hook.get_pandas_df.return_value = pd.DataFrame({
            'user_id': [1, 2, 3],
            'product_id': [1, 2, 3],
            'rating': [4.5, 3.0, 5.0]
        })

        context = {'task_instance': MagicMock()}
        result = prepare_training_data(**context)

        assert result is not None
        assert 'training_data.csv' in result
        mock_hook.get_pandas_df.assert_called_once()

    @patch('model_training_dag.mlflow')
    def test_train_recommendation_model(self, mock_mlflow):
        from model_training_dag import train_recommendation_model

        test_df = pd.DataFrame({
            'user_id': [1, 2, 3, 1, 2],
            'product_id': [1, 2, 3, 2, 1],
            'rating': [4.5, 3.0, 5.0, 4.0, 3.5]
        })
        test_df.to_csv('/tmp/test_training_data.csv', index=False)

        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock()

        context = {
            'task_instance': MagicMock()
        }
        context['task_instance'].xcom_pull.return_value = '/tmp/test_training_data.csv'

        result = train_recommendation_model(**context)
        assert 'model_path' in result
        assert 'metrics' in result

class TestEmbeddingsGenerationDAG:

    @patch('embeddings_generation_dag.PostgresHook')
    def test_get_products_needing_embeddings(self, mock_postgres_hook):
        from embeddings_generation_dag import get_products_needing_embeddings

        mock_hook = MagicMock()
        mock_postgres_hook.return_value = mock_hook
        mock_hook.get_pandas_df.return_value = pd.DataFrame({
            'product_id': [1, 2],
            'name': ['Product 1', 'Product 2'],
            'description': ['Desc 1', 'Desc 2']
        })

        context = {'task_instance': MagicMock()}
        result = get_products_needing_embeddings(**context)

        assert result is not None
        assert 'products_for_embeddings.csv' in result

    @patch('embeddings_generation_dag.SentenceTransformer')
    def test_generate_text_embeddings(self, mock_sentence_transformer):
        from embeddings_generation_dag import generate_text_embeddings

        test_df = pd.DataFrame({
            'product_id': [1, 2],
            'name': ['Product 1', 'Product 2'],
            'description': ['Description 1', 'Description 2'],
            'category': ['Category A', 'Category B'],
            'brand': ['Brand X', 'Brand Y']
        })
        test_df.to_csv('/tmp/test_products_embeddings.csv', index=False)

        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 384
        mock_sentence_transformer.return_value = mock_model

        context = {
            'task_instance': MagicMock()
        }
        context['task_instance'].xcom_pull.return_value = '/tmp/test_products_embeddings.csv'

        result = generate_text_embeddings(**context)
        assert result is not None
        assert 'generated_embeddings.csv' in result

if __name__ == '__main__':
    pytest.main([__file__])
