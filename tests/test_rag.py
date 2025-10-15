import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from ai.rag_node import RAGNode
from ai.chromadb_client import ChromaDBClient, KnowledgeBaseManager
from ai.knowledge_base_generator import FakeKnowledgeBaseGenerator

class TestRAGNode:

    @patch('ai.rag_node.ChromaDBClient')
    def test_rag_node_initialization(self, mock_chromadb_client):
        mock_client = MagicMock()
        mock_chromadb_client.return_value = mock_client

        rag_node = RAGNode(mock_client)

        assert rag_node.chroma_client == mock_client
        assert rag_node.kb_manager is not None

    def test_retrieve_relevant_context(self):
        mock_client = MagicMock()
        mock_kb_manager = MagicMock()

        rag_node = RAGNode(mock_client)
        rag_node.kb_manager = mock_kb_manager

        mock_kb_manager.search_product_knowledge.return_value = {
            'documents': [['Product information']],
            'metadatas': [[{'product_id': 'test_product', 'category': 'Electronics'}]]
        }

        mock_client.query_collection.return_value = {
            'documents': [['Knowledge snippet']],
            'metadatas': [[{'type': 'product_features'}]]
        }

        contexts = rag_node.retrieve_relevant_context("test query", n_results=3)

        assert len(contexts) > 0
        assert contexts[0]['content'] == 'Product information'
        assert contexts[0]['source'] == 'product_knowledge'

    def test_format_context(self):
        mock_client = MagicMock()
        rag_node = RAGNode(mock_client)

        contexts = [
            {
                'content': 'Test product information',
                'metadata': {'product_id': 'test_123', 'price': 299.99},
                'source': 'product_knowledge',
                'relevance_score': 0.9
            }
        ]

        formatted = rag_node.format_context(contexts)

        assert 'Test product information' in formatted
        assert 'Product ID: test_123' in formatted
        assert 'Price: $299.99' in formatted
        assert 'Relevance: 0.90' in formatted

    def test_generate_template_response(self):
        mock_client = MagicMock()
        rag_node = RAGNode(mock_client)

        contexts = [
            {
                'content': 'Great laptop with excellent performance',
                'metadata': {'product_id': 'laptop_123', 'category': 'Laptops', 'price': 999.99},
                'source': 'product_knowledge',
                'relevance_score': 0.95
            }
        ]

        response = rag_node.generate_template_response("laptop recommendations", contexts, "User profile info")

        assert 'laptop_123' in response
        assert 'Laptops' in response
        assert '$999.99' in response
        assert 'recommendations' in response.lower()

class TestChromaDBClient:

    @patch('ai.chromadb_client.chromadb.HttpClient')
    def test_chromadb_client_initialization(self, mock_http_client):
        mock_client = MagicMock()
        mock_http_client.return_value = mock_client

        client = ChromaDBClient(host="localhost", port=8001)

        assert client.client == mock_client
        mock_http_client.assert_called_once()

    @patch('ai.chromadb_client.chromadb.HttpClient')
    def test_create_collection(self, mock_http_client):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_http_client.return_value = mock_client

        client = ChromaDBClient()
        collection = client.create_collection("test_collection")

        assert collection == mock_collection
        assert "test_collection" in client.collections

    @patch('ai.chromadb_client.chromadb.HttpClient')
    def test_add_documents(self, mock_http_client):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_http_client.return_value = mock_client

        client = ChromaDBClient()

        documents = ["Document 1", "Document 2"]
        metadatas = [{"type": "test"}, {"type": "test"}]
        ids = ["doc1", "doc2"]

        client.add_documents("test_collection", documents, metadatas, ids)

        mock_collection.add.assert_called_once_with(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

class TestKnowledgeBaseManager:

    def test_add_product_knowledge(self):
        mock_client = MagicMock()
        kb_manager = KnowledgeBaseManager(mock_client)

        products_df = pd.DataFrame([
            {
                'product_id': 'prod_001',
                'name': 'Test Laptop',
                'category': 'Laptops',
                'brand': 'TestBrand',
                'price': 999.99,
                'description': 'High-performance laptop'
            }
        ])

        kb_manager.add_product_knowledge(products_df)

        mock_client.add_documents.assert_called_once()
        call_args = mock_client.add_documents.call_args

        assert call_args[0][0] == kb_manager.product_collection
        assert len(call_args[0][1]) == 1
        assert 'Test Laptop' in call_args[0][1][0]
        assert 'prod_001' in call_args[0][3][0]

class TestFakeKnowledgeBaseGenerator:

    def test_generate_fake_products(self):
        generator = FakeKnowledgeBaseGenerator()

        products_df = generator.generate_fake_products(num_products=10)

        assert len(products_df) == 10
        assert 'product_id' in products_df.columns
        assert 'name' in products_df.columns
        assert 'category' in products_df.columns
        assert 'price' in products_df.columns
        assert 'description' in products_df.columns

        assert all(products_df['category'].isin(generator.categories))
        assert all(products_df['price'] > 0)

    def test_generate_fake_users(self):
        generator = FakeKnowledgeBaseGenerator()

        users = generator.generate_fake_users(num_users=5)

        assert len(users) == 5
        assert all('user_id' in user for user in users)
        assert all('categories' in user for user in users)
        assert all('min_price' in user for user in users)
        assert all('max_price' in user for user in users)

    def test_generate_fake_interactions(self):
        generator = FakeKnowledgeBaseGenerator()

        users = generator.generate_fake_users(num_users=3)
        products_df = generator.generate_fake_products(num_products=5)

        interactions = generator.generate_fake_interactions(users, products_df, num_interactions=10)

        assert len(interactions) == 10
        assert all('user_id' in interaction for interaction in interactions)
        assert all('product_id' in interaction for interaction in interactions)
        assert all('action' in interaction for interaction in interactions)
        assert all(interaction['action'] in generator.actions for interaction in interactions)

    def test_generate_knowledge_snippets(self):
        generator = FakeKnowledgeBaseGenerator()

        products_df = generator.generate_fake_products(num_products=5)
        snippets = generator.generate_knowledge_snippets(products_df)

        assert len(snippets) > 0
        assert all('id' in snippet for snippet in snippets)
        assert all('content' in snippet for snippet in snippets)
        assert all('type' in snippet for snippet in snippets)
        assert all('product_id' in snippet for snippet in snippets)

if __name__ == '__main__':
    pytest.main([__file__])
