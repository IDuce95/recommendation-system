import chromadb
from chromadb.config import Settings
import pandas as pd
from typing import List, Dict, Any, Optional
import json
import numpy as np

class ChromaDBClient:
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                chroma_client_auth_credentials="test-token"
            )
        )
        self.collections = {}

    def create_collection(self, name: str, metadata: Optional[Dict] = None) -> Any:
        try:
            collection = self.client.get_collection(name)
            print(f"Collection '{name}' already exists")
        except Exception:
            collection = self.client.create_collection(
                name=name,
                metadata=metadata or {}
            )
            print(f"Created collection '{name}'")

        self.collections[name] = collection
        return collection

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ) -> None:
        if collection_name not in self.collections:
            self.create_collection(collection_name)

        collection = self.collections[collection_name]

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Added {len(documents)} documents to '{collection_name}'")

    def query_collection(
        self,
        collection_name: str,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        if collection_name not in self.collections:
            collection = self.client.get_collection(collection_name)
            self.collections[collection_name] = collection
        else:
            collection = self.collections[collection_name]

        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where
        )

        return results

    def get_collection_info(self, collection_name: str) -> Dict:
        if collection_name not in self.collections:
            collection = self.client.get_collection(collection_name)
            self.collections[collection_name] = collection
        else:
            collection = self.collections[collection_name]

        count = collection.count()

        return {
            "name": collection_name,
            "count": count,
            "metadata": collection.metadata
        }

    def list_collections(self) -> List[str]:
        collections = self.client.list_collections()
        return [col.name for col in collections]

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name)
        if collection_name in self.collections:
            del self.collections[collection_name]
        print(f"Deleted collection '{collection_name}'")

class KnowledgeBaseManager:
    def __init__(self, chroma_client: ChromaDBClient):
        self.chroma_client = chroma_client
        self.product_collection = "product_knowledge"
        self.user_collection = "user_knowledge"
        self.interaction_collection = "interaction_knowledge"

    def initialize_knowledge_base(self) -> None:
        self.chroma_client.create_collection(
            self.product_collection,
            metadata={"description": "Product information and descriptions"}
        )

        self.chroma_client.create_collection(
            self.user_collection,
            metadata={"description": "User preferences and behaviors"}
        )

        self.chroma_client.create_collection(
            self.interaction_collection,
            metadata={"description": "User-product interactions and feedback"}
        )

    def add_product_knowledge(self, products_df: pd.DataFrame) -> None:
        documents = []
        metadatas = []
        ids = []

        for _, row in products_df.iterrows():
            doc_text = f"Product: {row.get('name', 'Unknown')}. "
            doc_text += f"Category: {row.get('category', 'Unknown')}. "
            doc_text += f"Brand: {row.get('brand', 'Unknown')}. "
            doc_text += f"Price: ${row.get('price', 0)}. "
            doc_text += f"Description: {row.get('description', 'No description')}"

            documents.append(doc_text)
            metadatas.append({
                "product_id": str(row.get('product_id', '')),
                "category": row.get('category', 'Unknown'),
                "brand": row.get('brand', 'Unknown'),
                "price": float(row.get('price', 0)),
                "type": "product_info"
            })
            ids.append(f"product_{row.get('product_id', '')}")

        self.chroma_client.add_documents(
            self.product_collection,
            documents,
            metadatas,
            ids
        )

    def add_user_knowledge(self, user_id: str, preferences: Dict, history: List[Dict]) -> None:
        documents = []
        metadatas = []
        ids = []

        pref_text = f"User {user_id} preferences: "
        pref_text += f"Favorite categories: {', '.join(preferences.get('categories', []))}. "
        pref_text += f"Price range: ${preferences.get('min_price', 0)}-${preferences.get('max_price', 1000)}. "
        pref_text += f"Preferred brands: {', '.join(preferences.get('brands', []))}"

        documents.append(pref_text)
        metadatas.append({
            "user_id": user_id,
            "type": "user_preferences",
            "categories": preferences.get('categories', []),
            "price_range": [preferences.get('min_price', 0), preferences.get('max_price', 1000)]
        })
        ids.append(f"user_pref_{user_id}")

        for i, interaction in enumerate(history[-10:]):
            interaction_text = f"User {user_id} interaction: "
            interaction_text += f"Product {interaction.get('product_id')} - "
            interaction_text += f"Action: {interaction.get('action', 'viewed')} - "
            interaction_text += f"Rating: {interaction.get('rating', 'N/A')} - "
            interaction_text += f"Timestamp: {interaction.get('timestamp', 'Unknown')}"

            documents.append(interaction_text)
            metadatas.append({
                "user_id": user_id,
                "product_id": str(interaction.get('product_id', '')),
                "action": interaction.get('action', 'viewed'),
                "rating": interaction.get('rating'),
                "type": "user_interaction"
            })
            ids.append(f"user_interaction_{user_id}_{i}")

        self.chroma_client.add_documents(
            self.user_collection,
            documents,
            metadatas,
            ids
        )

    def search_product_knowledge(self, query: str, n_results: int = 5) -> Dict:
        return self.chroma_client.query_collection(
            self.product_collection,
            [query],
            n_results=n_results
        )

    def search_user_knowledge(self, query: str, user_id: Optional[str] = None, n_results: int = 5) -> Dict:
        where_filter = {"user_id": user_id} if user_id else None

        return self.chroma_client.query_collection(
            self.user_collection,
            [query],
            n_results=n_results,
            where=where_filter
        )

    def get_similar_products(self, product_id: str, n_results: int = 5) -> Dict:
        product_query = f"product_{product_id}"

        return self.chroma_client.query_collection(
            self.product_collection,
            [product_query],
            n_results=n_results
        )

    def get_knowledge_base_stats(self) -> Dict:
        return {
            "product_knowledge": self.chroma_client.get_collection_info(self.product_collection),
            "user_knowledge": self.chroma_client.get_collection_info(self.user_collection),
            "interaction_knowledge": self.chroma_client.get_collection_info(self.interaction_collection)
        }
