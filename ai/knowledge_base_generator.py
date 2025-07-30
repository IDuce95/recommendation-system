import pandas as pd
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
from ai.chromadb_client import ChromaDBClient, KnowledgeBaseManager

class FakeKnowledgeBaseGenerator:
    def __init__(self):
        self.categories = [
            "Electronics", "Laptops", "Smartphones", "Tablets",
            "Headphones", "Smartwatches", "Gaming", "Accessories"
        ]

        self.brands = [
            "Apple", "Samsung", "Sony", "Microsoft", "Dell",
            "HP", "Lenovo", "Google", "Amazon", "Xiaomi"
        ]

        self.product_names = {
            "Laptops": ["MacBook Pro", "ThinkPad", "XPS", "Surface Laptop", "Pavilion"],
            "Smartphones": ["iPhone", "Galaxy", "Pixel", "Xperia", "Mi"],
            "Tablets": ["iPad", "Galaxy Tab", "Surface Pro", "Fire Tablet"],
            "Headphones": ["AirPods", "WH-1000XM", "QuietComfort", "Studio"],
            "Smartwatches": ["Apple Watch", "Galaxy Watch", "Fitbit", "Garmin"]
        }

        self.actions = ["viewed", "clicked", "purchased", "added_to_cart", "rated", "reviewed"]

    def generate_fake_products(self, num_products: int = 100) -> pd.DataFrame:
        products = []

        for i in range(num_products):
            category = random.choice(self.categories)
            brand = random.choice(self.brands)

            if category in self.product_names:
                base_name = random.choice(self.product_names[category])
                name = f"{brand} {base_name} {random.randint(1, 20)}"
            else:
                name = f"{brand} {category} {random.randint(1, 20)}"

            price = round(random.uniform(50, 2000), 2)

            descriptions = [
                f"High-quality {category.lower()} from {brand}",
                f"Latest {category.lower()} with advanced features",
                f"Premium {category.lower()} for professionals",
                f"Affordable {category.lower()} with great performance",
                f"Innovative {category.lower()} with cutting-edge technology"
            ]

            description = random.choice(descriptions)

            products.append({
                "product_id": f"prod_{i+1:04d}",
                "name": name,
                "category": category,
                "brand": brand,
                "price": price,
                "description": description,
                "rating": round(random.uniform(3.0, 5.0), 1),
                "reviews_count": random.randint(10, 500),
                "in_stock": random.choice([True, True, True, False]),
                "created_at": datetime.now() - timedelta(days=random.randint(1, 365))
            })

        return pd.DataFrame(products)

    def generate_fake_users(self, num_users: int = 50) -> List[Dict]:
        users = []

        for i in range(num_users):
            user_preferences = {
                "user_id": f"user_{i+1:04d}",
                "categories": random.sample(self.categories, random.randint(1, 3)),
                "brands": random.sample(self.brands, random.randint(1, 2)),
                "min_price": random.choice([0, 50, 100, 200]),
                "max_price": random.choice([500, 1000, 1500, 2000]),
                "age_group": random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]),
                "location": random.choice(["US", "EU", "Asia", "Other"])
            }
            users.append(user_preferences)

        return users

    def generate_fake_interactions(self, users: List[Dict], products: pd.DataFrame, num_interactions: int = 1000) -> List[Dict]:
        interactions = []

        for _ in range(num_interactions):
            user = random.choice(users)
            product = products.sample(1).iloc[0]

            interaction = {
                "user_id": user["user_id"],
                "product_id": product["product_id"],
                "action": random.choice(self.actions),
                "timestamp": datetime.now() - timedelta(days=random.randint(1, 30)),
                "rating": random.choice([None, None, random.randint(1, 5)]),
                "session_id": f"session_{random.randint(1000, 9999)}"
            }

            interactions.append(interaction)

        return interactions

    def generate_knowledge_snippets(self, products: pd.DataFrame) -> List[Dict]:
        snippets = []

        for _, product in products.sample(20).iterrows():

            feature_snippet = {
                "id": f"feature_{product['product_id']}",
                "content": f"The {product['name']} features advanced technology and premium build quality. Popular in {product['category']} category.",
                "type": "product_features",
                "product_id": product['product_id'],
                "relevance_score": random.uniform(0.7, 1.0)
            }
            snippets.append(feature_snippet)

            comparison_snippet = {
                "id": f"comparison_{product['product_id']}",
                "content": f"Compared to other {product['category']} products, {product['name']} offers excellent value at ${product['price']}.",
                "type": "product_comparison",
                "product_id": product['product_id'],
                "relevance_score": random.uniform(0.6, 0.9)
            }
            snippets.append(comparison_snippet)

            review_snippet = {
                "id": f"review_{product['product_id']}",
                "content": f"Users love the {product['name']} for its reliability and performance. Average rating: {product['rating']}/5.",
                "type": "user_reviews",
                "product_id": product['product_id'],
                "relevance_score": random.uniform(0.8, 1.0)
            }
            snippets.append(review_snippet)

        return snippets

    def populate_chromadb(self, chroma_client: ChromaDBClient, num_products: int = 100, num_users: int = 50) -> Dict:
        print("Generating fake knowledge base...")

        kb_manager = KnowledgeBaseManager(chroma_client)
        kb_manager.initialize_knowledge_base()

        products_df = self.generate_fake_products(num_products)
        print(f"Generated {len(products_df)} fake products")

        users = self.generate_fake_users(num_users)
        print(f"Generated {len(users)} fake users")

        interactions = self.generate_fake_interactions(users, products_df, num_interactions=1000)
        print(f"Generated {len(interactions)} fake interactions")

        kb_manager.add_product_knowledge(products_df)
        print("Added product knowledge to ChromaDB")

        for user in users[:10]:
            user_interactions = [i for i in interactions if i['user_id'] == user['user_id']]
            kb_manager.add_user_knowledge(user['user_id'], user, user_interactions)
        print("Added user knowledge to ChromaDB")

        knowledge_snippets = self.generate_knowledge_snippets(products_df)

        docs = [snippet['content'] for snippet in knowledge_snippets]
        metas = [{k: v for k, v in snippet.items() if k != 'content'} for snippet in knowledge_snippets]
        ids = [snippet['id'] for snippet in knowledge_snippets]

        chroma_client.add_documents("knowledge_snippets", docs, metas, ids)
        print("Added knowledge snippets to ChromaDB")

        stats = kb_manager.get_knowledge_base_stats()
        print(f"Knowledge base populated successfully: {stats}")

        return {
            "products": len(products_df),
            "users": len(users),
            "interactions": len(interactions),
            "knowledge_snippets": len(knowledge_snippets),
            "collections": stats
        }

def setup_fake_knowledge_base():
    try:
        chroma_client = ChromaDBClient(host="localhost", port=8001)
        generator = FakeKnowledgeBaseGenerator()

        result = generator.populate_chromadb(chroma_client, num_products=200, num_users=100)

        print("✅ Fake knowledge base setup completed!")
        print(f"📊 Statistics: {result}")

        return result

    except Exception as e:
        print(f"❌ Error setting up knowledge base: {e}")
        return None

if __name__ == "__main__":
    setup_fake_knowledge_base()
