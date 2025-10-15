import sys
import os
import logging
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.feature_store import (
    FeatureStore,
    FeatureStoreConfig,
    FeatureType,
    FeatureGroup
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_basic_feature_operations():
    print("\n" + "="*60)
    print("DEMO: Basic Feature Store Operations")
    print("="*60)

    config = FeatureStoreConfig(
        redis_host="localhost",
        redis_port=6379,
        default_ttl=3600,
        enable_versioning=True
    )

    try:
        fs = FeatureStore(config)
        print("✅ Feature Store initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Feature Store: {e}")
        print("💡 Make sure Redis is running: docker run -d -p 6379:6379 redis:7-alpine")
        return False

    product_data = {
        "product_id": 123,
        "name": "iPhone 15 Pro",
        "category": "smartphones",
        "description": "Latest iPhone with advanced camera system and A17 Pro chip",
        "brand": "Apple",
        "price": 999.99,
        "text_embedding": [0.1, 0.2, 0.3] * 128,  # 384-dim embedding
        "image_embedding": [0.4, 0.5, 0.6] * 170   # 512-dim embedding (approximate)
    }

    print("\n1. Writing features to store...")
    success = fs.write_features(
        entity_id="product_123",
        features=product_data,
        feature_group="product_basic"
    )

    if success:
        print("✅ Features written successfully")
    else:
        print("❌ Failed to write features")
        return False

    print("\n2. Reading features from store...")
    retrieved_features = fs.get_features(
        entity_id="product_123",
        feature_names=["name", "category", "price", "text_embedding"]
    )

    if retrieved_features:
        print("✅ Features retrieved successfully:")
        for name, value in retrieved_features.items():
            if isinstance(value, list) and len(value) > 10:
                print(f"   {name}: [embedding with {len(value)} dimensions]")
            else:
                print(f"   {name}: {value}")
    else:
        print("❌ Failed to retrieve features")

    print("\n3. Testing batch operations...")

    products = [
        {
            "entity_id": "product_124",
            "features": {
                "product_id": 124,
                "name": "Samsung Galaxy S24",
                "category": "smartphones",
                "price": 799.99,
                "text_embedding": [0.2, 0.3, 0.4] * 128
            }
        },
        {
            "entity_id": "product_125",
            "features": {
                "product_id": 125,
                "name": "MacBook Pro M3",
                "category": "laptops",
                "price": 1999.99,
                "text_embedding": [0.3, 0.4, 0.5] * 128
            }
        }
    ]

    for product in products:
        fs.write_features(product["entity_id"], product["features"])

    batch_features = fs.get_features_batch(
        entity_ids=["product_123", "product_124", "product_125"],
        feature_names=["name", "category", "price"]
    )

    print("✅ Batch retrieved features:")
    for entity_id, features in batch_features.items():
        print(f"   {entity_id}: {features}")

    print("\n4. Testing version management...")

    updated_features = product_data.copy()
    updated_features["price"] = 899.99  # Price drop
    updated_features["description"] = "iPhone 15 Pro - Now with reduced price!"

    fs.write_features("product_123", updated_features)

    versions = fs.list_feature_versions("product_123", "price")
    print(f"✅ Price feature versions: {versions}")

    lineage = fs.get_feature_lineage("product_123", "price")
    print(f"✅ Price feature lineage: {lineage}")

    print("\n5. Performance metrics...")
    metrics = fs.get_performance_metrics()
    print("✅ Performance metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value}")

    print("\n6. Health check...")
    health = fs.health_check()
    print(f"✅ Health status: {health['status']}")
    print(f"   Redis connected: {health['redis_connected']}")

    return True

def demo_advanced_features():
    print("\n" + "="*60)
    print("DEMO: Advanced Feature Store Capabilities")
    print("="*60)

    config = FeatureStoreConfig(enable_versioning=True, enable_monitoring=True)

    try:
        fs = FeatureStore(config)
    except Exception as e:
        print(f"❌ Failed to initialize Feature Store: {e}")
        return False

    print("\n1. Testing feature validation...")

    valid_features = {
        "product_id": 456,
        "name": "Valid Product",
        "category": "electronics",
        "price": 299.99
    }

    success = fs.write_features("product_456", valid_features)
    print(f"✅ Valid features written: {success}")

    invalid_features = {
        "product_id": "not_a_number",  # Should be numeric
        "name": None,  # Should not be null if not nullable
        "price": -100   # Might violate constraints
    }

    success = fs.write_features("product_invalid", invalid_features)
    print(f"📝 Invalid features result: {success}")

    print("\n2. Testing feature computation...")

    raw_product_data = {
        "id": 789,
        "name": "Wireless Headphones",
        "description": "Premium noise-canceling wireless headphones with 30-hour battery life",
        "category": "audio",
        "brand": "Sony",
        "price": 249.99
    }

    computed_success = fs.compute_and_store_features(
        entity_id="product_789",
        raw_data=raw_product_data
    )
    print(f"✅ Feature computation success: {computed_success}")

    computed_features = fs.get_features(
        entity_id="product_789",
        feature_names=["name", "description_length", "text_quality_score", "feature_completeness"]
    )
    print("✅ Computed features:")
    for name, value in computed_features.items():
        print(f"   {name}: {value}")

    print("\n3. Testing feature export...")

    entity_ids = ["product_123", "product_456", "product_789"]
    feature_names = ["name", "category", "price"]

    try:
        df = fs.export_features(entity_ids, feature_names, format="pandas")
        print("✅ Exported features as DataFrame:")
        print(df.to_string())
    except Exception as e:
        print(f"📝 DataFrame export error: {e}")

    json_data = fs.export_features(entity_ids, feature_names, format="json")
    print("\n✅ Exported features as JSON:")
    print(json_data[:200] + "..." if len(json_data) > 200 else json_data)

    return True

def demo_integration_example():
    print("\n" + "="*60)
    print("DEMO: ML Workflow Integration Example")
    print("="*60)

    config = FeatureStoreConfig()

    try:
        fs = FeatureStore(config)
    except Exception as e:
        print(f"❌ Failed to initialize Feature Store: {e}")
        return False

    print("\n1. Preparing training data...")

    training_entities = [f"product_{i}" for i in range(100, 110)]

    for i, entity_id in enumerate(training_entities):
        features = {
            "product_id": 100 + i,
            "name": f"Product {100 + i}",
            "category": ["electronics", "clothing", "books"][i % 3],
            "price": 50.0 + i * 10,
            "text_embedding": [0.1 * i, 0.2 * i, 0.3 * i] * 128,
            "popularity_score": min(i / 10.0, 1.0)
        }

        fs.write_features(entity_id, features)

    print(f"✅ Created features for {len(training_entities)} training entities")

    print("\n2. Extracting features for model training...")

    training_features = [
        "category", "price", "text_embedding", "popularity_score"
    ]

    training_data = fs.get_features_batch(
        entity_ids=training_entities,
        feature_names=training_features
    )

    print(f"✅ Extracted training data for {len(training_data)} entities")

    print("\n3. Simulating real-time inference...")

    inference_entity = "product_105"
    inference_features = ["category", "price", "text_embedding"]

    start_time = datetime.now()
    features = fs.get_features(inference_entity, inference_features)
    inference_time = (datetime.now() - start_time).total_seconds() * 1000

    print(f"✅ Real-time feature retrieval took {inference_time:.2f}ms")
    print(f"   Retrieved features: {list(features.keys())}")

    print("\n4. Feature monitoring and alerting...")

    metrics = fs.get_performance_metrics()

    if metrics["cache_hit_rate"] < 0.8:
        print("🚨 ALERT: Low cache hit rate detected!")
    else:
        print("✅ Cache performance is good")

    if metrics["average_computation_time"] > 0.1:  # 100ms threshold
        print("🚨 ALERT: High feature computation time!")
    else:
        print("✅ Feature computation time is acceptable")

    return True

def main():
    print("🚀 Feature Store Demo and Testing")
    print("=" * 80)

    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis connection verified")
    except Exception as e:
        print(f"❌ Redis not available: {e}")
        print("💡 Start Redis with: docker run -d -p 6379:6379 redis:7-alpine")
        return

    demos = [
        demo_basic_feature_operations,
        demo_advanced_features,
        demo_integration_example
    ]

    results = []
    for demo in demos:
        try:
            result = demo()
            results.append(result)
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            results.append(False)

    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)

    success_count = sum(results)
    total_demos = len(results)

    print(f"✅ Successful demos: {success_count}/{total_demos}")

    if success_count == total_demos:
        print("🎉 All Feature Store demos completed successfully!")
        print("\n📚 Feature Store is ready for production use:")
        print("   - High-performance feature serving with Redis")
        print("   - Feature versioning and lineage tracking")
        print("   - Automated feature computation")
        print("   - Production monitoring and validation")
        print("   - ML workflow integration")
    else:
        print("⚠️  Some demos failed. Check the error messages above.")

if __name__ == "__main__":
    main()
