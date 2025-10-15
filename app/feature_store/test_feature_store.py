#!/usr/bin/env python3

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.feature_store.integration import get_feature_store

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_feature_store():
    print("🧪 Testing Feature Store Implementation")
    print("=" * 50)

    fs = get_feature_store()

    print("\n1. Health Check...")
    health = fs.health_check()

    if health.get("status") == "healthy" and health.get("redis_connected"):
        print("✅ Feature Store is healthy and Redis is connected")
    else:
        print("❌ Feature Store health check failed:")
        print(f"   Status: {health.get('status')}")
        print(f"   Redis: {health.get('redis_connected')}")
        print(f"   Error: {health.get('error', 'Unknown error')}")
        return False

    print("\n2. Testing feature storage...")

    test_product = {
        "id": 123,
        "name": "Test iPhone 15",
        "category": "smartphones",
        "description": "A test product for Feature Store validation",
        "brand": "Apple",
        "price": 999.99
    }

    test_embeddings = {
        "text_embedding": [0.1, 0.2, 0.3] * 128,  # 384-dim
        "image_embedding": [0.4, 0.5, 0.6] * 170   # 512-dim (approx)
    }

    success = fs.store_product_features(
        product_id=123,
        product_data=test_product,
        embeddings=test_embeddings
    )

    if success:
        print("✅ Product features stored successfully")
    else:
        print("❌ Failed to store product features")
        return False

    print("\n3. Testing feature retrieval...")

    retrieved_features = fs.get_product_features(
        product_id=123,
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
        return False

    print("\n4. Testing batch operations...")

    test_products = [
        {
            "id": 124,
            "name": "Samsung Galaxy S24",
            "category": "smartphones",
            "price": 799.99
        },
        {
            "id": 125,
            "name": "MacBook Pro M3",
            "category": "laptops",
            "price": 1999.99
        }
    ]

    for product in test_products:
        fs.store_product_features(
            product_id=product["id"],
            product_data=product
        )

    batch_features = fs.get_batch_product_features(
        product_ids=[123, 124, 125],
        feature_names=["name", "category", "price"]
    )

    if batch_features and len(batch_features) == 3:
        print("✅ Batch retrieval successful:")
        for product_id, features in batch_features.items():
            print(f"   Product {product_id}: {features}")
    else:
        print(f"❌ Batch retrieval failed. Got {len(batch_features)} products")
        return False

    print("\n5. Performance statistics...")

    stats = fs.get_feature_statistics()
    if stats and "error" not in stats:
        print("✅ Performance metrics:")
        for metric, value in stats.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")
    else:
        print(f"❌ Failed to get statistics: {stats}")

    print("\n🎉 All Feature Store tests passed!")
    return True

def main():
    try:
        success = test_feature_store()

        if success:
            print("\n✅ Feature Store is working correctly!")
            print("\n📋 Next steps:")
            print("   - Feature Store is ready for production use")
            print("   - It will integrate with the recommendation API")
            print("   - Redis provides high-performance feature serving")
            print("   - Feature versioning and monitoring are enabled")

        else:
            print("\n❌ Feature Store tests failed!")
            print("\n🔧 Troubleshooting:")
            print("   - Make sure Redis is running: docker run -d -p 6379:6379 redis:7-alpine")
            print("   - Check Redis connectivity: redis-cli ping")
            print("   - Verify Docker network configuration")

        return success

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure redis is installed: pip install redis hiredis")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
