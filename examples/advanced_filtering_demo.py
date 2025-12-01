#!/usr/bin/env python3
"""
Advanced Filtering Demo for pyruvector

Demonstrates the advanced filtering capabilities including:
- PayloadIndexManager for optimized queries
- FilterBuilder fluent API
- Geospatial filtering
- Text search
- Complex composite filters
"""

import sys
sys.path.insert(0, "../target/release")

from pyruvector import (
    PayloadIndexManager,
    FilterBuilder,
    FilterEvaluator,
    IndexType,
)


def demo_basic_filtering():
    """Demonstrate basic filter operations"""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Filtering Operations")
    print("=" * 60)

    # Create index manager
    manager = PayloadIndexManager()

    # Create indices for different field types
    print("\n1. Creating indices...")
    manager.create_index("category", IndexType.Keyword)
    manager.create_index("price", IndexType.Float)
    manager.create_index("in_stock", IndexType.Boolean)
    manager.create_index("rating", IndexType.Integer)

    # List indices
    indices = manager.list_indices()
    print(f"   Created indices: {indices}")

    # Index some sample data
    print("\n2. Indexing sample products...")
    products = [
        {
            "id": "prod1",
            "payload": {
                "category": "electronics",
                "price": 299.99,
                "in_stock": True,
                "rating": 4,
                "name": "Laptop"
            }
        },
        {
            "id": "prod2",
            "payload": {
                "category": "electronics",
                "price": 899.99,
                "in_stock": True,
                "rating": 5,
                "name": "High-end Laptop"
            }
        },
        {
            "id": "prod3",
            "payload": {
                "category": "books",
                "price": 29.99,
                "in_stock": False,
                "rating": 3,
                "name": "Python Programming"
            }
        },
        {
            "id": "prod4",
            "payload": {
                "category": "electronics",
                "price": 499.99,
                "in_stock": True,
                "rating": 4,
                "name": "Tablet"
            }
        },
    ]

    for product in products:
        manager.index_payload(product["id"], product["payload"])
        print(f"   Indexed: {product['id']} - {product['payload']['name']}")

    # Create evaluator
    evaluator = FilterEvaluator(manager)

    # Test 1: Simple equality filter
    print("\n3. Simple Equality Filter (category = 'electronics'):")
    filter1 = FilterBuilder().eq("category", "electronics").build()
    results1 = evaluator.evaluate(filter1)
    print(f"   Matching IDs: {results1}")

    # Test 2: Range filter
    print("\n4. Range Filter (100 < price <= 500):")
    filter2 = (
        FilterBuilder()
        .gt("price", 100.0)
        .lte("price", 500.0)
        .build()
    )
    results2 = evaluator.evaluate(filter2)
    print(f"   Matching IDs: {results2}")

    # Test 3: Boolean filter
    print("\n5. Boolean Filter (in_stock = True):")
    filter3 = FilterBuilder().eq("in_stock", True).build()
    results3 = evaluator.evaluate(filter3)
    print(f"   Matching IDs: {results3}")


def demo_composite_filters():
    """Demonstrate composite filter operations (AND/OR/NOT)"""
    print("\n" + "=" * 60)
    print("DEMO 2: Composite Filters (AND/OR/NOT)")
    print("=" * 60)

    # Setup
    manager = PayloadIndexManager()
    manager.create_index("category", IndexType.Keyword)
    manager.create_index("price", IndexType.Float)
    manager.create_index("rating", IndexType.Integer)

    # Index data
    products = [
        {"id": "p1", "payload": {"category": "electronics", "price": 299.99, "rating": 5}},
        {"id": "p2", "payload": {"category": "electronics", "price": 899.99, "rating": 4}},
        {"id": "p3", "payload": {"category": "books", "price": 29.99, "rating": 5}},
        {"id": "p4", "payload": {"category": "clothing", "price": 59.99, "rating": 3}},
    ]

    for p in products:
        manager.index_payload(p["id"], p["payload"])

    evaluator = FilterEvaluator(manager)

    # Test 1: AND filter (electronics AND price < 500)
    print("\n1. AND Filter (category='electronics' AND price < 500):")
    filter1 = (
        FilterBuilder()
        .and_([
            FilterBuilder().eq("category", "electronics"),
            FilterBuilder().lt("price", 500.0)
        ])
        .build()
    )
    results1 = evaluator.evaluate(filter1)
    print(f"   Matching IDs: {results1}")

    # Test 2: OR filter (electronics OR books)
    print("\n2. OR Filter (category='electronics' OR category='books'):")
    filter2 = (
        FilterBuilder()
        .or_([
            FilterBuilder().eq("category", "electronics"),
            FilterBuilder().eq("category", "books")
        ])
        .build()
    )
    results2 = evaluator.evaluate(filter2)
    print(f"   Matching IDs: {results2}")

    # Test 3: NOT filter (NOT books)
    print("\n3. NOT Filter (NOT category='books'):")
    filter3 = (
        FilterBuilder()
        .not_(FilterBuilder().eq("category", "books"))
        .build()
    )
    results3 = evaluator.evaluate(filter3)
    print(f"   Matching IDs: {results3}")

    # Test 4: Complex nested filter
    print("\n4. Complex Filter ((rating >= 4) AND (electronics OR books)):")
    filter4 = (
        FilterBuilder()
        .and_([
            FilterBuilder().gte("rating", 4),
            FilterBuilder().or_([
                FilterBuilder().eq("category", "electronics"),
                FilterBuilder().eq("category", "books")
            ])
        ])
        .build()
    )
    results4 = evaluator.evaluate(filter4)
    print(f"   Matching IDs: {results4}")


def demo_in_values_filter():
    """Demonstrate IN values filter"""
    print("\n" + "=" * 60)
    print("DEMO 3: IN Values Filter")
    print("=" * 60)

    manager = PayloadIndexManager()
    manager.create_index("status", IndexType.Keyword)

    statuses = [
        {"id": "order1", "payload": {"status": "pending"}},
        {"id": "order2", "payload": {"status": "shipped"}},
        {"id": "order3", "payload": {"status": "delivered"}},
        {"id": "order4", "payload": {"status": "cancelled"}},
    ]

    for s in statuses:
        manager.index_payload(s["id"], s["payload"])

    evaluator = FilterEvaluator(manager)

    print("\n1. IN Filter (status IN ['pending', 'shipped']):")
    filter1 = (
        FilterBuilder()
        .in_values("status", ["pending", "shipped"])
        .build()
    )
    results1 = evaluator.evaluate(filter1)
    print(f"   Matching IDs: {results1}")


def demo_geospatial_filter():
    """Demonstrate geospatial filtering"""
    print("\n" + "=" * 60)
    print("DEMO 4: Geospatial Filtering")
    print("=" * 60)

    # Create filter with geospatial query
    print("\n1. Creating geo-radius filter...")
    print("   Center: New York City (40.7128°N, 74.0060°W)")
    print("   Radius: 50 km")

    filter_builder = FilterBuilder()
    geo_filter = filter_builder.geo_radius(
        "location",
        lat=40.7128,
        lon=-74.0060,
        radius_km=50.0
    )

    # Build the filter
    filter_dict = geo_filter.build()
    print(f"\n2. Built filter structure: {filter_dict}")
    print("   (Ready to use with VectorDB.search())")


def demo_text_search_filter():
    """Demonstrate text search filtering"""
    print("\n" + "=" * 60)
    print("DEMO 5: Text Search Filtering")
    print("=" * 60)

    # Create text search filter
    print("\n1. Creating text search filter...")
    print("   Query: 'machine learning'")

    filter_builder = FilterBuilder()
    text_filter = filter_builder.text_match(
        "description",
        "machine learning"
    )

    # Build the filter
    filter_dict = text_filter.build()
    print(f"\n2. Built filter structure: {filter_dict}")
    print("   (Ready to use with VectorDB.search())")


def demo_filter_builder_chaining():
    """Demonstrate FilterBuilder method chaining"""
    print("\n" + "=" * 60)
    print("DEMO 6: FilterBuilder Method Chaining")
    print("=" * 60)

    print("\n1. Building complex filter with method chaining:")
    print("   (price >= 100 AND price <= 1000)")

    # Method chaining
    filter_builder = (
        FilterBuilder()
        .gte("price", 100.0)
        .lte("price", 1000.0)
    )

    filter_dict = filter_builder.build()
    print(f"\n2. Built filter: {filter_dict}")

    # Test with evaluator
    manager = PayloadIndexManager()
    manager.create_index("price", IndexType.Float)

    test_products = [
        {"id": "p1", "payload": {"price": 50.0}},
        {"id": "p2", "payload": {"price": 250.0}},
        {"id": "p3", "payload": {"price": 750.0}},
        {"id": "p4", "payload": {"price": 1500.0}},
    ]

    for p in test_products:
        manager.index_payload(p["id"], p["payload"])

    evaluator = FilterEvaluator(manager)
    results = evaluator.evaluate(filter_dict)
    print(f"\n3. Matching products (price 100-1000): {results}")


def demo_index_management():
    """Demonstrate index management operations"""
    print("\n" + "=" * 60)
    print("DEMO 7: Index Management")
    print("=" * 60)

    manager = PayloadIndexManager()

    # Create multiple indices
    print("\n1. Creating multiple indices...")
    indices_to_create = [
        ("name", IndexType.Keyword),
        ("age", IndexType.Integer),
        ("salary", IndexType.Float),
        ("active", IndexType.Boolean),
    ]

    for field, index_type in indices_to_create:
        manager.create_index(field, index_type)
        print(f"   Created {index_type} index on '{field}'")

    # List all indices
    print("\n2. Listing all indices:")
    all_indices = manager.list_indices()
    print(f"   Indices: {all_indices}")

    # Drop an index
    print("\n3. Dropping 'salary' index...")
    manager.drop_index("salary")
    remaining_indices = manager.list_indices()
    print(f"   Remaining indices: {remaining_indices}")

    # Try to create duplicate index (will fail)
    print("\n4. Attempting to create duplicate index on 'name'...")
    try:
        manager.create_index("name", IndexType.Keyword)
    except Exception as e:
        print(f"   Expected error: {e}")


def demo_payload_operations():
    """Demonstrate payload indexing and retrieval"""
    print("\n" + "=" * 60)
    print("DEMO 8: Payload Operations")
    print("=" * 60)

    manager = PayloadIndexManager()
    manager.create_index("category", IndexType.Keyword)

    # Index payload
    print("\n1. Indexing payload for 'vec1'...")
    payload = {
        "category": "science",
        "title": "Quantum Physics",
        "year": 2024,
        "tags": ["physics", "quantum"]
    }
    manager.index_payload("vec1", payload)
    print(f"   Payload: {payload}")

    # Retrieve payload
    print("\n2. Retrieving payload for 'vec1'...")
    retrieved = manager.get_payload("vec1")
    print(f"   Retrieved: {retrieved}")

    # Remove payload
    print("\n3. Removing payload for 'vec1'...")
    manager.remove_payload("vec1")

    # Try to retrieve after removal
    print("\n4. Attempting to retrieve removed payload...")
    removed = manager.get_payload("vec1")
    print(f"   Result: {removed}")


def demo_real_world_scenario():
    """Demonstrate a real-world e-commerce filtering scenario"""
    print("\n" + "=" * 60)
    print("DEMO 9: Real-World E-Commerce Scenario")
    print("=" * 60)

    # Setup
    manager = PayloadIndexManager()
    manager.create_index("category", IndexType.Keyword)
    manager.create_index("price", IndexType.Float)
    manager.create_index("rating", IndexType.Integer)
    manager.create_index("in_stock", IndexType.Boolean)
    manager.create_index("brand", IndexType.Keyword)

    # Product catalog
    print("\n1. Indexing product catalog...")
    products = [
        {
            "id": "laptop-001",
            "payload": {
                "category": "electronics",
                "brand": "TechCo",
                "price": 1299.99,
                "rating": 5,
                "in_stock": True,
                "name": "Pro Laptop 15"
            }
        },
        {
            "id": "laptop-002",
            "payload": {
                "category": "electronics",
                "brand": "CompuMax",
                "price": 899.99,
                "rating": 4,
                "in_stock": True,
                "name": "Business Laptop"
            }
        },
        {
            "id": "phone-001",
            "payload": {
                "category": "electronics",
                "brand": "PhonePlus",
                "price": 799.99,
                "rating": 5,
                "in_stock": False,
                "name": "Smartphone X"
            }
        },
        {
            "id": "book-001",
            "payload": {
                "category": "books",
                "brand": "TechPress",
                "price": 49.99,
                "rating": 5,
                "in_stock": True,
                "name": "Python Mastery"
            }
        },
    ]

    for product in products:
        manager.index_payload(product["id"], product["payload"])
        print(f"   {product['id']}: {product['payload']['name']}")

    evaluator = FilterEvaluator(manager)

    # Query 1: High-rated electronics in stock
    print("\n2. Query: High-rated electronics in stock (rating >= 4)")
    query1 = (
        FilterBuilder()
        .and_([
            FilterBuilder().eq("category", "electronics"),
            FilterBuilder().gte("rating", 4),
            FilterBuilder().eq("in_stock", True)
        ])
        .build()
    )
    results1 = evaluator.evaluate(query1)
    print(f"   Results: {results1}")

    # Query 2: Affordable products (price < 1000) with top ratings
    print("\n3. Query: Affordable products (price < 1000) with rating 5")
    query2 = (
        FilterBuilder()
        .and_([
            FilterBuilder().lt("price", 1000.0),
            FilterBuilder().eq("rating", 5)
        ])
        .build()
    )
    results2 = evaluator.evaluate(query2)
    print(f"   Results: {results2}")

    # Query 3: Premium electronics (price > 800) OR books
    print("\n4. Query: Premium electronics (price > 800) OR books")
    query3 = (
        FilterBuilder()
        .or_([
            FilterBuilder().and_([
                FilterBuilder().eq("category", "electronics"),
                FilterBuilder().gt("price", 800.0)
            ]),
            FilterBuilder().eq("category", "books")
        ])
        .build()
    )
    results3 = evaluator.evaluate(query3)
    print(f"   Results: {results3}")


def main():
    """Run all demos"""
    print("\n")
    print("=" * 60)
    print(" PyRuVector Advanced Filtering Demo")
    print("=" * 60)

    try:
        demo_basic_filtering()
        demo_composite_filters()
        demo_in_values_filter()
        demo_geospatial_filter()
        demo_text_search_filter()
        demo_filter_builder_chaining()
        demo_index_management()
        demo_payload_operations()
        demo_real_world_scenario()

        print("\n" + "=" * 60)
        print(" All demos completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
