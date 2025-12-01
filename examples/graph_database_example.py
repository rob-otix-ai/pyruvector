#!/usr/bin/env python3
"""
Graph Database Example for pyruvector

Demonstrates the graph database functionality including:
- Creating nodes and edges
- Graph traversal and algorithms
- ACID transactions
- Building real-world graph structures
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pyruvector import (
        GraphDB,
        Node,  # noqa: F401
        Edge,  # noqa: F401
        Transaction,  # noqa: F401
        IsolationLevel,  # noqa: F401
    )
except ImportError:
    print("Error: pyruvector not built. Run 'maturin develop' first.")
    sys.exit(1)


def example_basic_graph():
    """Example 1: Basic graph operations"""
    print("=" * 70)
    print("Example 1: Basic Graph Operations")
    print("=" * 70)

    # Create graph database
    graph = GraphDB()
    print(f"Created: {graph}\n")

    # Create nodes
    print("Creating nodes...")
    alice = graph.create_node("Person", {
        "name": "Alice",
        "age": 30,
        "occupation": "Engineer"
    })
    print(f"  Created node: {alice}")

    bob = graph.create_node("Person", {
        "name": "Bob",
        "age": 25,
        "occupation": "Designer"
    })
    print(f"  Created node: {bob}")

    # Retrieve and display node
    alice_node = graph.get_node(alice)
    print(f"\nRetrieved node: {alice_node}")
    print(f"  Properties: {alice_node.get_properties()}")

    # Create edge
    print("\nCreating edge...")
    edge_id = graph.create_edge(alice, bob, "KNOWS", {
        "since": 2020,
        "relationship": "colleagues"
    })
    print(f"  Created edge: {edge_id}")

    # Retrieve and display edge
    edge = graph.get_edge(edge_id)
    print(f"\nRetrieved edge: {edge}")
    print(f"  Properties: {edge.get_properties()}")

    # Show stats
    stats = graph.stats()
    print(f"\nGraph stats: {stats}")


def example_social_network():
    """Example 2: Social network graph"""
    print("\n" + "=" * 70)
    print("Example 2: Social Network Graph")
    print("=" * 70)

    graph = GraphDB()

    # Create users
    print("Building social network...")
    users = {}
    user_data = [
        ("Alice", 30, "NYC", ["Python", "Rust"]),
        ("Bob", 25, "SF", ["JavaScript", "Go"]),
        ("Charlie", 35, "LA", ["Python", "Java"]),
        ("Dave", 28, "NYC", ["Rust", "C++"]),
        ("Eve", 32, "SF", ["Python", "JavaScript"]),
    ]

    for name, age, city, skills in user_data:
        user_id = graph.create_node("User", {
            "name": name,
            "age": age,
            "city": city,
            "skills": skills,
            "active": True
        })
        users[name] = user_id
        print(f"  Created user: {name}")

    # Create friendships
    print("\nCreating friendships...")
    friendships = [
        ("Alice", "Bob", 0.9, "work"),
        ("Alice", "Charlie", 0.7, "college"),
        ("Bob", "Dave", 0.8, "hobby"),
        ("Charlie", "Eve", 0.6, "conference"),
        ("Dave", "Eve", 0.5, "meetup"),
        ("Alice", "Dave", 0.85, "neighbors"),
    ]

    for source, target, strength, context in friendships:
        graph.create_edge(users[source], users[target], "FRIENDS_WITH", {
            "strength": strength,
            "context": context,
            "mutual": True
        })
        print(f"  {source} -> {target} ({context})")

    stats = graph.stats()
    print(f"\nNetwork stats: {stats}")

    # Find Alice's friends
    print("\nFinding Alice's network (depth 1)...")
    neighbors = graph.find_neighbors(users["Alice"], depth=1)
    for neighbor in neighbors:
        print(f"  - {neighbor.get_property('name')} ({neighbor.get_property('city')})")

    # Find extended network (depth 2)
    print("\nFinding Alice's extended network (depth 2)...")
    extended = graph.find_neighbors(users["Alice"], depth=2)
    print(f"  Found {len(extended)} people in extended network")

    # Find path
    print("\nFinding shortest path from Alice to Eve...")
    path = graph.shortest_path(users["Alice"], users["Eve"])
    if path:
        print(f"  Path length: {len(path)} nodes")
        path_names = []
        for node_id in path:
            node = graph.get_node(node_id)
            path_names.append(node.get_property("name"))
        print(f"  Path: {' -> '.join(path_names)}")
    else:
        print("  No path found")


def example_knowledge_graph():
    """Example 3: Knowledge graph"""
    print("\n" + "=" * 70)
    print("Example 3: Knowledge Graph")
    print("=" * 70)

    graph = GraphDB()

    print("Building knowledge graph...")

    # Programming languages
    python = graph.create_node("Language", {
        "name": "Python",
        "year": 1991,
        "paradigm": "multi-paradigm"
    })

    rust = graph.create_node("Language", {
        "name": "Rust",
        "year": 2010,
        "paradigm": "systems"
    })

    javascript = graph.create_node("Language", {
        "name": "JavaScript",
        "year": 1995,
        "paradigm": "multi-paradigm"
    })

    # Domains
    ml = graph.create_node("Domain", {"name": "Machine Learning"})
    web = graph.create_node("Domain", {"name": "Web Development"})
    systems = graph.create_node("Domain", {"name": "Systems Programming"})

    # Libraries
    pytorch = graph.create_node("Library", {
        "name": "PyTorch",
        "stars": 70000
    })

    react = graph.create_node("Library", {
        "name": "React",
        "stars": 200000
    })

    # Create relationships
    print("\nCreating relationships...")
    relationships = [
        (python, ml, "USED_FOR", {"primary": True}),
        (rust, systems, "USED_FOR", {"primary": True}),
        (javascript, web, "USED_FOR", {"primary": True}),
        (python, web, "USED_FOR", {"primary": False}),
        (pytorch, python, "WRITTEN_IN", {}),
        (pytorch, ml, "TARGETS", {}),
        (react, javascript, "WRITTEN_IN", {}),
        (react, web, "TARGETS", {}),
    ]

    for source, target, rel_type, props in relationships:
        graph.create_edge(source, target, rel_type, props)

    stats = graph.stats()
    print(f"Knowledge graph stats: {stats}")

    # Query connections
    print("\nFinding what Python is used for...")
    neighbors = graph.find_neighbors(python, depth=1)
    for neighbor in neighbors:
        print(f"  - {neighbor.label}: {neighbor.get_property('name')}")


def example_transactions():
    """Example 4: ACID transactions"""
    print("\n" + "=" * 70)
    print("Example 4: ACID Transactions")
    print("=" * 70)

    graph = GraphDB()

    # Successful transaction
    print("Example 4a: Successful transaction commit")
    tx = graph.begin_transaction(isolation_level=IsolationLevel.Serializable)

    print("  Creating nodes in transaction...")
    node1 = tx.create_node("Account", {"balance": 1000, "name": "Alice"})
    node2 = tx.create_node("Account", {"balance": 500, "name": "Bob"})

    print("  Creating edge in transaction...")
    tx.create_edge(node1, node2, "TRANSFER", {"amount": 100})

    print("  Committing transaction...")
    success = tx.commit()
    print(f"  Transaction committed: {success}")

    stats = graph.stats()
    print(f"  Graph stats after commit: {stats}")

    # Rollback transaction
    print("\nExample 4b: Transaction rollback")
    tx2 = graph.begin_transaction()

    print("  Creating nodes in transaction...")
    temp_node = tx2.create_node("TempAccount", {"balance": 9999})

    print("  Rolling back transaction...")
    tx2.rollback()
    print("  Transaction rolled back")

    # Verify node doesn't exist
    node = graph.get_node(temp_node)
    print(f"  Node exists after rollback: {node is not None}")

    stats = graph.stats()
    print(f"  Graph stats after rollback: {stats}")


def example_batch_operations():
    """Example 5: Batch operations with transactions"""
    print("\n" + "=" * 70)
    print("Example 5: Batch Operations")
    print("=" * 70)

    graph = GraphDB()

    print("Creating 100 nodes in a transaction...")
    tx = graph.begin_transaction()

    node_ids = []
    for i in range(100):
        node_id = tx.create_node("Item", {
            "id": i,
            "value": i * 10,
            "category": "batch"
        })
        node_ids.append(node_id)

    print("Creating 99 sequential edges...")
    for i in range(len(node_ids) - 1):
        tx.create_edge(node_ids[i], node_ids[i + 1], "NEXT", {"index": i})

    print("Committing batch transaction...")
    success = tx.commit()
    print(f"Transaction committed: {success}")

    stats = graph.stats()
    print(f"Final graph stats: {stats}")

    # Verify first and last nodes
    first_node = graph.get_node(node_ids[0])
    last_node = graph.get_node(node_ids[-1])
    print(f"\nFirst node value: {first_node.get_property('value')}")
    print(f"Last node value: {last_node.get_property('value')}")

    # Find path from first to last
    print("\nFinding path from first to last node...")
    path = graph.shortest_path(node_ids[0], node_ids[-1])
    print(f"Path length: {len(path)} nodes")


def example_graph_algorithms():
    """Example 6: Graph algorithms"""
    print("\n" + "=" * 70)
    print("Example 6: Graph Algorithms")
    print("=" * 70)

    graph = GraphDB()

    # Create a more complex graph structure
    print("Creating complex graph structure...")
    nodes = {}
    for i in range(10):
        node_id = graph.create_node("Node", {"id": i, "name": f"Node_{i}"})
        nodes[i] = node_id

    # Create edges forming a specific topology
    edges = [
        (0, 1), (0, 2),  # Node 0 connects to 1 and 2
        (1, 3), (1, 4),  # Node 1 connects to 3 and 4
        (2, 4), (2, 5),  # Node 2 connects to 4 and 5
        (3, 6),          # Node 3 connects to 6
        (4, 6), (4, 7),  # Node 4 connects to 6 and 7
        (5, 7),          # Node 5 connects to 7
        (6, 8),          # Node 6 connects to 8
        (7, 8), (7, 9),  # Node 7 connects to 8 and 9
        (8, 9),          # Node 8 connects to 9
    ]

    for source, target in edges:
        graph.create_edge(nodes[source], nodes[target], "CONNECTED_TO")

    stats = graph.stats()
    print(f"Graph created: {stats}")

    # Test neighbor finding at different depths
    print("\nNeighbor analysis for Node 0:")
    for depth in [1, 2, 3]:
        neighbors = graph.find_neighbors(nodes[0], depth=depth)
        print(f"  Depth {depth}: {len(neighbors)} neighbors")

    # Test shortest paths
    print("\nShortest path analysis:")
    test_paths = [(0, 5), (0, 9), (1, 8)]

    for start, end in test_paths:
        path = graph.shortest_path(nodes[start], nodes[end])
        if path:
            print(f"  Node {start} -> Node {end}: {len(path)} hops")
        else:
            print(f"  Node {start} -> Node {end}: No path found")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("PyRuvector Graph Database Examples")
    print("=" * 70)

    try:
        example_basic_graph()
        example_social_network()
        example_knowledge_graph()
        example_transactions()
        example_batch_operations()
        example_graph_algorithms()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
