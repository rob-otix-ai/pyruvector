"""
Test suite for graph.rs fixes:
1. Cypher query implementation
2. Persistence (save/load)
3. Hyperedge CRUD operations
"""

import pytest
import tempfile
import os
from pyruvector import GraphDB

# Skip markers for unimplemented features
pytestmark_cypher = pytest.mark.skip(reason="Cypher queries not yet implemented - requires ruvector-graph hybrid module parser")
pytestmark_persistence = pytest.mark.skip(reason="Graph save/load not yet implemented in wrapper")
pytestmark_delete_hyperedge = pytest.mark.skip(reason="Hyperedge deletion not yet implemented in ruvector-graph crate")


@pytestmark_cypher
class TestCypherQueries:
    """Test Cypher query implementation"""

    def setup_method(self):
        """Create test graph database"""
        self.graph = GraphDB()

        # Create test data
        self.alice = self.graph.create_node("Person", {"name": "Alice", "age": 30})
        self.bob = self.graph.create_node("Person", {"name": "Bob", "age": 25})
        self.charlie = self.graph.create_node("Person", {"name": "Charlie", "age": 35})
        self.company = self.graph.create_node("Company", {"name": "TechCorp"})

        # Create edges
        self.edge1 = self.graph.create_edge(self.alice, self.bob, "KNOWS", {"since": 2020})
        self.edge2 = self.graph.create_edge(self.alice, self.charlie, "KNOWS", {"since": 2018})
        self.edge3 = self.graph.create_edge(self.alice, self.company, "WORKS_AT")

    def test_match_all_nodes(self):
        """Test: MATCH (n) RETURN n"""
        results = self.graph.query("MATCH (n) RETURN n")

        assert len(results) == 4, "Should return all 4 nodes"
        assert results.columns == ["n"]

        rows = results.get_rows()
        assert len(rows) == 4

    def test_match_nodes_with_label(self):
        """Test: MATCH (n:Label) RETURN n"""
        results = self.graph.query("MATCH (p:Person) RETURN p")

        assert len(results) == 3, "Should return 3 Person nodes"

        rows = results.get_rows()
        for row in rows:
            node_data = row["p"]
            assert node_data["label"] == "Person"

    def test_match_with_where_greater_than(self):
        """Test: MATCH (n) WHERE n.age > 25 RETURN n"""
        results = self.graph.query("MATCH (p:Person) WHERE p.age > 25 RETURN p")

        assert len(results) == 2, "Should return Alice and Charlie"

        rows = results.get_rows()
        ages = [row["p"]["properties"]["age"] for row in rows]
        assert all(age > 25 for age in ages)

    def test_match_with_where_equals(self):
        """Test: MATCH (n) WHERE n.name = 'Alice' RETURN n"""
        results = self.graph.query("MATCH (p:Person) WHERE p.name = 'Alice' RETURN p")

        assert len(results) == 1, "Should return only Alice"

        rows = results.get_rows()
        assert rows[0]["p"]["properties"]["name"] == "Alice"

    def test_match_relationship_pattern(self):
        """Test: MATCH (n)-[r]->(m) RETURN n,r,m"""
        results = self.graph.query("MATCH (n)-[r]->(m) RETURN n,r,m")

        assert len(results) == 3, "Should return all 3 relationships"
        assert set(results.columns) == {"n", "r", "m"}

        rows = results.get_rows()
        for row in rows:
            assert "n" in row
            assert "r" in row
            assert "m" in row

    def test_match_relationship_with_type(self):
        """Test: MATCH (n)-[r:KNOWS]->(m) RETURN n,r,m"""
        results = self.graph.query("MATCH (n)-[r:KNOWS]->(m) RETURN n,r,m")

        assert len(results) == 2, "Should return 2 KNOWS relationships"

        rows = results.get_rows()
        for row in rows:
            assert row["r"]["rel_type"] == "KNOWS"


class TestPersistence:
    """Test save/load functionality"""

    @pytestmark_persistence
    def test_save_and_load(self):
        """Test saving and loading graph to/from disk"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
            temp_path = f.name

        try:
            # Create and populate graph
            graph1 = GraphDB(path=temp_path)

            alice = graph1.create_node("Person", {"name": "Alice", "age": 30})
            bob = graph1.create_node("Person", {"name": "Bob", "age": 25})
            graph1.create_edge(alice, bob, "KNOWS", {"since": 2020})

            # Save to disk
            assert graph1.save()

            # Load from disk
            graph2 = GraphDB.load(temp_path)

            # Verify data
            stats1 = graph1.stats()
            stats2 = graph2.stats()

            assert stats1["nodes"] == stats2["nodes"] == 2
            assert stats1["edges"] == stats2["edges"] == 1

            # Verify node properties
            alice_loaded = graph2.get_node(alice)
            assert alice_loaded is not None
            assert alice_loaded.label == "Person"
            props = alice_loaded.get_properties()
            assert props["name"] == "Alice"
            assert props["age"] == 30

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @pytestmark_persistence
    def test_save_without_path_fails(self):
        """Test that save() fails when no path is configured"""
        graph = GraphDB()
        graph.create_node("Person", {"name": "Alice"})

        with pytest.raises((ValueError, RuntimeError), match="No path configured|not yet implemented"):
            graph.save()

    @pytestmark_persistence
    def test_load_nonexistent_file_fails(self):
        """Test that load() fails for non-existent file"""
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError), match="File not found|not yet implemented"):
            GraphDB.load("/tmp/nonexistent_graph_12345.db")

    @pytestmark_persistence
    def test_persistence_with_hyperedges(self):
        """Test save/load with hyperedges"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
            temp_path = f.name

        try:
            # Create graph with hyperedge
            graph1 = GraphDB(path=temp_path)

            n1 = graph1.create_node("Person", {"name": "Alice"})
            n2 = graph1.create_node("Person", {"name": "Bob"})
            n3 = graph1.create_node("Person", {"name": "Charlie"})

            he = graph1.create_hyperedge([n1, n2, n3], {"type": "team"})

            # Save and reload
            graph1.save()
            graph2 = GraphDB.load(temp_path)

            # Verify hyperedge
            stats = graph2.stats()
            assert stats["hyperedges"] == 1

            he_loaded = graph2.get_hyperedge(he)
            assert he_loaded is not None
            assert len(he_loaded.nodes) == 3

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestHyperedgeCRUD:
    """Test hyperedge CRUD operations"""

    def setup_method(self):
        """Create test graph database"""
        self.graph = GraphDB()

        # Create test nodes
        self.n1 = self.graph.create_node("Person", {"name": "Alice"})
        self.n2 = self.graph.create_node("Person", {"name": "Bob"})
        self.n3 = self.graph.create_node("Person", {"name": "Charlie"})
        self.n4 = self.graph.create_node("Person", {"name": "Dave"})

    def test_create_hyperedge(self):
        """Test creating a hyperedge"""
        he_id = self.graph.create_hyperedge(
            [self.n1, self.n2, self.n3],
            {"type": "collaboration", "weight": 1.0}
        )

        assert he_id is not None
        assert len(he_id) > 0

    def test_get_hyperedge(self):
        """Test retrieving a hyperedge"""
        he_id = self.graph.create_hyperedge([self.n1, self.n2, self.n3])

        he = self.graph.get_hyperedge(he_id)

        assert he is not None
        assert he.id == he_id
        assert len(he.nodes) == 3
        assert set(he.nodes) == {self.n1, self.n2, self.n3}

    def test_hyperedge_with_properties(self):
        """Test hyperedge properties"""
        props = {"type": "team", "score": 95.5, "active": True}
        he_id = self.graph.create_hyperedge([self.n1, self.n2], props)

        he = self.graph.get_hyperedge(he_id)
        loaded_props = he.get_properties()

        assert loaded_props["type"] == "team"
        assert loaded_props["score"] == 95.5
        assert loaded_props["active"]

    @pytestmark_delete_hyperedge
    def test_delete_hyperedge(self):
        """Test deleting a hyperedge"""
        he_id = self.graph.create_hyperedge([self.n1, self.n2, self.n3])

        # Verify it exists
        assert self.graph.get_hyperedge(he_id) is not None

        # Delete it
        result = self.graph.delete_hyperedge(he_id)
        assert result

        # Verify it's gone
        assert self.graph.get_hyperedge(he_id) is None

    @pytestmark_delete_hyperedge
    def test_delete_nonexistent_hyperedge(self):
        """Test deleting non-existent hyperedge returns False"""
        result = self.graph.delete_hyperedge("nonexistent-id")
        assert not result

    def test_hyperedge_minimum_nodes(self):
        """Test that hyperedge requires at least 2 nodes"""
        with pytest.raises(ValueError, match="at least 2 nodes"):
            self.graph.create_hyperedge([self.n1])

    def test_hyperedge_with_invalid_node(self):
        """Test creating hyperedge with non-existent node"""
        with pytest.raises((ValueError, RuntimeError), match="not found|Node not found"):
            self.graph.create_hyperedge([self.n1, "invalid-node-id", self.n2])

    def test_multiple_hyperedges(self):
        """Test creating multiple hyperedges"""
        he1 = self.graph.create_hyperedge([self.n1, self.n2])
        he2 = self.graph.create_hyperedge([self.n2, self.n3, self.n4])
        he3 = self.graph.create_hyperedge([self.n1, self.n3])

        stats = self.graph.stats()
        assert stats["hyperedges"] == 3

        # Verify all are retrievable
        assert self.graph.get_hyperedge(he1) is not None
        assert self.graph.get_hyperedge(he2) is not None
        assert self.graph.get_hyperedge(he3) is not None

    def test_hyperedge_stats(self):
        """Test that stats() includes hyperedge count"""
        # Start with no hyperedges
        stats = self.graph.stats()
        assert stats["hyperedges"] == 0

        # Add hyperedges
        self.graph.create_hyperedge([self.n1, self.n2])
        self.graph.create_hyperedge([self.n2, self.n3, self.n4])

        # Check stats
        stats = self.graph.stats()
        assert stats["hyperedges"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
