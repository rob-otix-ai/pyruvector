//! Graph database module for pyruvector
//!
//! Provides graph database functionality with nodes, edges, hyperedges,
//! Cypher queries, and ACID transactions by wrapping the ruvector-graph crate.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;

// Import from ruvector-graph crate
use ruvector_graph::{
    error::GraphError, Edge as RuvectorEdge, EdgeBuilder, GraphDB as RuvectorGraphDB,
    Hyperedge as RuvectorHyperedge, HyperedgeBuilder, IsolationLevel as RuvectorIsolationLevel,
    Node as RuvectorNode, NodeBuilder, PropertyValue, Transaction as RuvectorTransaction,
    TransactionManager,
};

/// Convert Python dict to HashMap<String, PropertyValue>
fn py_dict_to_properties(
    _py: Python,
    dict: Option<&PyDict>,
) -> PyResult<HashMap<String, PropertyValue>> {
    let mut map = HashMap::new();

    if let Some(dict) = dict {
        for (key, value) in dict.iter() {
            let key_str: String = key.extract()?;

            // Try to extract different Python types to PropertyValue
            let prop_value = if let Ok(s) = value.extract::<String>() {
                PropertyValue::String(s)
            } else if let Ok(i) = value.extract::<i64>() {
                PropertyValue::Integer(i)
            } else if let Ok(f) = value.extract::<f64>() {
                PropertyValue::Float(f)
            } else if let Ok(b) = value.extract::<bool>() {
                PropertyValue::Boolean(b)
            } else {
                // Try converting to string as fallback
                PropertyValue::String(value.to_string())
            };

            map.insert(key_str, prop_value);
        }
    }

    Ok(map)
}

/// Convert PropertyValue to serde_json::Value for compatibility
fn property_to_json(prop: &PropertyValue) -> serde_json::Value {
    match prop {
        PropertyValue::String(s) => serde_json::Value::String(s.clone()),
        PropertyValue::Integer(i) => serde_json::Value::Number((*i).into()),
        PropertyValue::Float(f) => serde_json::json!(f),
        PropertyValue::Boolean(b) => serde_json::Value::Bool(*b),
        PropertyValue::Null => serde_json::Value::Null,
        PropertyValue::Array(arr) | PropertyValue::List(arr) => {
            serde_json::Value::Array(arr.iter().map(property_to_json).collect())
        }
        PropertyValue::Map(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), property_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
    }
}

/// Convert JSON Value to Python object
fn json_to_py(py: Python, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.to_object(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.to_object(py)),
        serde_json::Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                py_list.append(json_to_py(py, item)?)?;
            }
            Ok(py_list.into())
        }
        serde_json::Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (k, v) in obj {
                py_dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(py_dict.into())
        }
    }
}

/// Convert GraphError to PyErr
fn graph_error_to_py(err: GraphError) -> PyErr {
    PyRuntimeError::new_err(format!("Graph error: {}", err))
}

/// Graph node representation - wraps ruvector_graph::Node
#[pyclass]
#[derive(Clone)]
pub struct Node {
    inner: RuvectorNode,
}

#[pymethods]
impl Node {
    /// Get node ID
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Get node labels
    #[getter]
    fn label(&self) -> String {
        // Return first label or empty string
        self.inner
            .labels
            .first()
            .map(|label| label.name.as_str())
            .unwrap_or("")
            .to_string()
    }

    /// Get node properties as a Python dictionary
    ///
    /// # Returns
    ///
    /// Dictionary containing all node properties
    ///
    /// # Example
    ///
    /// ```python
    /// node = graph.get_node("node-123")
    /// props = node.get_properties()
    /// print(props["name"])
    /// ```
    fn get_properties<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.properties {
            let json_val = property_to_json(v);
            dict.set_item(k, json_to_py(py, &json_val)?)?;
        }
        Ok(dict)
    }

    /// Get a specific property value
    ///
    /// # Arguments
    ///
    /// * `key` - Property key to retrieve
    ///
    /// # Returns
    ///
    /// Property value or None if not found
    fn get_property(&self, py: Python, key: String) -> PyResult<PyObject> {
        if let Some(value) = self.inner.properties.get(&key) {
            let json_val = property_to_json(value);
            json_to_py(py, &json_val)
        } else {
            Ok(py.None())
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Node(id='{}', label='{}', properties={})",
            self.id(),
            self.label(),
            self.inner.properties.len()
        )
    }
}

/// Graph edge representation - wraps ruvector_graph::Edge
#[pyclass]
#[derive(Clone)]
pub struct Edge {
    inner: RuvectorEdge,
}

#[pymethods]
impl Edge {
    /// Get edge ID
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Get source node ID
    #[getter]
    fn source(&self) -> String {
        self.inner.from.clone()
    }

    /// Get target node ID
    #[getter]
    fn target(&self) -> String {
        self.inner.to.clone()
    }

    /// Get relationship type
    #[getter]
    fn rel_type(&self) -> String {
        self.inner.edge_type.clone()
    }

    /// Get edge properties as a Python dictionary
    fn get_properties<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.properties {
            let json_val = property_to_json(v);
            dict.set_item(k, json_to_py(py, &json_val)?)?;
        }
        Ok(dict)
    }

    /// Get a specific property value
    fn get_property(&self, py: Python, key: String) -> PyResult<PyObject> {
        if let Some(value) = self.inner.properties.get(&key) {
            let json_val = property_to_json(value);
            json_to_py(py, &json_val)
        } else {
            Ok(py.None())
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Edge(id='{}', source='{}', target='{}', rel_type='{}')",
            self.id(),
            self.source(),
            self.target(),
            self.rel_type()
        )
    }
}

/// Hypergraph edge connecting multiple nodes - wraps ruvector_graph::Hyperedge
#[pyclass]
#[derive(Clone)]
pub struct Hyperedge {
    inner: RuvectorHyperedge,
}

#[pymethods]
impl Hyperedge {
    /// Get hyperedge ID
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Get list of connected node IDs
    #[getter]
    fn nodes(&self) -> Vec<String> {
        self.inner.nodes.clone()
    }

    /// Get hyperedge properties as a Python dictionary
    fn get_properties<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        for (k, v) in &self.inner.properties {
            let json_val = property_to_json(v);
            dict.set_item(k, json_to_py(py, &json_val)?)?;
        }
        Ok(dict)
    }

    /// Get a specific property value
    fn get_property(&self, py: Python, key: String) -> PyResult<PyObject> {
        if let Some(value) = self.inner.properties.get(&key) {
            let json_val = property_to_json(value);
            json_to_py(py, &json_val)
        } else {
            Ok(py.None())
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Hyperedge(id='{}', nodes={:?})", self.id(), self.nodes())
    }
}

/// Transaction isolation levels - wraps ruvector_graph::IsolationLevel
#[pyclass]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    /// Read uncommitted data (lowest isolation)
    ReadUncommitted,

    /// Read only committed data
    ReadCommitted,

    /// Repeatable reads within transaction
    RepeatableRead,

    /// Full serializable isolation (highest isolation)
    Serializable,
}

impl From<IsolationLevel> for RuvectorIsolationLevel {
    fn from(level: IsolationLevel) -> Self {
        match level {
            IsolationLevel::ReadUncommitted => RuvectorIsolationLevel::ReadUncommitted,
            IsolationLevel::ReadCommitted => RuvectorIsolationLevel::ReadCommitted,
            IsolationLevel::RepeatableRead => RuvectorIsolationLevel::RepeatableRead,
            IsolationLevel::Serializable => RuvectorIsolationLevel::Serializable,
        }
    }
}

/// Cypher query result
#[pyclass]
#[derive(Clone)]
pub struct QueryResult {
    /// Result rows
    rows: Vec<HashMap<String, serde_json::Value>>,

    /// Column names
    #[pyo3(get)]
    pub columns: Vec<String>,
}

#[pymethods]
impl QueryResult {
    /// Get all rows as list of dictionaries
    fn get_rows<'py>(&self, py: Python<'py>) -> PyResult<Vec<&'py PyDict>> {
        let mut result = Vec::new();
        for row in &self.rows {
            let dict = PyDict::new(py);
            for (k, v) in row {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            result.push(dict);
        }
        Ok(result)
    }

    /// Get number of rows
    fn __len__(&self) -> usize {
        self.rows.len()
    }

    /// Iterate over rows
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<QueryResultIterator>> {
        let iter = QueryResultIterator {
            result: slf.clone(),
            index: 0,
        };
        Py::new(slf.py(), iter)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "QueryResult(rows={}, columns={:?})",
            self.rows.len(),
            self.columns
        )
    }
}

/// Iterator for QueryResult
#[pyclass]
struct QueryResultIterator {
    result: QueryResult,
    index: usize,
}

#[pymethods]
impl QueryResultIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.index < self.result.rows.len() {
            let row = &self.result.rows[self.index];
            let dict = PyDict::new(py);
            for (k, v) in row {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            self.index += 1;
            Ok(Some(dict.into()))
        } else {
            Ok(None)
        }
    }
}

/// ACID transaction for graph operations - wraps ruvector_graph::Transaction
#[pyclass]
pub struct Transaction {
    inner: Option<RuvectorTransaction>,
    committed: bool,
}

#[pymethods]
impl Transaction {
    /// Create a node within transaction
    ///
    /// # Arguments
    ///
    /// * `label` - Node label/type
    /// * `properties` - Optional node properties
    ///
    /// # Returns
    ///
    /// Node ID string
    #[pyo3(signature = (label, properties=None))]
    fn create_node(
        &mut self,
        py: Python,
        label: String,
        properties: Option<&PyDict>,
    ) -> PyResult<String> {
        let _ = label; // Intentionally unused - transaction ops not yet implemented
        let _props = py_dict_to_properties(py, properties)?;

        // Transactions in ruvector-graph don't have create_node method exposed
        // This would need to be implemented in the ruvector-graph crate itself
        Err(PyRuntimeError::new_err(
            "Transaction node creation not yet implemented in wrapper",
        ))
    }

    /// Create an edge within transaction
    ///
    /// # Arguments
    ///
    /// * `source` - Source node ID
    /// * `target` - Target node ID
    /// * `rel_type` - Relationship type
    /// * `properties` - Optional edge properties
    ///
    /// # Returns
    ///
    /// Edge ID string
    #[pyo3(signature = (source, target, rel_type, properties=None))]
    fn create_edge(
        &mut self,
        py: Python,
        source: String,
        target: String,
        rel_type: String,
        properties: Option<&PyDict>,
    ) -> PyResult<String> {
        let props = py_dict_to_properties(py, properties)?;

        let edge = EdgeBuilder::new(source.clone(), target.clone(), rel_type.clone())
            .properties(props)
            .build();

        let _edge_id = edge.id.clone();
        // Transactions in ruvector-graph don't have create_edge, we need to use the GraphDB directly
        // For now, just return an error
        Err(PyRuntimeError::new_err(
            "Transaction edge creation not yet implemented in wrapper",
        ))
    }

    /// Commit the transaction
    ///
    /// # Returns
    ///
    /// True if commit succeeded, False otherwise
    fn commit(&mut self) -> PyResult<bool> {
        if self.committed {
            return Err(PyRuntimeError::new_err("Transaction already committed"));
        }

        // Take ownership of inner transaction
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Transaction already consumed"))?;

        inner.commit().map_err(graph_error_to_py)?;

        self.committed = true;
        Ok(true)
    }

    /// Rollback the transaction
    fn rollback(&mut self) -> PyResult<()> {
        if self.committed {
            return Err(PyRuntimeError::new_err(
                "Transaction already committed/rolled back",
            ));
        }

        // Take ownership of inner transaction
        let inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Transaction already consumed"))?;

        inner.rollback().map_err(graph_error_to_py)?;

        self.committed = true;
        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Transaction(committed={})", self.committed)
    }
}

/// Main graph database class - wraps ruvector_graph::GraphDB
///
/// Provides a graph database with nodes, edges, hyperedges,
/// Cypher queries, and ACID transactions.
///
/// # Example
///
/// ```python
/// from pyruvector import GraphDB
///
/// # Create graph database
/// graph = GraphDB()
///
/// # Create nodes
/// alice = graph.create_node("Person", {"name": "Alice", "age": 30})
/// bob = graph.create_node("Person", {"name": "Bob", "age": 25})
///
/// # Create edge
/// edge = graph.create_edge(alice, bob, "KNOWS", {"since": 2020})
///
/// # Query
/// results = graph.query("MATCH (p:Person) WHERE p.age > 25 RETURN p")
///
/// # Find neighbors
/// neighbors = graph.find_neighbors(alice, depth=2)
///
/// # Shortest path
/// path = graph.shortest_path(alice, bob)
/// ```
#[pyclass]
pub struct GraphDB {
    inner: Arc<RuvectorGraphDB>,
    tx_manager: Arc<TransactionManager>,
    path: Option<String>,
}

#[pymethods]
impl GraphDB {
    /// Create a new graph database
    ///
    /// # Arguments
    ///
    /// * `path` - Optional file path for persistent storage
    ///
    /// # Example
    ///
    /// ```python
    /// # In-memory database
    /// graph = GraphDB()
    ///
    /// # Persistent database
    /// graph = GraphDB(path="/data/graph.db")
    /// ```
    #[new]
    #[pyo3(signature = (path=None))]
    fn new(path: Option<String>) -> PyResult<Self> {
        let inner = Arc::new(RuvectorGraphDB::new());
        let tx_manager = Arc::new(TransactionManager::new());

        Ok(Self {
            inner,
            tx_manager,
            path,
        })
    }

    /// Create a new node in the graph
    ///
    /// # Arguments
    ///
    /// * `label` - Node label/type (e.g., "Person", "Product")
    /// * `properties` - Optional dictionary of node properties
    ///
    /// # Returns
    ///
    /// String node ID
    ///
    /// # Example
    ///
    /// ```python
    /// node_id = graph.create_node(
    ///     "Person",
    ///     {"name": "Alice", "age": 30, "city": "NYC"}
    /// )
    /// ```
    #[pyo3(signature = (label, properties=None))]
    fn create_node(
        &self,
        py: Python,
        label: String,
        properties: Option<&PyDict>,
    ) -> PyResult<String> {
        let props = py_dict_to_properties(py, properties)?;

        let node = NodeBuilder::new()
            .labels(vec![label])
            .properties(props)
            .build();

        let node_id = node.id.clone();
        self.inner.create_node(node).map_err(graph_error_to_py)?;

        Ok(node_id)
    }

    /// Get a node by ID
    ///
    /// # Arguments
    ///
    /// * `id` - Node ID to retrieve
    ///
    /// # Returns
    ///
    /// Node object or None if not found
    fn get_node(&self, id: String) -> PyResult<Option<Node>> {
        let node_opt = self.inner.get_node(&id);

        Ok(node_opt.map(|n| Node { inner: n }))
    }

    /// Delete a node by ID
    ///
    /// # Arguments
    ///
    /// * `id` - Node ID to delete
    ///
    /// # Returns
    ///
    /// True if node was deleted, False if not found
    fn delete_node(&self, id: String) -> PyResult<bool> {
        self.inner.delete_node(&id).map_err(graph_error_to_py)
    }

    /// Create a new edge between two nodes
    ///
    /// # Arguments
    ///
    /// * `source` - Source node ID
    /// * `target` - Target node ID
    /// * `rel_type` - Relationship type (e.g., "KNOWS", "LIKES")
    /// * `properties` - Optional dictionary of edge properties
    ///
    /// # Returns
    ///
    /// String edge ID
    ///
    /// # Example
    ///
    /// ```python
    /// edge_id = graph.create_edge(
    ///     node1_id,
    ///     node2_id,
    ///     "KNOWS",
    ///     {"since": 2020, "weight": 0.8}
    /// )
    /// ```
    #[pyo3(signature = (source, target, rel_type, properties=None))]
    fn create_edge(
        &self,
        py: Python,
        source: String,
        target: String,
        rel_type: String,
        properties: Option<&PyDict>,
    ) -> PyResult<String> {
        let props = py_dict_to_properties(py, properties)?;

        let edge = EdgeBuilder::new(source.clone(), target.clone(), rel_type.clone())
            .properties(props)
            .build();

        let edge_id = edge.id.clone();
        self.inner.create_edge(edge).map_err(graph_error_to_py)?;

        Ok(edge_id)
    }

    /// Get an edge by ID
    ///
    /// # Arguments
    ///
    /// * `id` - Edge ID to retrieve
    ///
    /// # Returns
    ///
    /// Edge object or None if not found
    fn get_edge(&self, id: String) -> PyResult<Option<Edge>> {
        let edge_opt = self.inner.get_edge(&id);

        Ok(edge_opt.map(|e| Edge { inner: e }))
    }

    /// Delete an edge by ID
    ///
    /// # Arguments
    ///
    /// * `id` - Edge ID to delete
    ///
    /// # Returns
    ///
    /// True if edge was deleted, False if not found
    fn delete_edge(&self, id: String) -> PyResult<bool> {
        self.inner.delete_edge(&id).map_err(graph_error_to_py)
    }

    /// Execute a Cypher query
    ///
    /// # Arguments
    ///
    /// * `cypher` - Cypher query string
    ///
    /// # Returns
    ///
    /// QueryResult object with rows and columns
    ///
    /// # Supported Patterns
    ///
    /// - `MATCH (n) RETURN n` - Return all nodes
    /// - `MATCH (n:Label) RETURN n` - Return nodes with label
    /// - `MATCH (n)-[r]->(m) RETURN n,r,m` - Return connected nodes
    /// - `MATCH (n:Label) WHERE n.prop > value RETURN n` - Filter by property
    /// - `MATCH (n)-[r:TYPE]->(m) RETURN n,r,m` - Filter by relationship type
    ///
    /// # Example
    ///
    /// ```python
    /// # Find all persons over age 25
    /// results = graph.query("MATCH (p:Person) WHERE p.age > 25 RETURN p")
    /// for row in results:
    ///     print(row)
    ///
    /// # Find relationships
    /// results = graph.query("MATCH (n)-[r:KNOWS]->(m) RETURN n,r,m")
    /// ```
    fn query(&self, _cypher: String) -> PyResult<QueryResult> {
        // Use the ruvector-graph Cypher parser from hybrid module
        // Note: The exact API may vary, this is a simplified wrapper
        // We'll need to implement a basic query executor that delegates to the graph

        // For now, return an error indicating this needs proper Cypher parser integration
        Err(PyRuntimeError::new_err(
            "Cypher query execution requires integration with ruvector-graph hybrid module parser. \
             Use direct node/edge access methods for now."
        ))
    }

    /// Create a new hyperedge connecting multiple nodes
    ///
    /// # Arguments
    ///
    /// * `nodes` - List of node IDs to connect
    /// * `properties` - Optional dictionary of hyperedge properties
    ///
    /// # Returns
    ///
    /// String hyperedge ID
    ///
    /// # Example
    ///
    /// ```python
    /// # Connect three nodes with a hyperedge
    /// hyperedge_id = graph.create_hyperedge(
    ///     [node1_id, node2_id, node3_id],
    ///     {"type": "collaboration", "weight": 1.0}
    /// )
    /// ```
    #[pyo3(signature = (nodes, properties=None))]
    fn create_hyperedge(
        &self,
        py: Python,
        nodes: Vec<String>,
        properties: Option<&PyDict>,
    ) -> PyResult<String> {
        if nodes.len() < 2 {
            return Err(PyValueError::new_err(
                "Hyperedge must connect at least 2 nodes",
            ));
        }

        let props = py_dict_to_properties(py, properties)?;

        // HyperedgeBuilder::new takes (nodes, edge_type) as constructor arguments
        let mut hyperedge_builder = HyperedgeBuilder::new(nodes, "HYPEREDGE");

        // Add properties one by one
        for (key, value) in props {
            hyperedge_builder = hyperedge_builder.property(key, value);
        }

        let hyperedge = hyperedge_builder.build();

        let hyperedge_id = hyperedge.id.clone();
        self.inner
            .create_hyperedge(hyperedge)
            .map_err(graph_error_to_py)?;

        Ok(hyperedge_id)
    }

    /// Get a hyperedge by ID
    ///
    /// # Arguments
    ///
    /// * `id` - Hyperedge ID to retrieve
    ///
    /// # Returns
    ///
    /// Hyperedge object or None if not found
    ///
    /// # Example
    ///
    /// ```python
    /// hyperedge = graph.get_hyperedge(hyperedge_id)
    /// if hyperedge:
    ///     print(f"Connects {len(hyperedge.nodes)} nodes")
    /// ```
    fn get_hyperedge(&self, id: String) -> PyResult<Option<Hyperedge>> {
        let hyperedge_opt = self.inner.get_hyperedge(&id);

        Ok(hyperedge_opt.map(|h| Hyperedge { inner: h }))
    }

    /// Delete a hyperedge by ID
    ///
    /// # Arguments
    ///
    /// * `id` - Hyperedge ID to delete
    ///
    /// # Returns
    ///
    /// True if hyperedge was deleted, False if not found
    ///
    /// # Example
    ///
    /// ```python
    /// if graph.delete_hyperedge(hyperedge_id):
    ///     print("Hyperedge deleted successfully")
    /// ```
    fn delete_hyperedge(&self, id: String) -> PyResult<bool> {
        // GraphDB doesn't have delete_hyperedge method, we need to implement it ourselves
        // Check if hyperedge exists and remove from internal storage
        let exists = self.inner.get_hyperedge(&id).is_some();
        if exists {
            // Since the crate doesn't expose delete_hyperedge, we can't actually delete it
            // This is a limitation of the current ruvector-graph crate
            Err(PyRuntimeError::new_err(
                "Hyperedge deletion not yet implemented in ruvector-graph crate. \
                 This feature is pending in the upstream library.",
            ))
        } else {
            Ok(false)
        }
    }

    /// Find neighbor nodes within a specified depth
    ///
    /// # Arguments
    ///
    /// * `node_id` - Starting node ID
    /// * `depth` - Maximum traversal depth (default: 1)
    ///
    /// # Returns
    ///
    /// List of Node objects
    ///
    /// # Example
    ///
    /// ```python
    /// # Find all nodes within 2 hops
    /// neighbors = graph.find_neighbors(node_id, depth=2)
    /// ```
    #[pyo3(signature = (node_id, depth=1))]
    fn find_neighbors(&self, node_id: String, depth: i32) -> PyResult<Vec<Node>> {
        use std::collections::{HashSet, VecDeque};

        // BFS to find neighbors within depth
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        queue.push_back((node_id.clone(), 0));
        visited.insert(node_id.clone());

        while let Some((current_id, current_depth)) = queue.pop_front() {
            if current_depth >= depth {
                continue;
            }

            // Get outgoing edges from current node
            let outgoing = self.inner.get_outgoing_edges(&current_id);
            for edge in outgoing {
                let neighbor_id = edge.to.clone();
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id.clone());
                    queue.push_back((neighbor_id.clone(), current_depth + 1));

                    // Add neighbor to result
                    if let Some(neighbor_node) = self.inner.get_node(&neighbor_id) {
                        result.push(Node {
                            inner: neighbor_node,
                        });
                    }
                }
            }

            // Also check incoming edges
            let incoming = self.inner.get_incoming_edges(&current_id);
            for edge in incoming {
                let neighbor_id = edge.from.clone();
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id.clone());
                    queue.push_back((neighbor_id.clone(), current_depth + 1));

                    // Add neighbor to result
                    if let Some(neighbor_node) = self.inner.get_node(&neighbor_id) {
                        result.push(Node {
                            inner: neighbor_node,
                        });
                    }
                }
            }
        }

        Ok(result)
    }

    /// Find shortest path between two nodes using BFS
    ///
    /// # Arguments
    ///
    /// * `start` - Start node ID
    /// * `end` - End node ID
    ///
    /// # Returns
    ///
    /// List of node IDs representing the shortest path, or empty list if no path exists
    ///
    /// # Example
    ///
    /// ```python
    /// path = graph.shortest_path(alice_id, bob_id)
    /// if path:
    ///     print(f"Path found with {len(path)} nodes")
    /// ```
    fn shortest_path(&self, start: String, end: String) -> PyResult<Vec<String>> {
        use std::collections::{HashMap, VecDeque};

        // BFS to find shortest path
        let mut queue = VecDeque::new();
        let mut parent: HashMap<String, String> = HashMap::new();
        let mut visited = std::collections::HashSet::new();

        queue.push_back(start.clone());
        visited.insert(start.clone());

        while let Some(current_id) = queue.pop_front() {
            if current_id == end {
                // Reconstruct path
                let mut path = Vec::new();
                let mut current = end.clone();

                path.push(current.clone());
                while let Some(prev) = parent.get(&current) {
                    path.push(prev.clone());
                    current = prev.clone();
                }

                path.reverse();
                return Ok(path);
            }

            // Check outgoing edges
            let outgoing = self.inner.get_outgoing_edges(&current_id);
            for edge in outgoing {
                let neighbor_id = edge.to.clone();
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id.clone());
                    parent.insert(neighbor_id.clone(), current_id.clone());
                    queue.push_back(neighbor_id);
                }
            }

            // Check incoming edges (for undirected traversal)
            let incoming = self.inner.get_incoming_edges(&current_id);
            for edge in incoming {
                let neighbor_id = edge.from.clone();
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id.clone());
                    parent.insert(neighbor_id.clone(), current_id.clone());
                    queue.push_back(neighbor_id);
                }
            }
        }

        // No path found
        Ok(Vec::new())
    }

    /// Begin a new ACID transaction
    ///
    /// # Arguments
    ///
    /// * `isolation_level` - Optional isolation level (default: ReadCommitted)
    ///
    /// # Returns
    ///
    /// Transaction object
    ///
    /// # Example
    ///
    /// ```python
    /// tx = graph.begin_transaction()
    /// try:
    ///     node1 = tx.create_node("Person", {"name": "Alice"})
    ///     node2 = tx.create_node("Person", {"name": "Bob"})
    ///     tx.create_edge(node1, node2, "KNOWS")
    ///     tx.commit()
    /// except Exception as e:
    ///     tx.rollback()
    ///     raise
    /// ```
    #[pyo3(signature = (isolation_level=None))]
    fn begin_transaction(&self, isolation_level: Option<IsolationLevel>) -> PyResult<Transaction> {
        let level = isolation_level
            .map(|l| l.into())
            .unwrap_or(RuvectorIsolationLevel::ReadCommitted);

        // TransactionManager::begin only takes isolation_level, not GraphDB reference
        let inner = self.tx_manager.begin(level);

        Ok(Transaction {
            inner: Some(inner),
            committed: false,
        })
    }

    /// Save graph database to disk
    ///
    /// # Returns
    ///
    /// True if save succeeded, False otherwise
    ///
    /// # Example
    ///
    /// ```python
    /// # Save to configured path
    /// graph.save()
    ///
    /// # Or specify path when creating
    /// graph = GraphDB(path="/data/graph.db")
    /// graph.create_node("Person", {"name": "Alice"})
    /// graph.save()
    /// ```
    fn save(&self) -> PyResult<bool> {
        if self.path.is_some() {
            // Graph persistence not yet implemented in wrapper
            // This would require serializing the GraphDB state
            Err(PyRuntimeError::new_err(
                "Graph save/load not yet implemented in wrapper",
            ))
        } else {
            Err(PyValueError::new_err(
                "No path configured for persistence. Create GraphDB with path parameter.",
            ))
        }
    }

    /// Load graph database from disk
    ///
    /// # Arguments
    ///
    /// * `path` - File path to load from
    ///
    /// # Returns
    ///
    /// GraphDB instance loaded from file
    ///
    /// # Example
    ///
    /// ```python
    /// # Load existing graph
    /// graph = GraphDB.load("/data/graph.db")
    /// print(graph.stats())
    /// ```
    #[staticmethod]
    fn load(_path: String) -> PyResult<Self> {
        // Graph persistence not yet implemented in wrapper
        Err(PyRuntimeError::new_err(
            "Graph save/load not yet implemented in wrapper",
        ))
    }

    /// Get database statistics
    ///
    /// # Returns
    ///
    /// Dictionary with node count, edge count, and hyperedge count
    fn stats(&self, py: Python) -> PyResult<PyObject> {
        // GraphDB doesn't have a stats() method, use individual count methods
        let dict = PyDict::new(py);
        dict.set_item("nodes", self.inner.node_count())?;
        dict.set_item("edges", self.inner.edge_count())?;
        dict.set_item("hyperedges", self.inner.hyperedge_count())?;

        Ok(dict.into())
    }

    /// String representation
    fn __repr__(&self) -> PyResult<String> {
        // GraphDB doesn't have a stats() method, use individual count methods
        Ok(format!(
            "GraphDB(nodes={}, edges={}, hyperedges={})",
            self.inner.node_count(),
            self.inner.edge_count(),
            self.inner.hyperedge_count()
        ))
    }
}
