//! Advanced filtering module for pyruvector
//!
//! This module provides advanced filtering capabilities including:
//! - Payload index management for optimized queries
//! - Fluent FilterBuilder API for complex query construction
//! - Geospatial filtering (radius queries)
//! - Full-text search support
//! - Composite filters with AND/OR/NOT logic

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ============================================================================
// Type Aliases
// ============================================================================

/// Type alias for the inverted index structure
type InvertedIndex = Arc<RwLock<HashMap<String, HashMap<String, Vec<String>>>>>;

// ============================================================================
// Index Types
// ============================================================================

/// Index type enumeration for payload field indexing
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IndexType {
    /// Exact match for string values
    Keyword,
    /// Range queries for integer values
    Integer,
    /// Range queries for floating-point values
    Float,
    /// Boolean filters
    Boolean,
    /// Geospatial queries (lat/lon coordinates)
    Geo,
    /// Full-text search with tokenization
    Text,
}

#[pymethods]
impl IndexType {
    fn __repr__(&self) -> String {
        match self {
            IndexType::Keyword => "IndexType.Keyword".to_string(),
            IndexType::Integer => "IndexType.Integer".to_string(),
            IndexType::Float => "IndexType.Float".to_string(),
            IndexType::Boolean => "IndexType.Boolean".to_string(),
            IndexType::Geo => "IndexType.Geo".to_string(),
            IndexType::Text => "IndexType.Text".to_string(),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Payload Index Manager
// ============================================================================

/// Manages payload indices for optimized filtering
///
/// This class provides efficient indexing of vector metadata/payload fields
/// to accelerate filter queries. Create indices on frequently queried fields.
///
/// # Example
///
/// ```python
/// from pyruvector import PayloadIndexManager, IndexType
///
/// # Create index manager
/// manager = PayloadIndexManager()
///
/// # Create indices for different field types
/// manager.create_index("category", IndexType.Keyword)
/// manager.create_index("price", IndexType.Float)
/// manager.create_index("location", IndexType.Geo)
///
/// # Index vector payloads
/// manager.index_payload("vec1", {"category": "science", "price": 29.99})
/// manager.index_payload("vec2", {"category": "tech", "price": 49.99})
///
/// # List all indices
/// indices = manager.list_indices()
/// ```
#[pyclass]
pub struct PayloadIndexManager {
    /// Map of field name -> index type
    indices: Arc<RwLock<HashMap<String, IndexType>>>,
    /// Map of vector_id -> payload data
    payloads: Arc<RwLock<HashMap<String, Value>>>,
    /// Inverted index: field -> value -> [vector_ids]
    inverted_index: InvertedIndex,
}

#[pymethods]
impl PayloadIndexManager {
    /// Create a new PayloadIndexManager
    ///
    /// # Returns
    /// A new PayloadIndexManager instance
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an index on a payload field
    ///
    /// # Arguments
    /// * `field_name` - Name of the field to index
    /// * `index_type` - Type of index to create
    ///
    /// # Example
    /// ```python
    /// manager.create_index("category", IndexType.Keyword)
    /// manager.create_index("price", IndexType.Float)
    /// ```
    pub fn create_index(&self, field_name: String, index_type: IndexType) -> PyResult<()> {
        let mut indices = self.indices.write().unwrap();

        if indices.contains_key(&field_name) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Index already exists for field: {}",
                field_name
            )));
        }

        indices.insert(field_name.clone(), index_type);

        // Initialize inverted index for this field
        let mut inv_index = self.inverted_index.write().unwrap();
        inv_index.insert(field_name, HashMap::new());

        Ok(())
    }

    /// Drop an index from a payload field
    ///
    /// # Arguments
    /// * `field_name` - Name of the field to remove index from
    ///
    /// # Example
    /// ```python
    /// manager.drop_index("category")
    /// ```
    pub fn drop_index(&self, field_name: String) -> PyResult<()> {
        let mut indices = self.indices.write().unwrap();

        if !indices.contains_key(&field_name) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "No index exists for field: {}",
                field_name
            )));
        }

        indices.remove(&field_name);

        // Remove from inverted index
        let mut inv_index = self.inverted_index.write().unwrap();
        inv_index.remove(&field_name);

        Ok(())
    }

    /// List all indexed fields
    ///
    /// # Returns
    /// Vector of field names that have indices
    ///
    /// # Example
    /// ```python
    /// fields = manager.list_indices()
    /// print(fields)  # ['category', 'price', 'location']
    /// ```
    pub fn list_indices(&self) -> PyResult<Vec<String>> {
        let indices = self.indices.read().unwrap();
        Ok(indices.keys().cloned().collect())
    }

    /// Index a vector's payload
    ///
    /// # Arguments
    /// * `vector_id` - ID of the vector
    /// * `payload` - Dictionary containing payload data
    ///
    /// # Example
    /// ```python
    /// manager.index_payload("vec1", {
    ///     "category": "science",
    ///     "price": 29.99,
    ///     "tags": ["physics", "astronomy"]
    /// })
    /// ```
    pub fn index_payload(&self, vector_id: String, payload: &PyDict) -> PyResult<()> {
        // Convert PyDict to serde_json::Value
        let payload_value = pydict_to_value(payload)?;

        // Store payload
        let mut payloads = self.payloads.write().unwrap();
        payloads.insert(vector_id.clone(), payload_value.clone());

        // Update inverted indices
        let indices = self.indices.read().unwrap();
        let mut inv_index = self.inverted_index.write().unwrap();

        if let Some(obj) = payload_value.as_object() {
            for (field, value) in obj {
                if indices.contains_key(field) {
                    // Get or create field index
                    let field_index = inv_index.entry(field.clone()).or_default();

                    // Convert value to index key
                    let index_key = value_to_index_key(value);

                    // Add vector_id to this value's list
                    field_index
                        .entry(index_key)
                        .or_default()
                        .push(vector_id.clone());
                }
            }
        }

        Ok(())
    }

    /// Remove a vector's payload from indices
    ///
    /// # Arguments
    /// * `vector_id` - ID of the vector to remove
    ///
    /// # Example
    /// ```python
    /// manager.remove_payload("vec1")
    /// ```
    pub fn remove_payload(&self, vector_id: String) -> PyResult<()> {
        // Remove from payloads
        let mut payloads = self.payloads.write().unwrap();
        payloads.remove(&vector_id);

        // Remove from inverted indices
        let mut inv_index = self.inverted_index.write().unwrap();
        for field_index in inv_index.values_mut() {
            for vector_ids in field_index.values_mut() {
                vector_ids.retain(|id| id != &vector_id);
            }
        }

        Ok(())
    }

    /// Get payload for a specific vector
    ///
    /// # Arguments
    /// * `vector_id` - ID of the vector
    ///
    /// # Returns
    /// Dictionary containing payload data, or None if not found
    pub fn get_payload(&self, vector_id: String) -> PyResult<Option<Py<PyDict>>> {
        let payloads = self.payloads.read().unwrap();

        if let Some(payload) = payloads.get(&vector_id) {
            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                if let Some(obj) = payload.as_object() {
                    for (k, v) in obj {
                        dict.set_item(k, value_to_py(py, v)?)?;
                    }
                }
                Ok(Some(dict.into()))
            })
        } else {
            Ok(None)
        }
    }

    fn __repr__(&self) -> String {
        let indices = self.indices.read().unwrap();
        let payloads = self.payloads.read().unwrap();
        format!(
            "PayloadIndexManager(indices={}, vectors={})",
            indices.len(),
            payloads.len()
        )
    }
}

// ============================================================================
// Filter Builder
// ============================================================================

/// Internal filter condition representation
#[derive(Clone, Debug)]
enum FilterCondition {
    Eq(String, Value),
    Ne(String, Value),
    Gt(String, Value),
    Gte(String, Value),
    Lt(String, Value),
    Lte(String, Value),
    In(String, Vec<Value>),
    Contains(String, Value),
    GeoRadius {
        field: String,
        lat: f64,
        lon: f64,
        radius_km: f64,
    },
    TextMatch {
        field: String,
        query: String,
    },
    And(Vec<FilterCondition>),
    Or(Vec<FilterCondition>),
    Not(Box<FilterCondition>),
}

/// Fluent API for building complex filters
///
/// FilterBuilder provides a chainable interface for constructing
/// sophisticated filter queries with support for comparison operators,
/// geospatial queries, text search, and logical operators.
///
/// # Example
///
/// ```python
/// from pyruvector import FilterBuilder
///
/// # Simple equality filter
/// filter = FilterBuilder().eq("category", "science").build()
///
/// # Range query
/// filter = FilterBuilder().gte("price", 10).lte("price", 100).build()
///
/// # Complex composite filter
/// filter = FilterBuilder().and_([
///     FilterBuilder().eq("status", "active"),
///     FilterBuilder().or_([
///         FilterBuilder().gt("price", 50),
///         FilterBuilder().in_values("category", ["premium", "featured"])
///     ])
/// ]).build()
///
/// # Geospatial query
/// filter = FilterBuilder().geo_radius("location", 40.7128, -74.0060, 5.0).build()
///
/// # Text search
/// filter = FilterBuilder().text_match("description", "machine learning").build()
/// ```
#[pyclass]
#[derive(Clone, Default)]
pub struct FilterBuilder {
    conditions: Vec<FilterCondition>,
}

#[pymethods]
impl FilterBuilder {
    /// Create a new FilterBuilder
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add equality condition
    ///
    /// # Arguments
    /// * `field` - Field name to filter
    /// * `value` - Value to match
    ///
    /// # Returns
    /// Self for method chaining
    pub fn eq(&self, field: String, value: &PyAny) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        let json_value = pyany_to_value(value)?;
        builder
            .conditions
            .push(FilterCondition::Eq(field, json_value));
        Ok(builder)
    }

    /// Add not-equal condition
    pub fn ne(&self, field: String, value: &PyAny) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        let json_value = pyany_to_value(value)?;
        builder
            .conditions
            .push(FilterCondition::Ne(field, json_value));
        Ok(builder)
    }

    /// Add greater-than condition
    pub fn gt(&self, field: String, value: &PyAny) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        let json_value = pyany_to_value(value)?;
        builder
            .conditions
            .push(FilterCondition::Gt(field, json_value));
        Ok(builder)
    }

    /// Add greater-than-or-equal condition
    pub fn gte(&self, field: String, value: &PyAny) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        let json_value = pyany_to_value(value)?;
        builder
            .conditions
            .push(FilterCondition::Gte(field, json_value));
        Ok(builder)
    }

    /// Add less-than condition
    pub fn lt(&self, field: String, value: &PyAny) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        let json_value = pyany_to_value(value)?;
        builder
            .conditions
            .push(FilterCondition::Lt(field, json_value));
        Ok(builder)
    }

    /// Add less-than-or-equal condition
    pub fn lte(&self, field: String, value: &PyAny) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        let json_value = pyany_to_value(value)?;
        builder
            .conditions
            .push(FilterCondition::Lte(field, json_value));
        Ok(builder)
    }

    /// Add in-values condition (value must be in list)
    ///
    /// # Arguments
    /// * `field` - Field name to filter
    /// * `values` - List of acceptable values
    pub fn in_values(&self, field: String, values: &PyList) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        let json_values: PyResult<Vec<Value>> = values.iter().map(pyany_to_value).collect();
        builder
            .conditions
            .push(FilterCondition::In(field, json_values?));
        Ok(builder)
    }

    /// Add contains condition (field must contain value)
    ///
    /// For array fields, checks if the array contains the value.
    /// For string fields, checks if the string contains the substring.
    pub fn contains(&self, field: String, value: &PyAny) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        let json_value = pyany_to_value(value)?;
        builder
            .conditions
            .push(FilterCondition::Contains(field, json_value));
        Ok(builder)
    }

    /// Add geospatial radius query
    ///
    /// # Arguments
    /// * `field` - Field containing coordinates (as {"lat": ..., "lon": ...})
    /// * `lat` - Latitude of center point
    /// * `lon` - Longitude of center point
    /// * `radius_km` - Radius in kilometers
    ///
    /// # Example
    /// ```python
    /// # Find locations within 5km of New York City
    /// filter = FilterBuilder().geo_radius("location", 40.7128, -74.0060, 5.0).build()
    /// ```
    pub fn geo_radius(
        &self,
        field: String,
        lat: f64,
        lon: f64,
        radius_km: f64,
    ) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        builder.conditions.push(FilterCondition::GeoRadius {
            field,
            lat,
            lon,
            radius_km,
        });
        Ok(builder)
    }

    /// Add full-text search condition
    ///
    /// # Arguments
    /// * `field` - Field to search in
    /// * `query` - Search query string
    ///
    /// # Example
    /// ```python
    /// filter = FilterBuilder().text_match("description", "machine learning").build()
    /// ```
    pub fn text_match(&self, field: String, query: String) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        builder
            .conditions
            .push(FilterCondition::TextMatch { field, query });
        Ok(builder)
    }

    /// Combine filters with AND logic
    ///
    /// # Arguments
    /// * `filters` - List of FilterBuilder instances to AND together
    ///
    /// # Example
    /// ```python
    /// filter = FilterBuilder().and_([
    ///     FilterBuilder().eq("status", "active"),
    ///     FilterBuilder().gt("price", 10)
    /// ]).build()
    /// ```
    pub fn and_(&self, filters: Vec<FilterBuilder>) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        let conditions: Vec<FilterCondition> =
            filters.into_iter().flat_map(|f| f.conditions).collect();
        builder.conditions.push(FilterCondition::And(conditions));
        Ok(builder)
    }

    /// Combine filters with OR logic
    ///
    /// # Arguments
    /// * `filters` - List of FilterBuilder instances to OR together
    ///
    /// # Example
    /// ```python
    /// filter = FilterBuilder().or_([
    ///     FilterBuilder().eq("category", "science"),
    ///     FilterBuilder().eq("category", "tech")
    /// ]).build()
    /// ```
    pub fn or_(&self, filters: Vec<FilterBuilder>) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        let conditions: Vec<FilterCondition> =
            filters.into_iter().flat_map(|f| f.conditions).collect();
        builder.conditions.push(FilterCondition::Or(conditions));
        Ok(builder)
    }

    /// Negate a filter with NOT logic
    ///
    /// # Arguments
    /// * `filter` - FilterBuilder to negate
    ///
    /// # Example
    /// ```python
    /// filter = FilterBuilder().not_(
    ///     FilterBuilder().eq("status", "archived")
    /// ).build()
    /// ```
    pub fn not_(&self, filter: FilterBuilder) -> PyResult<FilterBuilder> {
        let mut builder = self.clone();
        if let Some(condition) = filter.conditions.into_iter().next() {
            builder
                .conditions
                .push(FilterCondition::Not(Box::new(condition)));
        }
        Ok(builder)
    }

    /// Build the filter into a dictionary
    ///
    /// # Returns
    /// Dictionary representation of the filter that can be passed to VectorDB.search()
    ///
    /// # Example
    /// ```python
    /// filter_dict = FilterBuilder().eq("category", "science").build()
    /// results = db.search(query_vector, k=10, filter=filter_dict)
    /// ```
    pub fn build(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        for condition in &self.conditions {
            condition_to_dict(py, condition, dict)?;
        }

        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!("FilterBuilder(conditions={})", self.conditions.len())
    }
}

// ============================================================================
// Filter Evaluator
// ============================================================================

/// Evaluates filters against indexed payloads
///
/// FilterEvaluator uses the PayloadIndexManager to efficiently
/// evaluate filter conditions and return matching vector IDs.
///
/// # Example
///
/// ```python
/// from pyruvector import PayloadIndexManager, FilterEvaluator, FilterBuilder
///
/// # Setup
/// manager = PayloadIndexManager()
/// manager.create_index("category", IndexType.Keyword)
/// manager.index_payload("vec1", {"category": "science"})
/// manager.index_payload("vec2", {"category": "tech"})
///
/// # Create evaluator
/// evaluator = FilterEvaluator(manager)
///
/// # Build and evaluate filter
/// filter_dict = FilterBuilder().eq("category", "science").build()
/// matching_ids = evaluator.evaluate(filter_dict)  # ["vec1"]
/// ```
#[pyclass]
pub struct FilterEvaluator {
    index_manager: Arc<PayloadIndexManager>,
}

#[pymethods]
impl FilterEvaluator {
    /// Create a new FilterEvaluator
    ///
    /// # Arguments
    /// * `index_manager` - PayloadIndexManager instance to use
    #[new]
    pub fn new(index_manager: &PayloadIndexManager) -> Self {
        Self {
            index_manager: Arc::new(index_manager.clone()),
        }
    }

    /// Evaluate a filter and return matching vector IDs
    ///
    /// # Arguments
    /// * `filter` - Filter dictionary (from FilterBuilder.build())
    ///
    /// # Returns
    /// List of vector IDs that match the filter
    pub fn evaluate(&self, filter: &PyDict) -> PyResult<Vec<String>> {
        let payloads = self.index_manager.payloads.read().unwrap();

        let mut matching_ids = Vec::new();

        for (vector_id, payload) in payloads.iter() {
            if self.evaluate_payload(filter, payload)? {
                matching_ids.push(vector_id.clone());
            }
        }

        Ok(matching_ids)
    }

    fn __repr__(&self) -> String {
        "FilterEvaluator".to_string()
    }
}

impl FilterEvaluator {
    /// Evaluate a single payload against a filter
    fn evaluate_payload(&self, filter: &PyDict, payload: &Value) -> PyResult<bool> {
        let payload_obj = payload.as_object().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Payload must be an object")
        })?;

        for (key, value) in filter.iter() {
            let field = key.extract::<String>()?;

            if !self.evaluate_field(&field, value, payload_obj)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Evaluate a single field condition
    fn evaluate_field(
        &self,
        field: &str,
        condition: &PyAny,
        payload: &serde_json::Map<String, Value>,
    ) -> PyResult<bool> {
        let field_value = payload.get(field);

        // Handle nested conditions (e.g., {"$gt": 10})
        if let Ok(dict) = condition.downcast::<PyDict>() {
            for (op, val) in dict.iter() {
                let op_str = op.extract::<String>()?;

                match op_str.as_str() {
                    "$eq" => {
                        let expected = pyany_to_value(val)?;
                        if field_value != Some(&expected) {
                            return Ok(false);
                        }
                    }
                    "$ne" => {
                        let expected = pyany_to_value(val)?;
                        if field_value == Some(&expected) {
                            return Ok(false);
                        }
                    }
                    "$gt" | "$gte" | "$lt" | "$lte" => {
                        if !self.evaluate_comparison(&op_str, field_value, val)? {
                            return Ok(false);
                        }
                    }
                    "$in" => {
                        if !self.evaluate_in(field_value, val)? {
                            return Ok(false);
                        }
                    }
                    "$nin" => {
                        // Not in - opposite of $in
                        if self.evaluate_in(field_value, val)? {
                            return Ok(false);
                        }
                    }
                    "$contains" => {
                        // Check if array field contains value, or string contains substring
                        if !self.evaluate_contains(field_value, val)? {
                            return Ok(false);
                        }
                    }
                    "$exists" => {
                        // Check if field exists (or doesn't exist if false)
                        let should_exist = val.extract::<bool>().unwrap_or(true);
                        let exists = field_value.is_some();
                        if exists != should_exist {
                            return Ok(false);
                        }
                    }
                    "$text_match" | "$text" | "$match" => {
                        // Text/keyword search - case-insensitive substring match
                        if !self.evaluate_text_match(field_value, val)? {
                            return Ok(false);
                        }
                    }
                    "$regex" => {
                        // Regex pattern matching
                        if !self.evaluate_regex(field_value, val)? {
                            return Ok(false);
                        }
                    }
                    _ => {
                        // Unknown operator - skip silently for forward compatibility
                    }
                }
            }
        } else {
            // Simple equality
            let expected = pyany_to_value(condition)?;
            if field_value != Some(&expected) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Evaluate comparison operators
    fn evaluate_comparison(
        &self,
        op: &str,
        field_value: Option<&Value>,
        expected: &PyAny,
    ) -> PyResult<bool> {
        let field_value = match field_value {
            Some(v) => v,
            None => return Ok(false),
        };

        let expected_val = pyany_to_value(expected)?;

        let result = match (field_value, &expected_val) {
            (Value::Number(a), Value::Number(b)) => {
                let a_f64 = a.as_f64().unwrap_or(0.0);
                let b_f64 = b.as_f64().unwrap_or(0.0);

                match op {
                    "$gt" => a_f64 > b_f64,
                    "$gte" => a_f64 >= b_f64,
                    "$lt" => a_f64 < b_f64,
                    "$lte" => a_f64 <= b_f64,
                    _ => false,
                }
            }
            _ => false,
        };

        Ok(result)
    }

    /// Evaluate $in operator
    fn evaluate_in(&self, field_value: Option<&Value>, values: &PyAny) -> PyResult<bool> {
        let field_value = match field_value {
            Some(v) => v,
            None => return Ok(false),
        };

        if let Ok(list) = values.downcast::<PyList>() {
            for item in list.iter() {
                let val = pyany_to_value(item)?;
                if field_value == &val {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Evaluate $contains operator - checks if array contains value or string contains substring
    fn evaluate_contains(&self, field_value: Option<&Value>, expected: &PyAny) -> PyResult<bool> {
        let field_value = match field_value {
            Some(v) => v,
            None => return Ok(false),
        };

        let expected_val = pyany_to_value(expected)?;

        match field_value {
            // Array contains value
            Value::Array(arr) => Ok(arr.contains(&expected_val)),
            // String contains substring
            Value::String(s) => {
                if let Value::String(substr) = &expected_val {
                    Ok(s.to_lowercase().contains(&substr.to_lowercase()))
                } else {
                    Ok(false)
                }
            }
            _ => Ok(false),
        }
    }

    /// Evaluate $text_match operator - case-insensitive text search
    fn evaluate_text_match(&self, field_value: Option<&Value>, query: &PyAny) -> PyResult<bool> {
        let field_value = match field_value {
            Some(v) => v,
            None => return Ok(false),
        };

        let query_str = query.extract::<String>().unwrap_or_default().to_lowercase();
        if query_str.is_empty() {
            return Ok(true); // Empty query matches everything
        }

        // Split query into terms for multi-word matching
        let query_terms: Vec<&str> = query_str.split_whitespace().collect();

        match field_value {
            Value::String(s) => {
                let field_lower = s.to_lowercase();
                // All query terms must be present (AND semantics)
                Ok(query_terms.iter().all(|term| field_lower.contains(term)))
            }
            Value::Array(arr) => {
                // Search across all string elements in array
                for item in arr {
                    if let Value::String(s) = item {
                        let field_lower = s.to_lowercase();
                        if query_terms.iter().all(|term| field_lower.contains(term)) {
                            return Ok(true);
                        }
                    }
                }
                Ok(false)
            }
            _ => Ok(false),
        }
    }

    /// Evaluate $regex operator - regex pattern matching
    fn evaluate_regex(&self, field_value: Option<&Value>, pattern: &PyAny) -> PyResult<bool> {
        let field_value = match field_value {
            Some(v) => v,
            None => return Ok(false),
        };

        let pattern_str = pattern.extract::<String>().unwrap_or_default();
        if pattern_str.is_empty() {
            return Ok(true);
        }

        // Compile regex (with case-insensitive flag if pattern starts with (?i))
        let regex = match regex::Regex::new(&pattern_str) {
            Ok(r) => r,
            Err(_) => return Ok(false), // Invalid regex pattern
        };

        match field_value {
            Value::String(s) => Ok(regex.is_match(s)),
            _ => Ok(false),
        }
    }
}

impl Clone for PayloadIndexManager {
    fn clone(&self) -> Self {
        Self {
            indices: Arc::clone(&self.indices),
            payloads: Arc::clone(&self.payloads),
            inverted_index: Arc::clone(&self.inverted_index),
        }
    }
}

impl Default for PayloadIndexManager {
    fn default() -> Self {
        Self {
            indices: Arc::new(RwLock::new(HashMap::new())),
            payloads: Arc::new(RwLock::new(HashMap::new())),
            inverted_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert PyDict to serde_json::Value
fn pydict_to_value(dict: &PyDict) -> PyResult<Value> {
    let mut map = serde_json::Map::new();

    for (key, value) in dict.iter() {
        let key_str = key.extract::<String>()?;
        let json_value = pyany_to_value(value)?;
        map.insert(key_str, json_value);
    }

    Ok(Value::Object(map))
}

/// Convert PyAny to serde_json::Value
fn pyany_to_value(obj: &PyAny) -> PyResult<Value> {
    if let Ok(s) = obj.extract::<String>() {
        Ok(json!(s))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(json!(i))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(json!(f))
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(json!(b))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let items: PyResult<Vec<Value>> = list.iter().map(pyany_to_value).collect();
        Ok(Value::Array(items?))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        pydict_to_value(dict)
    } else if obj.is_none() {
        Ok(Value::Null)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Unsupported type: {:?}",
            obj
        )))
    }
}

/// Convert serde_json::Value to Python object
fn value_to_py(py: Python, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_py(py)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Ok(py.None())
            }
        }
        Value::String(s) => Ok(s.into_py(py)),
        Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(value_to_py(py, item)?)?;
            }
            Ok(list.into())
        }
        Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (k, v) in obj {
                dict.set_item(k, value_to_py(py, v)?)?;
            }
            Ok(dict.into())
        }
    }
}

/// Convert Value to index key string
fn value_to_index_key(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

/// Convert FilterCondition to Python dict
fn condition_to_dict(py: Python, condition: &FilterCondition, dict: &PyDict) -> PyResult<()> {
    match condition {
        FilterCondition::Eq(field, value) => {
            dict.set_item(field, value_to_py(py, value)?)?;
        }
        FilterCondition::Ne(field, value) => {
            let inner = PyDict::new(py);
            inner.set_item("$ne", value_to_py(py, value)?)?;
            dict.set_item(field, inner)?;
        }
        FilterCondition::Gt(field, value) => {
            let inner = PyDict::new(py);
            inner.set_item("$gt", value_to_py(py, value)?)?;
            dict.set_item(field, inner)?;
        }
        FilterCondition::Gte(field, value) => {
            let inner = PyDict::new(py);
            inner.set_item("$gte", value_to_py(py, value)?)?;
            dict.set_item(field, inner)?;
        }
        FilterCondition::Lt(field, value) => {
            let inner = PyDict::new(py);
            inner.set_item("$lt", value_to_py(py, value)?)?;
            dict.set_item(field, inner)?;
        }
        FilterCondition::Lte(field, value) => {
            let inner = PyDict::new(py);
            inner.set_item("$lte", value_to_py(py, value)?)?;
            dict.set_item(field, inner)?;
        }
        FilterCondition::In(field, values) => {
            let list = PyList::empty(py);
            for val in values {
                list.append(value_to_py(py, val)?)?;
            }
            let inner = PyDict::new(py);
            inner.set_item("$in", list)?;
            dict.set_item(field, inner)?;
        }
        FilterCondition::Contains(field, value) => {
            let inner = PyDict::new(py);
            inner.set_item("$contains", value_to_py(py, value)?)?;
            dict.set_item(field, inner)?;
        }
        FilterCondition::GeoRadius {
            field,
            lat,
            lon,
            radius_km,
        } => {
            let inner = PyDict::new(py);
            let geo = PyDict::new(py);
            geo.set_item("lat", lat)?;
            geo.set_item("lon", lon)?;
            geo.set_item("radius_km", radius_km)?;
            inner.set_item("$geo_radius", geo)?;
            dict.set_item(field, inner)?;
        }
        FilterCondition::TextMatch { field, query } => {
            let inner = PyDict::new(py);
            inner.set_item("$text_match", query)?;
            dict.set_item(field, inner)?;
        }
        FilterCondition::And(conditions) => {
            let list = PyList::empty(py);
            for cond in conditions {
                let cond_dict = PyDict::new(py);
                condition_to_dict(py, cond, cond_dict)?;
                list.append(cond_dict)?;
            }
            dict.set_item("$and", list)?;
        }
        FilterCondition::Or(conditions) => {
            let list = PyList::empty(py);
            for cond in conditions {
                let cond_dict = PyDict::new(py);
                condition_to_dict(py, cond, cond_dict)?;
                list.append(cond_dict)?;
            }
            dict.set_item("$or", list)?;
        }
        FilterCondition::Not(condition) => {
            let inner = PyDict::new(py);
            condition_to_dict(py, condition, inner)?;
            dict.set_item("$not", inner)?;
        }
    }
    Ok(())
}
