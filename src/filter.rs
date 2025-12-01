use serde_json::Value;
use std::collections::HashMap;

/// Filter operators for metadata filtering
#[derive(Debug, Clone, PartialEq)]
pub enum FilterOperator {
    /// Equality comparison
    Eq,
    /// Not equal comparison
    Ne,
    /// Greater than
    Gt,
    /// Greater than or equal
    Gte,
    /// Less than
    Lt,
    /// Less than or equal
    Lte,
    /// Value in set
    In,
    /// Value not in set
    Nin,
    /// Array contains value
    Contains,
    /// Field exists
    Exists,
}

impl FilterOperator {
    /// Parse operator from string key (e.g., "$eq", "$gt")
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "$eq" => Some(FilterOperator::Eq),
            "$ne" => Some(FilterOperator::Ne),
            "$gt" => Some(FilterOperator::Gt),
            "$gte" => Some(FilterOperator::Gte),
            "$lt" => Some(FilterOperator::Lt),
            "$lte" => Some(FilterOperator::Lte),
            "$in" => Some(FilterOperator::In),
            "$nin" => Some(FilterOperator::Nin),
            "$contains" => Some(FilterOperator::Contains),
            "$exists" => Some(FilterOperator::Exists),
            _ => None,
        }
    }
}

/// A single filter condition
#[derive(Debug, Clone)]
pub struct FilterCondition {
    /// Field name to filter on
    pub field: String,
    /// Comparison operator
    pub operator: FilterOperator,
    /// Value to compare against
    pub value: Value,
}

impl FilterCondition {
    /// Create a new filter condition
    pub fn new(field: String, operator: FilterOperator, value: Value) -> Self {
        Self {
            field,
            operator,
            value,
        }
    }

    /// Evaluate this condition against metadata
    pub fn evaluate(&self, metadata: &HashMap<String, Value>) -> bool {
        let field_value = metadata.get(&self.field);

        match self.operator {
            FilterOperator::Exists => {
                let expected = self.value.as_bool().unwrap_or(true);
                field_value.is_some() == expected
            }
            FilterOperator::Eq => field_value == Some(&self.value),
            FilterOperator::Ne => field_value != Some(&self.value),
            FilterOperator::Gt => field_value.is_some_and(|v| {
                compare_values(v, &self.value) == Some(std::cmp::Ordering::Greater)
            }),
            FilterOperator::Gte => field_value.is_some_and(|v| {
                matches!(
                    compare_values(v, &self.value),
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                )
            }),
            FilterOperator::Lt => field_value
                .is_some_and(|v| compare_values(v, &self.value) == Some(std::cmp::Ordering::Less)),
            FilterOperator::Lte => field_value.is_some_and(|v| {
                matches!(
                    compare_values(v, &self.value),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )
            }),
            FilterOperator::In => {
                if let Some(arr) = self.value.as_array() {
                    field_value.is_some_and(|v| arr.contains(v))
                } else {
                    false
                }
            }
            FilterOperator::Nin => {
                if let Some(arr) = self.value.as_array() {
                    field_value.map_or(true, |v| !arr.contains(v))
                } else {
                    true
                }
            }
            FilterOperator::Contains => {
                if let Some(field_arr) = field_value.and_then(|v| v.as_array()) {
                    field_arr.contains(&self.value)
                } else {
                    false
                }
            }
        }
    }
}

/// Compare two JSON values for ordering
fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Number(n1), Value::Number(n2)) => {
            let f1 = n1.as_f64()?;
            let f2 = n2.as_f64()?;
            f1.partial_cmp(&f2)
        }
        (Value::String(s1), Value::String(s2)) => Some(s1.cmp(s2)),
        (Value::Bool(b1), Value::Bool(b2)) => Some(b1.cmp(b2)),
        _ => None,
    }
}

/// Parse a filter dictionary into filter conditions
///
/// # Arguments
/// * `filter_dict` - HashMap from Python containing filter specifications
///
/// # Returns
/// Vector of FilterCondition objects
///
/// # Examples
/// ```
/// // Simple equality: {"category": "science"}
/// // Becomes: FilterCondition { field: "category", operator: Eq, value: "science" }
///
/// // Comparison: {"price": {"$gt": 10}}
/// // Becomes: FilterCondition { field: "price", operator: Gt, value: 10 }
///
/// // Multiple operators: {"status": {"$in": ["active", "pending"]}}
/// // Becomes: FilterCondition { field: "status", operator: In, value: ["active", "pending"] }
/// ```
pub fn parse_filter(filter_dict: HashMap<String, Value>) -> Vec<FilterCondition> {
    let mut conditions = Vec::new();

    for (field, value) in filter_dict {
        if let Some(obj) = value.as_object() {
            // Value is an object with operators like {"$gt": 10, "$lt": 20}
            for (op_key, op_value) in obj {
                if let Some(operator) = FilterOperator::from_str(op_key) {
                    conditions.push(FilterCondition::new(
                        field.clone(),
                        operator,
                        op_value.clone(),
                    ));
                }
            }
        } else {
            // Simple value - implicit $eq operator
            conditions.push(FilterCondition::new(field, FilterOperator::Eq, value));
        }
    }

    conditions
}

/// Evaluate all filter conditions against metadata
///
/// # Arguments
/// * `conditions` - Slice of filter conditions (AND-ed together)
/// * `metadata` - Metadata to evaluate against
///
/// # Returns
/// True if all conditions match, false otherwise
pub fn evaluate_filter(conditions: &[FilterCondition], metadata: &HashMap<String, Value>) -> bool {
    conditions.iter().all(|cond| cond.evaluate(metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_metadata(pairs: Vec<(&str, Value)>) -> HashMap<String, Value> {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    #[test]
    fn test_equality_operator() {
        let metadata = make_metadata(vec![("category", json!("science"))]);
        let filter = parse_filter(
            vec![("category".to_string(), json!("science"))]
                .into_iter()
                .collect(),
        );
        assert!(evaluate_filter(&filter, &metadata));

        let filter = parse_filter(
            vec![("category".to_string(), json!("tech"))]
                .into_iter()
                .collect(),
        );
        assert!(!evaluate_filter(&filter, &metadata));
    }

    #[test]
    fn test_not_equal_operator() {
        let metadata = make_metadata(vec![("status", json!("active"))]);
        let filter = parse_filter(
            vec![("status".to_string(), json!({"$ne": "inactive"}))]
                .into_iter()
                .collect(),
        );
        assert!(evaluate_filter(&filter, &metadata));

        let filter = parse_filter(
            vec![("status".to_string(), json!({"$ne": "active"}))]
                .into_iter()
                .collect(),
        );
        assert!(!evaluate_filter(&filter, &metadata));
    }

    #[test]
    fn test_comparison_operators() {
        let metadata = make_metadata(vec![("price", json!(25))]);

        // Greater than
        let filter = parse_filter(
            vec![("price".to_string(), json!({"$gt": 20}))]
                .into_iter()
                .collect(),
        );
        assert!(evaluate_filter(&filter, &metadata));

        // Greater than or equal
        let filter = parse_filter(
            vec![("price".to_string(), json!({"$gte": 25}))]
                .into_iter()
                .collect(),
        );
        assert!(evaluate_filter(&filter, &metadata));

        // Less than
        let filter = parse_filter(
            vec![("price".to_string(), json!({"$lt": 30}))]
                .into_iter()
                .collect(),
        );
        assert!(evaluate_filter(&filter, &metadata));

        // Less than or equal
        let filter = parse_filter(
            vec![("price".to_string(), json!({"$lte": 25}))]
                .into_iter()
                .collect(),
        );
        assert!(evaluate_filter(&filter, &metadata));
    }

    #[test]
    fn test_in_operator() {
        let metadata = make_metadata(vec![("status", json!("pending"))]);
        let filter = parse_filter(
            vec![("status".to_string(), json!({"$in": ["active", "pending"]}))]
                .into_iter()
                .collect(),
        );
        assert!(evaluate_filter(&filter, &metadata));

        let filter = parse_filter(
            vec![(
                "status".to_string(),
                json!({"$in": ["active", "completed"]}),
            )]
            .into_iter()
            .collect(),
        );
        assert!(!evaluate_filter(&filter, &metadata));
    }

    #[test]
    fn test_nin_operator() {
        let metadata = make_metadata(vec![("status", json!("archived"))]);
        let filter = parse_filter(
            vec![("status".to_string(), json!({"$nin": ["active", "pending"]}))]
                .into_iter()
                .collect(),
        );
        assert!(evaluate_filter(&filter, &metadata));
    }

    #[test]
    fn test_contains_operator() {
        let metadata = make_metadata(vec![("tags", json!(["rust", "python", "ai"]))]);
        let filter = parse_filter(
            vec![("tags".to_string(), json!({"$contains": "rust"}))]
                .into_iter()
                .collect(),
        );
        assert!(evaluate_filter(&filter, &metadata));

        let filter = parse_filter(
            vec![("tags".to_string(), json!({"$contains": "javascript"}))]
                .into_iter()
                .collect(),
        );
        assert!(!evaluate_filter(&filter, &metadata));
    }

    #[test]
    fn test_exists_operator() {
        let metadata = make_metadata(vec![("name", json!("test"))]);

        let filter = parse_filter(
            vec![("name".to_string(), json!({"$exists": true}))]
                .into_iter()
                .collect(),
        );
        assert!(evaluate_filter(&filter, &metadata));

        let filter = parse_filter(
            vec![("missing".to_string(), json!({"$exists": false}))]
                .into_iter()
                .collect(),
        );
        assert!(evaluate_filter(&filter, &metadata));
    }

    #[test]
    fn test_multiple_conditions() {
        let metadata = make_metadata(vec![
            ("category", json!("science")),
            ("price", json!(15)),
            ("status", json!("active")),
        ]);

        let mut filter_dict = HashMap::new();
        filter_dict.insert("category".to_string(), json!("science"));
        filter_dict.insert("price".to_string(), json!({"$gt": 10}));
        filter_dict.insert("status".to_string(), json!({"$in": ["active", "pending"]}));

        let filter = parse_filter(filter_dict);
        assert!(evaluate_filter(&filter, &metadata));
    }

    #[test]
    fn test_implicit_eq() {
        let filter_dict = vec![("name".to_string(), json!("test"))]
            .into_iter()
            .collect();

        let conditions = parse_filter(filter_dict);
        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].operator, FilterOperator::Eq);
    }

    #[test]
    fn test_multiple_operators_same_field() {
        // Range query: price > 10 AND price < 100
        let filter_dict = vec![("price".to_string(), json!({"$gt": 10, "$lt": 100}))]
            .into_iter()
            .collect();

        let conditions = parse_filter(filter_dict);
        assert_eq!(conditions.len(), 2);

        let metadata = make_metadata(vec![("price", json!(50))]);
        assert!(evaluate_filter(&conditions, &metadata));

        let metadata = make_metadata(vec![("price", json!(5))]);
        assert!(!evaluate_filter(&conditions, &metadata));
    }
}
