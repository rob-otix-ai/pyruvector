use pyo3::exceptions::{PyIOError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

/// Main error type for PyRuVector operations
#[derive(Error, Debug)]
pub enum PyRuVectorError {
    /// Vector dimension mismatch
    #[error("Invalid dimensions: expected {expected}, got {got}")]
    InvalidDimensions { expected: usize, got: usize },

    /// Invalid filter expression or parameters
    #[error("Invalid filter: {message}")]
    InvalidFilter { message: String },

    /// Vector or item not found
    #[error("Not found: {id}")]
    NotFound { id: String },

    /// Persistence or I/O error
    #[error("Persistence error: {message}")]
    PersistenceError { message: String },

    /// Internal error
    #[error("Internal error: {message}")]
    InternalError { message: String },

    /// Serialization/deserialization error
    #[error("Serialization error: {message}")]
    SerializationError { message: String },
}

/// Convert PyRuVectorError to Python exceptions
impl From<PyRuVectorError> for PyErr {
    fn from(err: PyRuVectorError) -> PyErr {
        match err {
            PyRuVectorError::InvalidDimensions { expected, got } => PyValueError::new_err(format!(
                "Invalid dimensions: expected {}, got {}",
                expected, got
            )),
            PyRuVectorError::InvalidFilter { message } => {
                PyValueError::new_err(format!("Invalid filter: {}", message))
            }
            PyRuVectorError::NotFound { id } => PyKeyError::new_err(format!("Not found: {}", id)),
            PyRuVectorError::PersistenceError { message } => {
                PyIOError::new_err(format!("Persistence error: {}", message))
            }
            PyRuVectorError::InternalError { message } => {
                PyRuntimeError::new_err(format!("Internal error: {}", message))
            }
            PyRuVectorError::SerializationError { message } => {
                PyValueError::new_err(format!("Serialization error: {}", message))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_error_display() {
        let err = PyRuVectorError::InvalidDimensions {
            expected: 128,
            got: 64,
        };
        assert_eq!(err.to_string(), "Invalid dimensions: expected 128, got 64");

        let err = PyRuVectorError::NotFound {
            id: "vec_123".to_string(),
        };
        assert_eq!(err.to_string(), "Not found: vec_123");
    }

    #[test]
    fn test_error_conversion() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let err = PyRuVectorError::InvalidDimensions {
                expected: 128,
                got: 64,
            };
            let py_err: PyErr = err.into();
            assert!(py_err.is_instance_of::<PyValueError>(py));

            let err = PyRuVectorError::NotFound {
                id: "test".to_string(),
            };
            let py_err: PyErr = err.into();
            assert!(py_err.is_instance_of::<PyKeyError>(py));

            let err = PyRuVectorError::PersistenceError {
                message: "disk full".to_string(),
            };
            let py_err: PyErr = err.into();
            assert!(py_err.is_instance_of::<PyIOError>(py));

            let err = PyRuVectorError::InternalError {
                message: "panic".to_string(),
            };
            let py_err: PyErr = err.into();
            assert!(py_err.is_instance_of::<PyRuntimeError>(py));
        });
    }
}
