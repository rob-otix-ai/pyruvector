//! Graph Neural Network module
//!
//! This module provides Python bindings for Graph Neural Network (GNN) functionality,
//! wrapping ruvector-gnn and exposing comprehensive GNN capabilities to Python.
//!
//! # Type Organization
//!
//! This module contains both:
//! - **Wrapped types** from ruvector-gnn crate (production-ready)
//! - **Custom Python implementations** for simplified use (educational)
//!
//! ## Wrapped Types (from ruvector-gnn)
//!
//! These types wrap the ruvector-gnn crate for 100% API compliance:
//! - `OptimizerType` → wraps `ruvector_gnn::OptimizerType`
//! - `SchedulerType` → wraps `ruvector_gnn::SchedulerType`
//! - `RuvectorLayer` → wraps `ruvector_gnn::RuvectorLayer` (attention, GRU, layer norm)
//! - Uses `ruvector_gnn::Optimizer`, `LearningRateScheduler` internally
//!
//! ## Custom Types (Python-specific)
//!
//! These are custom implementations for simplified Python usage:
//! - `PyTrainConfig` - Python training configuration (distinct from ruvector-gnn's TrainConfig)
//! - `BasicGNNLayer` - Simplified GNN layer for educational purposes
//! - `GNNModel` - Complete model with custom training loop
//! - `Tensor` - Simple tensor wrapper
//! - `ReplayBuffer` - Experience replay buffer
//!
//! # Features
//!
//! - Graph neural network layers with various activations
//! - Complete GNN model building and training
//! - Multiple optimizer types (SGD, Adam, AdamW)
//! - Learning rate schedulers (StepDecay, CosineAnnealing, WarmupCosine)
//! - Experience replay for continual learning
//! - Tensor operations and utilities
//! - Contrastive learning with InfoNCE loss
//!
//! # Example
//!
//! ```python
//! from pyruvector import GNNModel, GNNConfig, PyTrainConfig, OptimizerType
//!
//! # Create model configuration
//! config = GNNConfig(
//!     hidden_dims=[64, 128, 64],
//!     num_layers=3,
//!     dropout=0.1,
//!     activation="relu"
//! )
//!
//! # Initialize model
//! model = GNNModel(config)
//!
//! # Train the model
//! train_config = PyTrainConfig(
//!     epochs=100,
//!     learning_rate=0.001,
//!     batch_size=32,
//!     optimizer=OptimizerType.Adam
//! )
//! metrics = model.train(dataset, train_config)
//! ```

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rand::distributions::{Distribution, Uniform};

// Import from ruvector-gnn crate
use ruvector_gnn::{
    OptimizerType as RuvectorOptimizerType,
    Optimizer as RuvectorOptimizer,
    LearningRateScheduler,
    SchedulerType as RuvectorSchedulerType,
    cosine_similarity as ruvector_cosine_similarity,
    info_nce_loss as ruvector_info_nce_loss,
    sgd_step as ruvector_sgd_step,
    RuvectorLayer as RuvectorLayerCrate,
};
use ndarray::Array2;

/// Optimizer type enumeration
///
/// Defines the available optimization algorithms for training GNN models.
///
/// # Variants
///
/// * `SGD` - Stochastic Gradient Descent
/// * `Adam` - Adaptive Moment Estimation
/// * `AdamW` - Adam with Weight Decay (Note: ruvector-gnn doesn't have AdamW, maps to Adam)
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD,
    /// Adaptive Moment Estimation
    Adam,
    /// Adam with Weight Decay (maps to Adam in ruvector-gnn)
    AdamW,
}

#[pymethods]
impl OptimizerType {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        match self {
            OptimizerType::SGD => "SGD".to_string(),
            OptimizerType::Adam => "Adam".to_string(),
            OptimizerType::AdamW => "AdamW".to_string(),
        }
    }
}

impl OptimizerType {
    /// Convert Python OptimizerType to ruvector-gnn OptimizerType
    fn to_ruvector(&self, learning_rate: f32) -> RuvectorOptimizerType {
        match self {
            OptimizerType::SGD => RuvectorOptimizerType::Sgd {
                learning_rate,
                momentum: 0.9,
            },
            OptimizerType::Adam | OptimizerType::AdamW => RuvectorOptimizerType::Adam {
                learning_rate,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
        }
    }
}

/// Learning rate scheduler type enumeration
///
/// Defines the available learning rate scheduling strategies.
///
/// # Variants
///
/// * `Constant` - Fixed learning rate throughout training
/// * `StepDecay` - Decay learning rate by a factor at specified intervals
/// * `CosineAnnealing` - Cosine annealing schedule
/// * `WarmupCosine` - Warmup followed by cosine annealing (maps to WarmupLinear in ruvector-gnn)
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SchedulerType {
    /// Constant learning rate
    Constant,
    /// Step decay with multiplicative factor
    StepDecay,
    /// Cosine annealing schedule
    CosineAnnealing,
    /// Warmup followed by cosine annealing
    WarmupCosine,
}

#[pymethods]
impl SchedulerType {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        match self {
            SchedulerType::Constant => "Constant".to_string(),
            SchedulerType::StepDecay => "StepDecay".to_string(),
            SchedulerType::CosineAnnealing => "CosineAnnealing".to_string(),
            SchedulerType::WarmupCosine => "WarmupCosine".to_string(),
        }
    }
}

impl SchedulerType {
    /// Convert Python SchedulerType to ruvector-gnn SchedulerType
    fn to_ruvector(&self, total_epochs: usize) -> RuvectorSchedulerType {
        match self {
            SchedulerType::Constant => RuvectorSchedulerType::Constant,
            SchedulerType::StepDecay => RuvectorSchedulerType::StepDecay {
                step_size: 30,
                gamma: 0.1,
            },
            SchedulerType::CosineAnnealing => RuvectorSchedulerType::CosineAnnealing {
                t_max: total_epochs,
                eta_min: 0.0,
            },
            SchedulerType::WarmupCosine => RuvectorSchedulerType::WarmupLinear {
                warmup_steps: 10,
                total_steps: total_epochs,
            },
        }
    }
}

/// GNN model configuration
///
/// Specifies the architecture and hyperparameters for a GNN model.
///
/// # Attributes
///
/// * `hidden_dims` - List of hidden layer dimensions
/// * `num_layers` - Number of GNN layers
/// * `dropout` - Dropout rate for regularization (0.0 to 1.0)
/// * `activation` - Activation function name ("relu", "tanh", "sigmoid", "leaky_relu")
///
/// # Example
///
/// ```python
/// config = GNNConfig(
///     hidden_dims=[64, 128, 64],
///     num_layers=3,
///     dropout=0.1,
///     activation="relu"
/// )
/// ```
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GNNConfig {
    /// Hidden layer dimensions
    #[pyo3(get, set)]
    pub hidden_dims: Vec<usize>,
    /// Number of GNN layers
    #[pyo3(get, set)]
    pub num_layers: usize,
    /// Dropout rate (0.0 to 1.0)
    #[pyo3(get, set)]
    pub dropout: f32,
    /// Activation function ("relu", "tanh", "sigmoid", "leaky_relu")
    #[pyo3(get, set)]
    pub activation: String,
}

#[pymethods]
impl GNNConfig {
    /// Create a new GNN configuration
    ///
    /// # Arguments
    ///
    /// * `hidden_dims` - List of hidden layer dimensions
    /// * `num_layers` - Number of GNN layers
    /// * `dropout` - Dropout rate (default: 0.0)
    /// * `activation` - Activation function (default: "relu")
    ///
    /// # Returns
    ///
    /// A new `GNNConfig` instance
    ///
    /// # Raises
    ///
    /// * `ValueError` - If dropout is not in [0.0, 1.0] or activation is invalid
    #[new]
    #[pyo3(signature = (hidden_dims, num_layers, dropout=0.0, activation="relu".to_string()))]
    fn new(
        hidden_dims: Vec<usize>,
        num_layers: usize,
        dropout: f32,
        activation: String,
    ) -> PyResult<Self> {
        if dropout < 0.0 || dropout > 1.0 {
            return Err(PyValueError::new_err(
                "Dropout must be between 0.0 and 1.0"
            ));
        }

        let valid_activations = ["relu", "tanh", "sigmoid", "leaky_relu", "gelu", "swish"];
        if !valid_activations.contains(&activation.as_str()) {
            return Err(PyValueError::new_err(
                format!("Invalid activation '{}'. Must be one of: {:?}", activation, valid_activations)
            ));
        }

        Ok(GNNConfig {
            hidden_dims,
            num_layers,
            dropout,
            activation,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "GNNConfig(hidden_dims={:?}, num_layers={}, dropout={}, activation='{}')",
            self.hidden_dims, self.num_layers, self.dropout, self.activation
        )
    }
}

/// Python Training Configuration (Custom Implementation)
///
/// This is a **custom Python-specific** training configuration, distinct from
/// ruvector-gnn's internal `TrainConfig`. Named `PyTrainConfig` to avoid conflicts.
///
/// Specifies training hyperparameters and optimization settings for Python models.
///
/// # Attributes
///
/// * `epochs` - Number of training epochs
/// * `learning_rate` - Initial learning rate
/// * `batch_size` - Batch size for mini-batch training
/// * `optimizer` - Optimizer type (SGD, Adam, AdamW)
/// * `scheduler` - Learning rate scheduler type
///
/// # Note
///
/// This config wraps ruvector-gnn's `OptimizerType` and `SchedulerType` but provides
/// a simplified Python interface.
///
/// # Example
///
/// ```python
/// config = PyTrainConfig(
///     epochs=100,
///     learning_rate=0.001,
///     batch_size=32,
///     optimizer=OptimizerType.Adam,
///     scheduler=SchedulerType.CosineAnnealing
/// )
/// ```
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyTrainConfig {
    /// Number of training epochs
    #[pyo3(get, set)]
    pub epochs: usize,
    /// Learning rate
    #[pyo3(get, set)]
    pub learning_rate: f32,
    /// Batch size
    #[pyo3(get, set)]
    pub batch_size: usize,
    /// Optimizer type
    #[pyo3(get, set)]
    pub optimizer: OptimizerType,
    /// Learning rate scheduler
    #[pyo3(get, set)]
    pub scheduler: SchedulerType,
}

#[pymethods]
impl PyTrainConfig {
    /// Create a new training configuration
    ///
    /// # Arguments
    ///
    /// * `epochs` - Number of training epochs
    /// * `learning_rate` - Initial learning rate
    /// * `batch_size` - Batch size (default: 32)
    /// * `optimizer` - Optimizer type (default: Adam)
    /// * `scheduler` - Scheduler type (default: Constant)
    ///
    /// # Returns
    ///
    /// A new `TrainConfig` instance
    ///
    /// # Raises
    ///
    /// * `ValueError` - If learning_rate <= 0 or batch_size == 0
    #[new]
    #[pyo3(signature = (epochs, learning_rate, batch_size=32, optimizer=OptimizerType::Adam, scheduler=SchedulerType::Constant))]
    fn new(
        epochs: usize,
        learning_rate: f32,
        batch_size: usize,
        optimizer: OptimizerType,
        scheduler: SchedulerType,
    ) -> PyResult<Self> {
        if learning_rate <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive"));
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("Batch size must be greater than 0"));
        }

        Ok(PyTrainConfig {
            epochs,
            learning_rate,
            batch_size,
            optimizer,
            scheduler,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTrainConfig(epochs={}, learning_rate={}, batch_size={}, optimizer={:?}, scheduler={:?})",
            self.epochs, self.learning_rate, self.batch_size, self.optimizer, self.scheduler
        )
    }
}

/// Training metrics from model training
///
/// Contains training history and final statistics.
///
/// # Attributes
///
/// * `loss_history` - Loss values per epoch
/// * `accuracy_history` - Accuracy values per epoch
/// * `final_loss` - Final loss value
/// * `epochs_trained` - Number of epochs completed
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Loss history per epoch
    #[pyo3(get)]
    pub loss_history: Vec<f32>,
    /// Accuracy history per epoch
    #[pyo3(get)]
    pub accuracy_history: Vec<f32>,
    /// Final loss value
    #[pyo3(get)]
    pub final_loss: f32,
    /// Number of epochs trained
    #[pyo3(get)]
    pub epochs_trained: usize,
}

#[pymethods]
impl TrainingMetrics {
    fn __repr__(&self) -> String {
        format!(
            "TrainingMetrics(epochs_trained={}, final_loss={:.4}, avg_accuracy={:.4})",
            self.epochs_trained,
            self.final_loss,
            self.accuracy_history.iter().sum::<f32>() / self.accuracy_history.len() as f32
        )
    }

    /// Get summary statistics as a dictionary
    fn summary(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("epochs_trained", self.epochs_trained)?;
        dict.set_item("final_loss", self.final_loss)?;
        dict.set_item("min_loss", self.loss_history.iter().cloned().fold(f32::INFINITY, f32::min))?;
        dict.set_item("max_accuracy", self.accuracy_history.iter().cloned().fold(f32::NEG_INFINITY, f32::max))?;
        dict.set_item("final_accuracy", self.accuracy_history.last().unwrap_or(&0.0))?;
        Ok(dict.into())
    }
}

/// Simple tensor wrapper for neural network operations
///
/// Represents multi-dimensional arrays for GNN computations.
///
/// # Attributes
///
/// * `data` - 2D array of floats
/// * `shape` - Tensor dimensions (rows, columns)
#[pyclass]
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Tensor data as 2D array
    #[pyo3(get)]
    pub data: Vec<Vec<f32>>,
    /// Tensor shape (rows, cols)
    #[pyo3(get)]
    pub shape: (usize, usize),
}

#[pymethods]
impl Tensor {
    /// Create a new tensor
    ///
    /// # Arguments
    ///
    /// * `data` - 2D list of floats
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance
    ///
    /// # Raises
    ///
    /// * `ValueError` - If data is not a valid 2D array
    #[new]
    fn new(data: Vec<Vec<f32>>) -> PyResult<Self> {
        if data.is_empty() {
            return Err(PyValueError::new_err("Tensor data cannot be empty"));
        }

        let rows = data.len();
        let cols = data[0].len();

        // Validate all rows have same length
        for row in &data {
            if row.len() != cols {
                return Err(PyValueError::new_err(
                    "All rows must have the same length"
                ));
            }
        }

        Ok(Tensor {
            data,
            shape: (rows, cols),
        })
    }

    /// Convert tensor to nested list (returns data)
    fn to_list(&self) -> Vec<Vec<f32>> {
        self.data.clone()
    }

    /// Get tensor element at position
    fn get(&self, row: usize, col: usize) -> PyResult<f32> {
        self.data
            .get(row)
            .and_then(|r| r.get(col))
            .copied()
            .ok_or_else(|| PyValueError::new_err("Index out of bounds"))
    }

    /// Flatten tensor to 1D vector
    fn flatten(&self) -> Vec<f32> {
        self.data.iter().flatten().copied().collect()
    }

    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?})", self.shape)
    }

    fn __len__(&self) -> usize {
        self.shape.0
    }
}

/// Basic Graph Neural Network Layer (Custom Implementation)
///
/// This is a **simplified custom implementation** for educational purposes.
/// For production use with advanced features (attention, GRU, layer norm),
/// use `PyRuvectorLayer` which wraps ruvector-gnn's `RuvectorLayer`.
///
/// Represents a single GNN layer with learnable parameters using simple
/// neighbor aggregation and linear transformation.
///
/// # Example
///
/// ```python
/// layer = BasicGNNLayer(input_dim=64, output_dim=128, activation="relu")
/// output = layer.forward(node_features, adjacency_matrix)
/// params = layer.parameters()
/// ```
#[pyclass]
#[derive(Clone)]
pub struct BasicGNNLayer {
    input_dim: usize,
    output_dim: usize,
    activation: String,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    // Gradients for backpropagation
    weight_gradients: Vec<Vec<f32>>,
    bias_gradients: Vec<f32>,
    // Cache for backpropagation
    last_input: Vec<Vec<f32>>,
    last_aggregated: Vec<Vec<f32>>,
    last_output: Vec<Vec<f32>>,
}

#[pymethods]
impl BasicGNNLayer {
    /// Create a new GNN layer
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input feature dimension
    /// * `output_dim` - Output feature dimension
    /// * `activation` - Activation function name
    ///
    /// # Returns
    ///
    /// A new `GNNLayer` instance
    #[new]
    fn new(input_dim: usize, output_dim: usize, activation: String) -> PyResult<Self> {
        // Initialize weights with Xavier/Glorot initialization
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(-scale, scale);

        let weights = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| dist.sample(&mut rng))
                    .collect()
            })
            .collect();

        let bias = vec![0.0; output_dim];
        let weight_gradients = vec![vec![0.0; input_dim]; output_dim];
        let bias_gradients = vec![0.0; output_dim];

        Ok(BasicGNNLayer {
            input_dim,
            output_dim,
            activation,
            weights,
            bias,
            weight_gradients,
            bias_gradients,
            last_input: Vec::new(),
            last_aggregated: Vec::new(),
            last_output: Vec::new(),
        })
    }

    /// Forward pass through the layer
    ///
    /// # Arguments
    ///
    /// * `node_features` - Node feature matrix (N x input_dim)
    /// * `adjacency` - Adjacency list representation (list of neighbor lists)
    ///
    /// # Returns
    ///
    /// Output feature matrix (N x output_dim)
    fn forward(
        &mut self,
        node_features: Vec<Vec<f32>>,
        adjacency: Vec<Vec<usize>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let num_nodes = node_features.len();
        let mut output = vec![vec![0.0; self.output_dim]; num_nodes];
        let mut aggregated_features = vec![vec![0.0; self.input_dim]; num_nodes];

        for (i, neighbors) in adjacency.iter().enumerate() {
            // Aggregate neighbor features
            let mut aggregated = vec![0.0; self.input_dim];

            for &neighbor_idx in neighbors {
                if neighbor_idx < num_nodes {
                    for (j, &val) in node_features[neighbor_idx].iter().enumerate() {
                        aggregated[j] += val;
                    }
                }
            }

            // Normalize by degree
            let degree = neighbors.len() as f32;
            if degree > 0.0 {
                for val in aggregated.iter_mut() {
                    *val /= degree;
                }
            }

            aggregated_features[i] = aggregated.clone();

            // Apply linear transformation
            for (out_idx, weight_row) in self.weights.iter().enumerate() {
                let mut sum = self.bias[out_idx];
                for (in_idx, &w) in weight_row.iter().enumerate() {
                    sum += w * aggregated[in_idx];
                }
                output[i][out_idx] = self.apply_activation(sum);
            }
        }

        // Cache for backpropagation
        self.last_input = node_features;
        self.last_aggregated = aggregated_features;
        self.last_output = output.clone();

        Ok(output)
    }

    /// Get layer parameters
    ///
    /// # Returns
    ///
    /// Dictionary containing weights and biases
    fn parameters(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("weights", self.weights.clone())?;
        dict.set_item("bias", self.bias.clone())?;
        dict.set_item("input_dim", self.input_dim)?;
        dict.set_item("output_dim", self.output_dim)?;
        dict.set_item("activation", &self.activation)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "GNNLayer(input_dim={}, output_dim={}, activation='{}')",
            self.input_dim, self.output_dim, self.activation
        )
    }
}

impl BasicGNNLayer {
    fn apply_activation(&self, x: f32) -> f32 {
        match self.activation.as_str() {
            "relu" => x.max(0.0),
            "tanh" => x.tanh(),
            "sigmoid" => 1.0 / (1.0 + (-x).exp()),
            "leaky_relu" => {
                if x > 0.0 { x } else { 0.01 * x }
            }
            "gelu" => {
                // GELU approximation
                0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            }
            "swish" => x / (1.0 + (-x).exp()),
            _ => x, // identity
        }
    }

    fn activation_derivative(&self, x: f32) -> f32 {
        match self.activation.as_str() {
            "relu" => if x > 0.0 { 1.0 } else { 0.0 },
            "tanh" => {
                let t = x.tanh();
                1.0 - t * t
            }
            "sigmoid" => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            "leaky_relu" => if x > 0.0 { 1.0 } else { 0.01 },
            "gelu" => {
                // GELU derivative approximation
                let cdf = 0.5 * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * x).tanh());
                let pdf = ((-0.5 * x * x).exp()) / (2.0 * std::f32::consts::PI).sqrt();
                cdf + x * pdf
            }
            "swish" => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid + x * sigmoid * (1.0 - sigmoid)
            }
            _ => 1.0, // identity
        }
    }

    /// Backward pass - compute gradients
    fn backward(&mut self, output_gradients: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let num_nodes = self.last_output.len();
        let mut input_gradients = vec![vec![0.0; self.input_dim]; num_nodes];

        // Reset gradients
        self.weight_gradients = vec![vec![0.0; self.input_dim]; self.output_dim];
        self.bias_gradients = vec![0.0; self.output_dim];

        for i in 0..num_nodes {
            for out_idx in 0..self.output_dim {
                // Gradient through activation
                let pre_activation = {
                    let mut sum = self.bias[out_idx];
                    for in_idx in 0..self.input_dim {
                        sum += self.weights[out_idx][in_idx] * self.last_aggregated[i][in_idx];
                    }
                    sum
                };

                let activation_grad = self.activation_derivative(pre_activation);
                let grad = output_gradients[i][out_idx] * activation_grad;

                // Accumulate bias gradient
                self.bias_gradients[out_idx] += grad;

                // Accumulate weight gradients and compute input gradients
                for in_idx in 0..self.input_dim {
                    self.weight_gradients[out_idx][in_idx] += grad * self.last_aggregated[i][in_idx];
                    input_gradients[i][in_idx] += grad * self.weights[out_idx][in_idx];
                }
            }
        }

        input_gradients
    }
}

/// Ruvector Layer Wrapper
///
/// Python wrapper for ruvector-gnn's RuvectorLayer, providing HNSW-based GNN
/// layer with attention mechanisms, GRU updates, and layer normalization.
///
/// This layer operates on HNSW graph topology with:
/// - Multi-head attention for neighbor aggregation
/// - GRU cells for state updates
/// - Layer normalization
/// - Dropout regularization
///
/// # Example
///
/// ```python
/// layer = RuvectorLayer(input_dim=64, hidden_dim=128, heads=4, dropout=0.1)
///
/// # Single node's embedding
/// node_embedding = [0.1, 0.2, ..., 0.64]  # length = input_dim
///
/// # Neighbor embeddings
/// neighbor_embeddings = [
///     [0.2, 0.3, ..., 0.65],
///     [0.4, 0.1, ..., 0.72]
/// ]
///
/// # Edge weights (e.g., distances or attention scores)
/// edge_weights = [0.5, 0.8]
///
/// # Forward pass
/// output = layer.forward(node_embedding, neighbor_embeddings, edge_weights)
/// # output.len() == hidden_dim
/// ```
#[pyclass(name = "RuvectorLayer")]
#[derive(Clone)]
pub struct RuvectorLayerWrapper {
    inner: RuvectorLayerCrate,
    input_dim: usize,
    hidden_dim: usize,
    heads: usize,
    dropout: f32,
}

#[pymethods]
impl RuvectorLayerWrapper {
    /// Create a new Ruvector GNN layer
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Dimension of input node embeddings
    /// * `hidden_dim` - Dimension of hidden representations (must be divisible by heads)
    /// * `heads` - Number of attention heads
    /// * `dropout` - Dropout rate for regularization (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// A new `RuvectorLayer` instance
    ///
    /// # Raises
    ///
    /// * `ValueError` - If dropout not in [0.0, 1.0] or hidden_dim not divisible by heads
    ///
    /// # Example
    ///
    /// ```python
    /// # Create layer with 4 attention heads
    /// layer = RuvectorLayer(
    ///     input_dim=64,
    ///     hidden_dim=128,
    ///     heads=4,
    ///     dropout=0.1
    /// )
    /// ```
    #[new]
    fn new(input_dim: usize, hidden_dim: usize, heads: usize, dropout: f32) -> PyResult<Self> {
        if dropout < 0.0 || dropout > 1.0 {
            return Err(PyValueError::new_err(
                "Dropout must be between 0.0 and 1.0"
            ));
        }

        if hidden_dim % heads != 0 {
            return Err(PyValueError::new_err(
                format!("hidden_dim ({}) must be divisible by heads ({})", hidden_dim, heads)
            ));
        }

        let inner = RuvectorLayerCrate::new(input_dim, hidden_dim, heads, dropout);

        Ok(RuvectorLayerWrapper {
            inner,
            input_dim,
            hidden_dim,
            heads,
            dropout,
        })
    }

    /// Forward pass through the Ruvector GNN layer
    ///
    /// Performs message passing, attention-based aggregation, GRU update,
    /// and layer normalization on the HNSW graph structure.
    ///
    /// # Arguments
    ///
    /// * `node_embedding` - Current node's embedding vector (length = input_dim)
    /// * `neighbor_embeddings` - List of neighbor node embeddings (each length = input_dim)
    /// * `edge_weights` - Weights of edges to neighbors (e.g., distances, must match neighbor count)
    ///
    /// # Returns
    ///
    /// Updated node embedding vector (length = hidden_dim)
    ///
    /// # Raises
    ///
    /// * `ValueError` - If input dimensions are invalid or mismatched
    ///
    /// # Example
    ///
    /// ```python
    /// layer = RuvectorLayer(input_dim=64, hidden_dim=128, heads=4, dropout=0.1)
    ///
    /// node = [0.1] * 64
    /// neighbors = [[0.2] * 64, [0.3] * 64, [0.4] * 64]
    /// weights = [0.5, 0.8, 0.3]
    ///
    /// output = layer.forward(node, neighbors, weights)
    /// assert len(output) == 128
    /// ```
    fn forward(
        &self,
        node_embedding: Vec<f32>,
        neighbor_embeddings: Vec<Vec<f32>>,
        edge_weights: Vec<f32>,
    ) -> PyResult<Vec<f32>> {
        // Validate input dimensions
        if node_embedding.len() != self.input_dim {
            return Err(PyValueError::new_err(
                format!(
                    "node_embedding length ({}) must match input_dim ({})",
                    node_embedding.len(),
                    self.input_dim
                )
            ));
        }

        // Validate neighbor embeddings
        for (i, neighbor) in neighbor_embeddings.iter().enumerate() {
            if neighbor.len() != self.input_dim {
                return Err(PyValueError::new_err(
                    format!(
                        "neighbor_embeddings[{}] length ({}) must match input_dim ({})",
                        i,
                        neighbor.len(),
                        self.input_dim
                    )
                ));
            }
        }

        // Validate edge weights
        if !neighbor_embeddings.is_empty() && edge_weights.len() != neighbor_embeddings.len() {
            return Err(PyValueError::new_err(
                format!(
                    "edge_weights length ({}) must match number of neighbors ({})",
                    edge_weights.len(),
                    neighbor_embeddings.len()
                )
            ));
        }

        // Call the underlying ruvector-gnn layer
        let output = self.inner.forward(
            &node_embedding,
            &neighbor_embeddings,
            &edge_weights,
        );

        Ok(output)
    }

    /// Get layer configuration as a dictionary
    ///
    /// # Returns
    ///
    /// Dictionary containing layer parameters
    fn config(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("input_dim", self.input_dim)?;
        dict.set_item("hidden_dim", self.hidden_dim)?;
        dict.set_item("heads", self.heads)?;
        dict.set_item("dropout", self.dropout)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "RuvectorLayer(input_dim={}, hidden_dim={}, heads={}, dropout={})",
            self.input_dim, self.hidden_dim, self.heads, self.dropout
        )
    }
}

/// Graph Neural Network Model
///
/// Complete GNN model with multiple layers, training, and inference capabilities.
/// This wraps ruvector-gnn's Optimizer and LearningRateScheduler for proper training.
///
/// # Example
///
/// ```python
/// config = GNNConfig(hidden_dims=[64, 128, 64], num_layers=3)
/// model = GNNModel(config)
/// model.add_layer(GNNLayer(128, 64, "relu"))
/// predictions = model.predict(graph_data)
/// ```
#[pyclass]
pub struct GNNModel {
    config: GNNConfig,
    layers: Vec<BasicGNNLayer>,
}

#[pymethods]
impl GNNModel {
    /// Create a new GNN model
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    ///
    /// # Returns
    ///
    /// A new `GNNModel` instance
    #[new]
    fn new(config: GNNConfig) -> PyResult<Self> {
        Ok(GNNModel {
            config,
            layers: Vec::new(),
        })
    }

    /// Add a layer to the model
    ///
    /// # Arguments
    ///
    /// * `layer` - GNN layer to add
    fn add_layer(&mut self, layer: BasicGNNLayer) {
        self.layers.push(layer);
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `node_features` - Node feature matrix
    /// * `adjacency` - Graph adjacency structure
    ///
    /// # Returns
    ///
    /// Output tensor
    fn forward(
        &mut self,
        node_features: Vec<Vec<f32>>,
        adjacency: Vec<Vec<usize>>,
    ) -> PyResult<Tensor> {
        let mut current_features = node_features;

        for layer in &mut self.layers {
            current_features = layer.forward(current_features, adjacency.clone())?;
        }

        Tensor::new(current_features)
    }

    /// Train the model using ruvector-gnn's Optimizer and LearningRateScheduler
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training data (list of (features, adjacency, labels) tuples)
    /// * `config` - Training configuration
    ///
    /// # Returns
    ///
    /// Training metrics
    fn train(
        &mut self,
        train_data: Vec<(Vec<Vec<f32>>, Vec<Vec<usize>>, Vec<f32>)>,
        config: PyTrainConfig,
    ) -> PyResult<TrainingMetrics> {
        let mut loss_history = Vec::new();
        let mut accuracy_history = Vec::new();

        // Create optimizer using ruvector-gnn
        let optimizer_type = config.optimizer.to_ruvector(config.learning_rate);
        let mut optimizers: Vec<RuvectorOptimizer> = self.layers.iter()
            .map(|_| RuvectorOptimizer::new(optimizer_type.clone()))
            .collect();

        // Create learning rate scheduler using ruvector-gnn
        let scheduler_type = config.scheduler.to_ruvector(config.epochs);
        let mut scheduler = LearningRateScheduler::new(scheduler_type, config.learning_rate);

        for _epoch in 0..config.epochs {
            let mut epoch_loss = 0.0;
            let mut correct = 0;
            let total = train_data.len();

            // Get current learning rate from scheduler
            let current_lr = scheduler.get_lr();

            for (features, adjacency, labels) in &train_data {
                // Forward pass
                let output = self.forward(features.clone(), adjacency.clone())?;

                // Compute loss (MSE) and gradients
                let predictions = output.flatten();
                let num_nodes = output.shape.0;
                let output_dim = output.shape.1;

                let mut sample_loss = 0.0;
                let mut output_gradients = vec![vec![0.0; output_dim]; num_nodes];

                // Compute MSE loss and gradients
                for i in 0..num_nodes.min(labels.len()) {
                    for j in 0..output_dim {
                        let idx = i * output_dim + j;
                        if idx < predictions.len() && idx < labels.len() {
                            let pred = predictions[idx];
                            let label = labels[idx];
                            sample_loss += (pred - label).powi(2);
                            // Gradient of MSE: 2 * (pred - label)
                            output_gradients[i][j] = 2.0 * (pred - label) / labels.len() as f32;
                        }
                    }
                }
                sample_loss /= labels.len() as f32;
                epoch_loss += sample_loss;

                // Backward pass through all layers
                let mut gradients = output_gradients;
                for layer in self.layers.iter_mut().rev() {
                    gradients = layer.backward(gradients);
                }

                // Update weights using ruvector-gnn's Optimizer
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    // Convert layer weights to Array2
                    let mut params = Array2::from_shape_vec(
                        (layer.output_dim, layer.input_dim),
                        layer.weights.iter().flatten().copied().collect()
                    ).map_err(|e| PyRuntimeError::new_err(format!("Failed to create weight array: {}", e)))?;

                    let grads = Array2::from_shape_vec(
                        (layer.output_dim, layer.input_dim),
                        layer.weight_gradients.iter().flatten().copied().collect()
                    ).map_err(|e| PyRuntimeError::new_err(format!("Failed to create gradient array: {}", e)))?;

                    // Use ruvector-gnn's optimizer.step()
                    optimizers[layer_idx].step(&mut params, &grads)
                        .map_err(|e| PyRuntimeError::new_err(format!("Optimizer step failed: {:?}", e)))?;

                    // Update layer weights from Array2
                    for out_idx in 0..layer.output_dim {
                        for in_idx in 0..layer.input_dim {
                            layer.weights[out_idx][in_idx] = params[[out_idx, in_idx]];
                        }
                    }

                    // Update bias using simple SGD (ruvector-gnn's sgd_step)
                    ruvector_sgd_step(&mut layer.bias, &layer.bias_gradients, current_lr);
                }

                // Compute accuracy (simplified)
                if !predictions.is_empty() && !labels.is_empty() {
                    let pred_label = predictions[0].round();
                    if (pred_label - labels[0]).abs() < 0.5 {
                        correct += 1;
                    }
                }
            }

            epoch_loss /= total as f32;
            let accuracy = correct as f32 / total as f32;

            loss_history.push(epoch_loss);
            accuracy_history.push(accuracy);

            // Step scheduler using ruvector-gnn's scheduler.step()
            scheduler.step();
        }

        Ok(TrainingMetrics {
            loss_history: loss_history.clone(),
            accuracy_history: accuracy_history.clone(),
            final_loss: *loss_history.last().unwrap_or(&0.0),
            epochs_trained: config.epochs,
        })
    }

    /// Make predictions on new data
    ///
    /// # Arguments
    ///
    /// * `node_features` - Node features
    /// * `adjacency` - Graph structure
    ///
    /// # Returns
    ///
    /// Prediction vector
    fn predict(
        &mut self,
        node_features: Vec<Vec<f32>>,
        adjacency: Vec<Vec<usize>>,
    ) -> PyResult<Vec<f32>> {
        let output = self.forward(node_features, adjacency)?;
        Ok(output.flatten())
    }

    /// Save model to file
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save to
    fn save(&self, path: String) -> PyResult<()> {
        #[derive(Serialize)]
        struct ModelData {
            config: GNNConfig,
            layers: Vec<LayerData>,
        }

        #[derive(Serialize)]
        struct LayerData {
            input_dim: usize,
            output_dim: usize,
            activation: String,
            weights: Vec<Vec<f32>>,
            bias: Vec<f32>,
        }

        let layers_data: Vec<LayerData> = self.layers.iter().map(|layer| LayerData {
            input_dim: layer.input_dim,
            output_dim: layer.output_dim,
            activation: layer.activation.clone(),
            weights: layer.weights.clone(),
            bias: layer.bias.clone(),
        }).collect();

        let model_data = ModelData {
            config: self.config.clone(),
            layers: layers_data,
        };

        let serialized = serde_json::to_string_pretty(&model_data)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {}", e)))?;

        std::fs::write(&path, serialized)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to save model: {}", e)))?;

        Ok(())
    }

    /// Load model from file
    ///
    /// # Arguments
    ///
    /// * `path` - File path to load from
    ///
    /// # Returns
    ///
    /// Loaded GNN model
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        #[derive(Deserialize)]
        struct ModelData {
            config: GNNConfig,
            layers: Vec<LayerData>,
        }

        #[derive(Deserialize)]
        struct LayerData {
            input_dim: usize,
            output_dim: usize,
            activation: String,
            weights: Vec<Vec<f32>>,
            bias: Vec<f32>,
        }

        let content = std::fs::read_to_string(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

        let model_data: ModelData = serde_json::from_str(&content)
            .map_err(|e| PyRuntimeError::new_err(format!("Deserialization error: {}", e)))?;

        let mut model = GNNModel::new(model_data.config)?;

        // Restore layers with weights
        for layer_data in model_data.layers {
            let weight_gradients = vec![vec![0.0; layer_data.input_dim]; layer_data.output_dim];
            let bias_gradients = vec![0.0; layer_data.output_dim];

            let layer = BasicGNNLayer {
                input_dim: layer_data.input_dim,
                output_dim: layer_data.output_dim,
                activation: layer_data.activation,
                weights: layer_data.weights,
                bias: layer_data.bias,
                weight_gradients,
                bias_gradients,
                last_input: Vec::new(),
                last_aggregated: Vec::new(),
                last_output: Vec::new(),
            };
            model.add_layer(layer);
        }

        Ok(model)
    }

    fn __repr__(&self) -> String {
        format!(
            "GNNModel(num_layers={}, config={:?})",
            self.layers.len(),
            self.config
        )
    }
}

/// Experience replay buffer for continual learning
///
/// Wraps ruvector-gnn's ReplayBuffer for experience replay.
///
/// # Example
///
/// ```python
/// buffer = ReplayBuffer(capacity=1000)
/// buffer.add({"state": [1.0, 2.0], "reward": 0.5})
/// samples = buffer.sample(batch_size=32)
/// ```
#[pyclass]
pub struct ReplayBuffer {
    // We maintain our own buffer since ruvector-gnn's ReplayBuffer uses a different structure
    capacity: usize,
    buffer: Vec<HashMap<String, Vec<f32>>>,
}

#[pymethods]
impl ReplayBuffer {
    /// Create a new replay buffer
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum buffer capacity
    ///
    /// # Returns
    ///
    /// A new `ReplayBuffer` instance
    #[new]
    fn new(capacity: usize) -> Self {
        ReplayBuffer {
            capacity,
            buffer: Vec::new(),
        }
    }

    /// Add an entry to the buffer
    ///
    /// # Arguments
    ///
    /// * `entry` - Dictionary containing experience data
    ///
    /// # Note
    ///
    /// When buffer is full, oldest entries are removed (FIFO)
    fn add(&mut self, _py: Python, entry: &PyDict) -> PyResult<()> {
        let mut entry_map = HashMap::new();

        for (key, value) in entry.iter() {
            let key_str: String = key.extract()?;
            let value_list: Vec<f32> = value.extract()?;
            entry_map.insert(key_str, value_list);
        }

        if self.buffer.len() >= self.capacity {
            self.buffer.remove(0);
        }

        self.buffer.push(entry_map);
        Ok(())
    }

    /// Sample random entries from the buffer
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of samples to retrieve
    ///
    /// # Returns
    ///
    /// List of sampled entries
    fn sample(&self, py: Python, batch_size: usize) -> PyResult<PyObject> {
        use rand::seq::SliceRandom;

        let sample_size = batch_size.min(self.buffer.len());
        let mut samples: Vec<PyObject> = Vec::new();

        // Simple random sampling (without replacement)
        let mut indices: Vec<usize> = (0..self.buffer.len()).collect();

        // Shuffle indices using Fisher-Yates algorithm
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        for &idx in indices.iter().take(sample_size) {
            let dict = PyDict::new(py);
            for (key, value) in &self.buffer[idx] {
                dict.set_item(key, value.clone())?;
            }
            samples.push(dict.into());
        }

        Ok(PyList::new(py, samples).into())
    }

    fn __len__(&self) -> usize {
        self.buffer.len()
    }

    fn __repr__(&self) -> String {
        format!("ReplayBuffer(capacity={}, size={})", self.capacity, self.buffer.len())
    }

    /// Clear all entries from the buffer
    fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Check if buffer is full
    fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }
}

/// Compute cosine similarity between two vectors using ruvector-gnn
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// Cosine similarity in range [-1, 1]
///
/// # Raises
///
/// * `ValueError` - If vectors have different lengths or zero magnitude
#[pyfunction]
pub fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("Vectors must have the same length"));
    }

    // Use ruvector-gnn's cosine_similarity
    let similarity = ruvector_cosine_similarity(&a, &b);
    Ok(similarity)
}

/// Compute InfoNCE contrastive loss using ruvector-gnn
///
/// InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.
///
/// # Arguments
///
/// * `embeddings` - Tensor of embeddings (batch_size x embedding_dim)
/// * `temperature` - Temperature scaling parameter
///
/// # Returns
///
/// InfoNCE loss value
///
/// # Raises
///
/// * `ValueError` - If temperature <= 0 or embeddings are invalid
#[pyfunction]
pub fn info_nce_loss(embeddings: Tensor, temperature: f32) -> PyResult<f32> {
    if temperature <= 0.0 {
        return Err(PyValueError::new_err("Temperature must be positive"));
    }

    let batch_size = embeddings.shape.0;
    if batch_size < 2 {
        return Err(PyValueError::new_err("Need at least 2 samples for contrastive loss"));
    }

    let mut total_loss = 0.0;

    for i in 0..batch_size {
        let anchor = &embeddings.data[i];

        // Positive pair (next sample, circular)
        let positive_idx = (i + 1) % batch_size;
        let positive = &embeddings.data[positive_idx];

        // Collect negatives
        let mut negatives: Vec<&[f32]> = Vec::new();
        for j in 0..batch_size {
            if j != i && j != positive_idx {
                negatives.push(&embeddings.data[j]);
            }
        }

        // Use ruvector-gnn's info_nce_loss
        let loss = ruvector_info_nce_loss(anchor, &[positive.as_slice()], &negatives, temperature);
        total_loss += loss;
    }

    Ok(total_loss / batch_size as f32)
}
