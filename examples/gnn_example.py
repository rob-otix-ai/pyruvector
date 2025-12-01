"""
Example usage of pyruvector Graph Neural Network (GNN) module

This demonstrates how to:
1. Create and configure GNN models
2. Train GNN models on graph data
3. Make predictions
4. Use replay buffers for continual learning
5. Compute contrastive losses
"""

from pyruvector import (
    GNNModel,
    GNNLayer,
    GNNConfig,
    TrainConfig,
    OptimizerType,
    SchedulerType,
    Tensor,
    ReplayBuffer,
    cosine_similarity,
    info_nce_loss
)


def basic_gnn_example():
    """Basic GNN model creation and forward pass"""
    print("=" * 60)
    print("Basic GNN Example")
    print("=" * 60)

    # Create GNN configuration
    config = GNNConfig(
        hidden_dims=[64, 128, 64],
        num_layers=3,
        dropout=0.1,
        activation="relu"
    )
    print(f"Config: {config}")

    # Initialize model
    model = GNNModel(config)
    print(f"Model: {model}")

    # Add custom layers
    layer1 = GNNLayer(input_dim=32, output_dim=64, activation="relu")
    model.add_layer(layer1)
    print(f"Added layer: {layer1}")

    # Create sample graph data
    # Node features: 5 nodes with 32 features each
    node_features = [[float(i * j % 10) for j in range(32)] for i in range(5)]

    # Adjacency list: each node connected to its neighbors
    adjacency = [
        [1, 2],      # Node 0 connected to nodes 1, 2
        [0, 2, 3],   # Node 1 connected to nodes 0, 2, 3
        [0, 1, 4],   # Node 2 connected to nodes 0, 1, 4
        [1, 4],      # Node 3 connected to nodes 1, 4
        [2, 3]       # Node 4 connected to nodes 2, 3
    ]

    # Forward pass
    output = model.forward(node_features, adjacency)
    print(f"Output tensor shape: {output.shape}")
    print(f"Output tensor: {output}")


def training_example():
    """Train a GNN model on synthetic data"""
    print("\n" + "=" * 60)
    print("Training Example")
    print("=" * 60)

    # Create model configuration
    config = GNNConfig(
        hidden_dims=[16, 32, 16],
        num_layers=2,
        dropout=0.2,
        activation="relu"
    )

    model = GNNModel(config)

    # Add layers
    model.add_layer(GNNLayer(8, 16, "relu"))
    model.add_layer(GNNLayer(16, 1, "sigmoid"))

    # Create synthetic training data
    # Each sample: (node_features, adjacency, labels)
    training_data = []

    for i in range(10):
        # 3 nodes with 8 features each
        features = [[float(j + i) for j in range(8)] for _ in range(3)]
        adjacency = [[1, 2], [0, 2], [0, 1]]
        labels = [float(i % 2)]  # Binary classification
        training_data.append((features, adjacency, labels))

    # Configure training
    train_config = TrainConfig(
        epochs=50,
        learning_rate=0.001,
        batch_size=5,
        optimizer=OptimizerType.Adam,
        scheduler=SchedulerType.CosineAnnealing
    )
    print(f"Training config: {train_config}")

    # Train the model
    print("Training...")
    metrics = model.train(training_data, train_config)
    print(f"Training metrics: {metrics}")
    print(f"Final loss: {metrics.final_loss:.4f}")
    print(f"Epochs trained: {metrics.epochs_trained}")

    # Get summary
    summary = metrics.summary()
    print(f"Training summary: {summary}")


def prediction_example():
    """Make predictions with a trained model"""
    print("\n" + "=" * 60)
    print("Prediction Example")
    print("=" * 60)

    config = GNNConfig(
        hidden_dims=[16],
        num_layers=1,
        dropout=0.0,
        activation="relu"
    )

    model = GNNModel(config)
    model.add_layer(GNNLayer(8, 16, "relu"))
    model.add_layer(GNNLayer(16, 2, "sigmoid"))

    # Test data
    test_features = [[float(i) for i in range(8)] for _ in range(4)]
    test_adjacency = [[1, 2], [0, 3], [0, 3], [1, 2]]

    # Make predictions
    predictions = model.predict(test_features, test_adjacency)
    print(f"Predictions: {predictions}")


def replay_buffer_example():
    """Use replay buffer for continual learning"""
    print("\n" + "=" * 60)
    print("Replay Buffer Example")
    print("=" * 60)

    # Create replay buffer
    buffer = ReplayBuffer(capacity=100)
    print(f"Buffer: {buffer}")

    # Add experiences
    for i in range(10):
        experience = {
            "state": [float(i * j) for j in range(5)],
            "action": [float(i % 3)],
            "reward": [float(i * 0.1)],
            "next_state": [float((i + 1) * j) for j in range(5)]
        }
        buffer.add(experience)

    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer full: {buffer.is_full()}")

    # Sample batch
    batch = buffer.sample(batch_size=5)
    print(f"Sampled {len(batch)} experiences")

    # Clear buffer
    buffer.clear()
    print(f"After clear, buffer size: {len(buffer)}")


def tensor_operations_example():
    """Demonstrate tensor operations"""
    print("\n" + "=" * 60)
    print("Tensor Operations Example")
    print("=" * 60)

    # Create tensor
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    tensor = Tensor(data)
    print(f"Tensor: {tensor}")
    print(f"Shape: {tensor.shape}")
    print(f"Length: {len(tensor)}")

    # Access element
    value = tensor.get(0, 1)
    print(f"Element at (0, 1): {value}")

    # Convert to list
    as_list = tensor.to_list()
    print(f"As list: {as_list}")

    # Flatten
    flat = tensor.flatten()
    print(f"Flattened: {flat}")


def similarity_example():
    """Compute vector similarities"""
    print("\n" + "=" * 60)
    print("Similarity Example")
    print("=" * 60)

    # Two similar vectors
    vec1 = [1.0, 2.0, 3.0, 4.0]
    vec2 = [1.1, 2.1, 2.9, 4.0]

    similarity = cosine_similarity(vec1, vec2)
    print(f"Vector 1: {vec1}")
    print(f"Vector 2: {vec2}")
    print(f"Cosine similarity: {similarity:.4f}")

    # Orthogonal vectors
    vec3 = [1.0, 0.0, 0.0]
    vec4 = [0.0, 1.0, 0.0]

    similarity2 = cosine_similarity(vec3, vec4)
    print(f"\nVector 3: {vec3}")
    print(f"Vector 4: {vec4}")
    print(f"Cosine similarity (orthogonal): {similarity2:.4f}")


def contrastive_loss_example():
    """Compute InfoNCE contrastive loss"""
    print("\n" + "=" * 60)
    print("Contrastive Loss Example")
    print("=" * 60)

    # Create embeddings (batch of 4 samples, 8 dimensions each)
    embeddings_data = [
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.9, 0.1, 1.1, -0.1, 0.9, 0.1, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.1, 0.9, -0.1, 1.1, 0.1, 0.9, 0.0, 1.0]
    ]

    embeddings = Tensor(embeddings_data)
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute InfoNCE loss with different temperatures
    for temp in [0.1, 0.5, 1.0]:
        loss = info_nce_loss(embeddings, temperature=temp)
        print(f"InfoNCE loss (temp={temp}): {loss:.4f}")


def save_load_example():
    """Save and load GNN models"""
    print("\n" + "=" * 60)
    print("Save/Load Example")
    print("=" * 60)

    # Create and configure model
    config = GNNConfig(
        hidden_dims=[32, 64],
        num_layers=2,
        dropout=0.15,
        activation="tanh"
    )

    model = GNNModel(config)
    print(f"Original model: {model}")

    # Save model
    save_path = "/tmp/gnn_model.json"
    model.save(save_path)
    print(f"Model saved to: {save_path}")

    # Load model
    loaded_model = GNNModel.load(save_path)
    print(f"Loaded model: {loaded_model}")


def optimizer_scheduler_example():
    """Different optimizer and scheduler configurations"""
    print("\n" + "=" * 60)
    print("Optimizer & Scheduler Example")
    print("=" * 60)

    # SGD with step decay
    config1 = TrainConfig(
        epochs=100,
        learning_rate=0.01,
        batch_size=16,
        optimizer=OptimizerType.SGD,
        scheduler=SchedulerType.StepDecay
    )
    print(f"Config 1 (SGD + StepDecay): {config1}")

    # Adam with cosine annealing
    config2 = TrainConfig(
        epochs=100,
        learning_rate=0.001,
        batch_size=32,
        optimizer=OptimizerType.Adam,
        scheduler=SchedulerType.CosineAnnealing
    )
    print(f"Config 2 (Adam + CosineAnnealing): {config2}")

    # AdamW with warmup cosine
    config3 = TrainConfig(
        epochs=200,
        learning_rate=0.0001,
        batch_size=64,
        optimizer=OptimizerType.AdamW,
        scheduler=SchedulerType.WarmupCosine
    )
    print(f"Config 3 (AdamW + WarmupCosine): {config3}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("pyruvector GNN Examples")
    print("=" * 60)

    try:
        basic_gnn_example()
        training_example()
        prediction_example()
        replay_buffer_example()
        tensor_operations_example()
        similarity_example()
        contrastive_loss_example()
        save_load_example()
        optimizer_scheduler_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
