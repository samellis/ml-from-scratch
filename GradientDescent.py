def GradientDescent(X_train, y_train, learning_rate=0.001, epochs=5000, print_at=1000):
    """
    Implements simple gradient descent.
    Initialise weights and biases to 0
    Iterate through epochs:
        Zero loss and grads
        Iterate through samples:
            Calculate predictions, error, loss
            Calculate weights and bias grads
        Update weights and biases based on avg grads and learning_rate
        Print output at defined epochs
    Note: loss function is L2
    """

    n_samples = len(X_train)
    n_features = len(X_train[0])

    # Initialise weights and biases and loss
    weights = [0.0 for i in range(n_features)]
    bias = 0.0

    for epoch in range(epochs+1):
        total_loss = 0.0
        weight_gradients = [0.0 for i in range(n_features)]
        bias_gradient = 0.0
        predictions = []

        for sample_idx in range(n_samples):
            # Forward pass: Calculate predictions using prev weights and biases

            y_pred = bias
            for feature_idx in range(n_features):
                y_pred += X_train[sample_idx][feature_idx] * weights[feature_idx]
            predictions.append(y_pred)

            # Calculate error and loss for sample
            error = y_pred - y_train[sample_idx]
            total_loss += error**2

            # Backward pass: calculate and accumulate gradients/bias
            for feature_idx in range(n_features):
                weight_gradients[feature_idx] += error * 2 * X_train[sample_idx][feature_idx]
            bias_gradient += 2 * error

        # Set new weights and biases (weight - weight_grad * lr)
        for feature_idx in range(n_features):
            weights[feature_idx] = weights[feature_idx] - (
                weight_gradients[feature_idx]/n_samples) * learning_rate

        bias -= (bias_gradient/n_samples)*learning_rate

        # Calculate loss and display stats at defined epochs
        if epoch % print_at == 0:
            print(f"Epoch {epoch}")
            print(f"Loss: {total_loss/n_samples}")
            print(f"Weights: {weights}")
            print(f"Bias: {bias}")

    return weights, bias, predictions


if __name__ == "__main__":
    X = [[2, 4], [10, 11], [12, 11]]
    y = [4.4, -89.9, -118.9]

    GradientDescent(X, y, learning_rate=0.001, epochs=10000)
