"""
Double Predictor — A simple neural network that learns to predict double of any number.

This is a beginner-friendly ML model that learns the mathematical function f(x) = 2x
entirely from training data, without being explicitly programmed with the rule.

Framework: PyTorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── Model Definition ───────────────────────────────────────────────
class DoublePredictor(nn.Module):
    """A single-neuron network: learns y = weight * x + bias."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)   # 1 input → 1 output

    def forward(self, x):
        return self.linear(x)


def generate_data(n_samples=1000):
    """Generate (x, 2x) pairs as PyTorch tensors."""
    x = np.random.uniform(-100, 100, (n_samples, 1)).astype(np.float32)
    y = (2 * x).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def train(model, x_train, y_train, epochs=100, lr=0.01):
    """Train the model using MSE loss and Adam optimizer."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print_interval = max(1, epochs // 10)
    for epoch in range(1, epochs + 1):
        # Forward pass
        predictions = model(x_train)
        loss = criterion(predictions, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % print_interval == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}/{epochs}  │  Loss: {loss.item():.6f}")

    return loss.item()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Generate Data ──────────────────────────────────────────────
    print(" Generating training data (1000 samples)...")
    x_train, y_train = generate_data(1000)
    x_test, y_test = generate_data(200)

    # ── Build Model ────────────────────────────────────────────────
    model = DoublePredictor()
    print(f" Model architecture:\n{model}\n")

    # ── Train ──────────────────────────────────────────────────────
    print("🏋️  Training for 500 epochs...")
    final_loss = train(model, x_train, y_train, epochs=500, lr=0.05)

    # ── Evaluate on test set ───────────────────────────────────────
    model.eval()
    with torch.no_grad():
        test_loss = nn.MSELoss()(model(x_test), y_test).item()
    print(f"\n Test loss (MSE): {test_loss:.6f}")

    # ── Predict on sample inputs ───────────────────────────────────
    test_numbers = [1, 5, 10, 25, 50, 100, -7, 3.5]
    test_tensor = torch.tensor(test_numbers, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        predictions = model(test_tensor).squeeze().tolist()

    print("\n" + "=" * 50)
    print(f"  {'Input':>10}  │  {'Predicted':>12}  │  {'Actual (2x)':>12}")
    print("─" * 50)
    for num, pred in zip(test_numbers, predictions):
        print(f"  {num:>10.2f}  │  {pred:>12.4f}  │  {num * 2:>12.2f}")
    print("=" * 50)

    # ── Show what the model learned ────────────────────────────────
    weight = model.linear.weight.item()
    bias = model.linear.bias.item()
    print(f"\n Learned weight: {weight:.4f}  (ideal: 2.0)")
    print(f" Learned bias:   {bias:.4f}  (ideal: 0.0)")

    # ── Save the model ─────────────────────────────────────────────
    torch.save(model.state_dict(), "double_model.pth")
    print("\n Model saved to double_model.pth")


if __name__ == "__main__":
    main()
