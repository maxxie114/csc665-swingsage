import numpy as np
import torch
import torch.nn as nn

class VectorCombiner(nn.Module):
    def __init__(self):
        super(VectorCombiner, self).__init__()
        # Learnable weights alpha and beta
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, v1, v2):
        """
        Combine two vectors using weighted sum: alpha * v1 + beta * v2

        Args:
            v1: First vector of shape (3,) representing [buy, sell, hold]
            v2: Second vector of shape (3,) representing [buy, sell, hold]

        Returns:
            max_label: String indicating the maximum label ('buy', 'sell', or 'hold')
            combined_vector: The combined weighted vector
        """
        # Convert inputs to tensors if they aren't already
        if not isinstance(v1, torch.Tensor):
            v1 = torch.tensor(v1, dtype=torch.float32)
        if not isinstance(v2, torch.Tensor):
            v2 = torch.tensor(v2, dtype=torch.float32)

        # Ensure vectors are 1D
        v1 = v1.view(-1)
        v2 = v2.view(-1)

        # Combine vectors with learned weights
        combined_vector = self.alpha * v1 + self.beta * v2

        # Get the index of maximum value
        max_idx = torch.argmax(combined_vector).item()

        # Map index to label
        labels = ['buy', 'sell', 'hold']
        max_label = labels[max_idx]

        return max_label, combined_vector

def combine_vectors(v1, v2, alpha=0.5, beta=0.5):
    """
    Simple function to combine vectors without PyTorch model

    Args:
        v1: First vector [buy, sell, hold]
        v2: Second vector [buy, sell, hold]
        alpha: Weight for v1
        beta: Weight for v2

    Returns:
        max_label: String indicating the maximum label
        combined_vector: The combined weighted vector
    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Combine vectors
    combined_vector = alpha * v1 + beta * v2

    # Get maximum label
    labels = ['buy', 'sell', 'hold']
    max_idx = np.argmax(combined_vector)
    max_label = labels[max_idx]

    return max_label, combined_vector

# Example usage and testing
if __name__ == "__main__":
    # Test vectors
    v1 = [0.8, 0.1, 0.1]  # Strong buy signal
    v2 = [0.2, 0.7, 0.1]  # Strong sell signal

    print("Simple combination:")
    label, combined = combine_vectors(v1, v2)
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Combined: {combined}")
    print(f"Max label: {label}")
    print()

    print("Learned combination:")
    # Initialize model
    model = VectorCombiner()

    # Example forward pass
    label, combined = model(v1, v2)
    print(f"Alpha: {model.alpha.item():.3f}")
    print(f"Beta: {model.beta.item():.3f}")
    print(f"Combined: {combined.detach().numpy()}")
    print(f"Max label: {label}")
    print()

    # Example with different weights
    print("Custom weights (alpha=0.8, beta=0.2):")
    label, combined = combine_vectors(v1, v2, alpha=0.8, beta=0.2)
    print(f"Combined: {combined}")
    print(f"Max label: {label}")