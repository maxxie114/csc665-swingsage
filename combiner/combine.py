"""
Vector Combiner for merging LLM and LSTM predictions.
Learns optimal weights (alpha for LLM, beta for LSTM) to maximize prediction accuracy.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VectorCombiner(nn.Module):
    """
    Learnable model to combine LLM and LSTM prediction vectors.
    Uses weighted sum: alpha * LLM_vector + beta * LSTM_vector
    """
    
    def __init__(self, init_alpha=0.5, init_beta=0.5):
        super(VectorCombiner, self).__init__()
        # Learnable weights alpha (LLM weight) and beta (LSTM weight)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, v_llm, v_lstm):
        """
        Combine LLM and LSTM vectors using weighted sum.

        Args:
            v_llm: LLM vector of shape (batch, 3) or (3,) representing [buy, sell, hold]
            v_lstm: LSTM vector of shape (batch, 3) or (3,) representing [buy, sell, hold]

        Returns:
            combined_vector: The combined weighted vector (batch, 3) or (3,)
        """
        # Convert inputs to tensors if they aren't already
        if not isinstance(v_llm, torch.Tensor):
            v_llm = torch.tensor(v_llm, dtype=torch.float32)
        if not isinstance(v_lstm, torch.Tensor):
            v_lstm = torch.tensor(v_lstm, dtype=torch.float32)

        # Combine vectors with learned weights
        combined = self.alpha * v_llm + self.beta * v_lstm
        
        return combined
    
    def predict(self, v_llm, v_lstm):
        """
        Get the predicted label from combined vectors.
        
        Returns:
            Tuple of (label_string, combined_vector)
        """
        combined = self.forward(v_llm, v_lstm)
        
        # Handle batch vs single sample
        if combined.dim() == 1:
            max_idx = torch.argmax(combined).item()
        else:
            max_idx = torch.argmax(combined, dim=1)
        
        labels = ['BUY', 'SELL', 'HOLD']
        
        if isinstance(max_idx, int):
            return labels[max_idx], combined
        else:
            return [labels[i] for i in max_idx.tolist()], combined


def ground_truth_to_onehot(gt_labels):
    """Convert ground truth labels to one-hot vectors."""
    label_map = {'BUY': [1, 0, 0], 'SELL': [0, 1, 0], 'HOLD': [0, 0, 1]}
    return torch.tensor([label_map[label] for label in gt_labels], dtype=torch.float32)


def load_combined_dataset(csv_path='../datasets/combined_llm_lstm_aapl_90day.csv', use_soft_lstm=True):
    """
    Load the combined training dataset.
    
    Args:
        csv_path: Path to combined CSV file
        use_soft_lstm: If True, use soft LSTM probabilities; if False, use hard one-hot
    
    Returns:
        Tuple of (llm_vectors, lstm_vectors, ground_truth_vectors, ground_truth_labels, df)
    """
    df = pd.read_csv(csv_path)
    
    # Extract LLM vectors [buy, sell, hold]
    llm_vectors = torch.tensor(
        df[['LLM_Buy', 'LLM_Sell', 'LLM_Hold']].values,
        dtype=torch.float32
    )
    
    # Extract LSTM vectors (soft or hard)
    if use_soft_lstm:
        lstm_vectors = torch.tensor(
            df[['LSTM_Buy_Soft', 'LSTM_Sell_Soft', 'LSTM_Hold_Soft']].values,
            dtype=torch.float32
        )
    else:
        lstm_vectors = torch.tensor(
            df[['LSTM_Buy_Hard', 'LSTM_Sell_Hard', 'LSTM_Hold_Hard']].values,
            dtype=torch.float32
        )
    
    # Ground truth
    gt_labels = df['Ground_Truth'].tolist()
    gt_vectors = ground_truth_to_onehot(gt_labels)
    
    return llm_vectors, lstm_vectors, gt_vectors, gt_labels, df


def train_combiner(model, llm_vectors, lstm_vectors, gt_vectors, 
                   epochs=1000, lr=0.01, verbose=True):
    """
    Train the combiner model to learn optimal alpha/beta weights.
    
    Args:
        model: VectorCombiner instance
        llm_vectors: LLM prediction vectors (N, 3)
        lstm_vectors: LSTM prediction vectors (N, 3)
        gt_vectors: Ground truth one-hot vectors (N, 3)
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Print progress
    
    Returns:
        Training history dict
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Use NLLLoss since we're working with probability distributions
    # We'll apply log_softmax to the combined output before computing loss
    criterion = nn.NLLLoss()
    
    history = {'loss': [], 'accuracy': [], 'alpha': [], 'beta': []}
    
    # Convert ground truth to class indices
    gt_indices = torch.argmax(gt_vectors, dim=1)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass - get weighted combination of probability vectors
        combined = model(llm_vectors, lstm_vectors)
        
        # Apply log_softmax to convert combined probabilities to log-probabilities
        # This is mathematically correct for NLLLoss which expects log-probabilities
        log_probs = F.log_softmax(combined, dim=1)
        
        # Compute loss
        loss = criterion(log_probs, gt_indices)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            preds = torch.argmax(combined, dim=1)
            accuracy = (preds == gt_indices).float().mean().item()
        
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy)
        history['alpha'].append(model.alpha.item())
        history['beta'].append(model.beta.item())
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, "
                  f"Accuracy: {accuracy:.2%}, "
                  f"alpha: {model.alpha.item():.4f}, beta: {model.beta.item():.4f}")
    
    return history


def evaluate_model(model, llm_vectors, lstm_vectors, gt_labels, df=None):
    """
    Evaluate the trained model and print detailed results.
    """
    model.eval()
    with torch.no_grad():
        pred_labels, combined = model.predict(llm_vectors, lstm_vectors)
    
    # Calculate accuracies
    correct = sum(p == g for p, g in zip(pred_labels, gt_labels))
    accuracy = correct / len(gt_labels)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Combined Model Accuracy: {accuracy:.2%} ({correct}/{len(gt_labels)})")
    print(f"Learned alpha (LLM weight): {model.alpha.item():.4f}")
    print(f"Learned beta (LSTM weight): {model.beta.item():.4f}")
    
    # Compare with individual models
    if df is not None:
        llm_acc = (df['LLM_View'] == df['Ground_Truth']).mean()
        lstm_soft_acc = (df['LSTM_View_Soft'] == df['Ground_Truth']).mean()
        lstm_hard_acc = (df['LSTM_View_Hard'] == df['Ground_Truth']).mean()
        
        print(f"\nComparison:")
        print(f"  LLM alone:        {llm_acc:.2%}")
        print(f"  LSTM Soft alone:  {lstm_soft_acc:.2%}")
        print(f"  LSTM Hard alone:  {lstm_hard_acc:.2%}")
        print(f"  Combined:         {accuracy:.2%}")
        
        improvement = accuracy - max(llm_acc, lstm_soft_acc, lstm_hard_acc)
        if improvement > 0:
            print(f"\n  Improvement over best single model: +{improvement:.2%}")
        else:
            print(f"\n  Note: Combined model doesn't beat best single model (diff: {improvement:.2%})")
    
    # Detailed per-sample results
    print("\n" + "-"*60)
    print("Per-sample predictions:")
    print("-"*60)
    print(f"{'Date':<12} {'LLM':<6} {'LSTM':<6} {'Combined':<10} {'Ground Truth':<12} {'Match'}")
    print("-"*60)
    
    if df is not None:
        for i, row in df.iterrows():
            date = row['Date']
            llm_pred = row['LLM_View']
            lstm_pred = row['LSTM_View_Soft']
            combined_pred = pred_labels[i]
            gt = row['Ground_Truth']
            match = "Yes" if combined_pred == gt else "No"
            print(f"{date:<12} {llm_pred:<6} {lstm_pred:<6} {combined_pred:<10} {gt:<12} {match}")
    
    return accuracy, pred_labels


def combine_vectors_simple(v1, v2, alpha=0.5, beta=0.5):
    """
    Simple function to combine vectors without PyTorch model (for inference).

    Args:
        v1: First vector [buy, sell, hold] - LLM
        v2: Second vector [buy, sell, hold] - LSTM
        alpha: Weight for v1 (LLM)
        beta: Weight for v2 (LSTM)

    Returns:
        max_label: String indicating the maximum label
        combined_vector: The combined weighted vector
    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Combine vectors
    combined_vector = alpha * v1 + beta * v2

    # Get maximum label
    labels = ['BUY', 'SELL', 'HOLD']
    max_idx = np.argmax(combined_vector)
    max_label = labels[max_idx]

    return max_label, combined_vector


def main():
    """Main training and evaluation pipeline."""
    print("="*60)
    print("LLM + LSTM Combiner Training")
    print("="*60)
    
    # Train with soft LSTM probabilities
    print("\n[1] Training with SOFT LSTM probabilities...")
    print("-"*60)
    
    llm_soft, lstm_soft, gt_soft, gt_labels_soft, df_soft = load_combined_dataset(
        use_soft_lstm=True
    )
    
    model_soft = VectorCombiner(init_alpha=0.5, init_beta=0.5)
    history_soft = train_combiner(
        model_soft, llm_soft, lstm_soft, gt_soft,
        epochs=1000, lr=0.05, verbose=True
    )
    
    acc_soft, preds_soft = evaluate_model(
        model_soft, llm_soft, lstm_soft, gt_labels_soft, df_soft
    )
    
    # Train with hard LSTM probabilities
    print("\n" + "="*60)
    print("[2] Training with HARD LSTM probabilities (one-hot)...")
    print("-"*60)
    
    llm_hard, lstm_hard, gt_hard, gt_labels_hard, df_hard = load_combined_dataset(
        use_soft_lstm=False
    )
    
    model_hard = VectorCombiner(init_alpha=0.5, init_beta=0.5)
    history_hard = train_combiner(
        model_hard, llm_hard, lstm_hard, gt_hard,
        epochs=1000, lr=0.05, verbose=True
    )
    
    acc_hard, preds_hard = evaluate_model(
        model_hard, llm_hard, lstm_hard, gt_labels_hard, df_hard
    )
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Soft LSTM combiner: {acc_soft:.2%} accuracy")
    print(f"  alpha={model_soft.alpha.item():.4f}, beta={model_soft.beta.item():.4f}")
    print(f"\nHard LSTM combiner: {acc_hard:.2%} accuracy")
    print(f"  alpha={model_hard.alpha.item():.4f}, beta={model_hard.beta.item():.4f}")
    
    # Save best model weights
    if acc_soft >= acc_hard:
        best_model = model_soft
        best_type = "soft"
    else:
        best_model = model_hard
        best_type = "hard"
    
    print(f"\nBest model: {best_type.upper()} LSTM")
    print(f"Optimal weights: alpha={best_model.alpha.item():.4f}, beta={best_model.beta.item():.4f}")
    
    # Save model
    torch.save({
        'alpha': best_model.alpha.item(),
        'beta': best_model.beta.item(),
        'lstm_type': best_type,
        'accuracy': max(acc_soft, acc_hard)
    }, '../models/combiner_weights_90day.pt')
    print("\nSaved best model weights to combiner_weights.pt")
    
    return best_model, history_soft, history_hard


if __name__ == "__main__":
    best_model, hist_soft, hist_hard = main()
