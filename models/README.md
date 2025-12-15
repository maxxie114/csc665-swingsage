# Models

This folder contains trained model weights for the SwingSage stock prediction system.

## LSTM Model

- `lstm_model.keras` - LSTM model trained on AAPL stock data (2 years)
- `scaler.json` - MinMaxScaler parameters for normalizing price data

The LSTM predicts next-day closing prices, which are then converted to Buy/Sell/Hold signals based on predicted price movement.

## Combiner Models

The combiner learns optimal weights to blend LLM and LSTM predictions.

### combiner_weights_30day.pt

Trained on 30 samples (Oct 24 - Dec 5, 2025)

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| alpha (LLM) | 2.0544 | 66% weight |
| beta (LSTM) | 1.0641 | 34% weight |
| Accuracy | 53.33% | 16/30 correct |

This model trusts the LLM more than the LSTM.

### combiner_weights_90day.pt

Trained on 90 samples (Aug 5 - Dec 10, 2025)

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| alpha (LLM) | 0.2093 | 14% weight |
| beta (LSTM) | 1.3338 | 86% weight |
| Accuracy | 46.67% | 42/90 correct |

This model trusts the LSTM more than the LLM (because LLM performed poorly on extended data).

## Usage

```python
import torch

# Load weights
weights = torch.load('models/combiner_weights_90day.pt', weights_only=True)
alpha = weights['alpha']  # LLM weight
beta = weights['beta']    # LSTM weight

# Combine predictions
combined = alpha * llm_vector + beta * lstm_vector
prediction = ['BUY', 'SELL', 'HOLD'][combined.argmax()]
```

## Individual Model Performance (90-day dataset)

| Model | Accuracy |
|-------|----------|
| LLM alone | 32.22% |
| LSTM Soft | 45.56% |
| LSTM Hard | 46.67% |
| Combined | 46.67% |

