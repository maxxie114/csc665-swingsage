"""
Generate LSTM predictions for the same dates as the LLM training data.
Creates a combined dataset with both LLM and LSTM predictions for training the combiner model.
"""

import numpy as np
import pandas as pd
import json
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


def load_lstm_model_and_scaler(model_path='lstm_model.keras', scaler_path='scaler.json'):
    """Load the trained LSTM model and reconstruct the scaler."""
    model = load_model(model_path)
    
    scaler = MinMaxScaler()
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    
    if 'min_' in scaler_params:
        scaler.min_ = np.array(scaler_params['min_'])
    if 'scale_' in scaler_params:
        scaler.scale_ = np.array(scaler_params['scale_'])
    if 'data_min_' in scaler_params:
        scaler.data_min_ = np.array(scaler_params['data_min_'])
    if 'data_max_' in scaler_params:
        scaler.data_max_ = np.array(scaler_params['data_max_'])
    if 'data_range_' in scaler_params:
        scaler.data_range_ = np.array(scaler_params['data_range_'])
    if 'n_features_in_' in scaler_params:
        scaler.n_features_in_ = scaler_params['n_features_in_']
    
    return model, scaler


def download_stock_data(ticker, start_date, end_date):
    """Download stock data from yfinance."""
    data = yf.download(ticker, start=start_date, end=end_date)[['Close']].dropna()
    return data


def predict_price(model, scaler, historical_prices, time_step=100):
    """
    Predict the next day's closing price given historical prices.
    
    Args:
        model: Trained LSTM model
        scaler: Fitted MinMaxScaler
        historical_prices: Array of closing prices (at least time_step days)
        time_step: Number of days to look back (default 100)
    
    Returns:
        Predicted next day price
    """
    if len(historical_prices) < time_step:
        raise ValueError(f"Need at least {time_step} days of data, got {len(historical_prices)}")
    
    # Take last time_step days
    recent_prices = historical_prices[-time_step:].reshape(-1, 1)
    
    # Scale the data
    scaled_prices = scaler.transform(recent_prices)
    
    # Reshape for LSTM: (1, time_step, 1)
    X = scaled_prices.reshape(1, time_step, 1)
    
    # Predict
    pred_scaled = model.predict(X, verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]
    
    return pred_price


def generate_signal_hard(pct_change, threshold=0.005):
    """
    Generate hard (one-hot) signal based on percentage change.
    
    Args:
        pct_change: Predicted percentage change
        threshold: Threshold for buy/sell decision (default 0.5%)
    
    Returns:
        Tuple of (signal_name, one_hot_vector)
    """
    if pct_change > threshold:
        return "SELL", [0, 1, 0]  # Sell if price going up (sell high)
    elif pct_change < -threshold:
        return "BUY", [1, 0, 0]   # Buy if price going down (buy low)
    else:
        return "HOLD", [0, 0, 1]


def generate_signal_soft(pct_change, threshold=0.005, temperature=0.01):
    """
    Generate soft probability distribution based on percentage change.
    Uses softmax-like scaling to convert % change to probabilities.
    
    Args:
        pct_change: Predicted percentage change
        threshold: Base threshold for scaling
        temperature: Controls sharpness of distribution
    
    Returns:
        Tuple of (signal_name, probability_vector [buy, sell, hold])
    """
    # Scale the percentage change relative to threshold
    # Positive change -> higher sell probability
    # Negative change -> higher buy probability
    # Near zero -> higher hold probability
    
    sell_score = max(0, pct_change / temperature)
    buy_score = max(0, -pct_change / temperature)
    hold_score = max(0, (threshold - abs(pct_change)) / temperature)
    
    # Softmax normalization
    scores = np.array([buy_score, sell_score, hold_score])
    exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
    probs = exp_scores / exp_scores.sum()
    
    # Determine signal name based on highest probability
    signal_names = ["BUY", "SELL", "HOLD"]
    signal = signal_names[np.argmax(probs)]
    
    return signal, probs.tolist()


def generate_lstm_predictions(llm_data_path='llm_training_data_aapl.csv', 
                               ticker='AAPL',
                               output_path='combined_training_data.csv'):
    """
    Generate LSTM predictions for all dates in the LLM training data.
    
    Args:
        llm_data_path: Path to LLM training data CSV
        ticker: Stock ticker symbol
        output_path: Path for output combined dataset
    """
    print("Loading LSTM model and scaler...")
    model, scaler = load_lstm_model_and_scaler()
    
    print(f"Loading LLM training data from {llm_data_path}...")
    llm_df = pd.read_csv(llm_data_path)
    
    # Parse dates
    dates = pd.to_datetime(llm_df['Date'])
    min_date = dates.min()
    max_date = dates.max()
    
    # Need 100+ days before the first date for LSTM lookback, plus buffer
    start_date = (min_date - timedelta(days=150)).strftime('%Y-%m-%d')
    end_date = (max_date + timedelta(days=5)).strftime('%Y-%m-%d')
    
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    stock_data = download_stock_data(ticker, start_date, end_date)
    
    print(f"Stock data shape: {stock_data.shape}")
    print(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
    
    # Convert stock data to dict for easy lookup
    stock_prices = stock_data['Close'].to_dict()
    
    # Results storage
    results = []
    
    print(f"\nGenerating LSTM predictions for {len(llm_df)} dates...")
    
    for idx, row in llm_df.iterrows():
        date_str = row['Date']
        target_date = pd.to_datetime(date_str)
        
        # Find the closest trading day on or before target_date
        available_dates = [d for d in stock_data.index if d <= target_date]
        if not available_dates:
            print(f"  Warning: No data available for {date_str}, skipping...")
            continue
        
        actual_date = max(available_dates)
        
        # Get historical prices up to this date (need at least 101 days for prediction)
        historical_mask = stock_data.index <= actual_date
        historical_data = stock_data[historical_mask]['Close'].values
        
        if len(historical_data) < 101:
            print(f"  Warning: Not enough historical data for {date_str} ({len(historical_data)} days), skipping...")
            continue
        
        # Predict today's price (using 100 days before)
        pred_today = predict_price(model, scaler, historical_data[:-1])
        
        # Predict tomorrow's price (using last 100 days including today)
        pred_tomorrow = predict_price(model, scaler, historical_data)
        
        # Calculate predicted percentage change
        pct_change = (pred_tomorrow - pred_today) / pred_today
        
        # Generate signals
        hard_signal, hard_probs = generate_signal_hard(pct_change)
        soft_signal, soft_probs = generate_signal_soft(pct_change)
        
        # Get actual price for reference
        actual_price = float(historical_data[-1])
        
        result = {
            'Date': date_str,
            # LLM data
            'LLM_Buy': row['Buy_Conf'],
            'LLM_Sell': row['Sell_Conf'],
            'LLM_Hold': row['Hold_Conf'],
            'LLM_View': row['LLM_View'],
            # LSTM soft probabilities
            'LSTM_Buy_Soft': round(soft_probs[0], 4),
            'LSTM_Sell_Soft': round(soft_probs[1], 4),
            'LSTM_Hold_Soft': round(soft_probs[2], 4),
            'LSTM_View_Soft': soft_signal,
            # LSTM hard (one-hot) probabilities
            'LSTM_Buy_Hard': hard_probs[0],
            'LSTM_Sell_Hard': hard_probs[1],
            'LSTM_Hold_Hard': hard_probs[2],
            'LSTM_View_Hard': hard_signal,
            # LSTM prediction details
            'LSTM_Pred_Today': round(pred_today, 2),
            'LSTM_Pred_Tomorrow': round(pred_tomorrow, 2),
            'LSTM_Pct_Change': round(pct_change * 100, 4),
            # Ground truth
            'Ground_Truth': row['Ground_Truth'],
            'Next_Return': row['Next_Return'],
            'Actual_Price': actual_price
        }
        
        results.append(result)
        print(f"  {date_str}: LSTM={soft_signal} (soft), {hard_signal} (hard), LLM={row['LLM_View']}, GT={row['Ground_Truth']}")
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved combined dataset to {output_path}")
    print(f"Total samples: {len(output_df)}")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"LSTM Soft accuracy: {(output_df['LSTM_View_Soft'] == output_df['Ground_Truth']).mean():.2%}")
    print(f"LSTM Hard accuracy: {(output_df['LSTM_View_Hard'] == output_df['Ground_Truth']).mean():.2%}")
    print(f"LLM accuracy: {(output_df['LLM_View'] == output_df['Ground_Truth']).mean():.2%}")
    
    return output_df


if __name__ == "__main__":
    df = generate_lstm_predictions()
    print("\nFirst few rows of combined dataset:")
    print(df.head())

