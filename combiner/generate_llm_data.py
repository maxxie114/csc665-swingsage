"""
Generate LLM training data for multiple dates.
Calls the LLM for each historical date to get Buy/Sell/Hold predictions.

Requirements:
- OPENROUTER_API_KEY environment variable must be set
- Run from project root: uv run python combiner/generate_llm_data.py
"""

import json
import os
import re
import time
# Client created fresh each time to avoid caching issues with API keys
from textwrap import dedent
from typing import Any, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import ta

# Load environment variables with override to ensure fresh values
load_dotenv(override=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = "deepseek/deepseek-chat-v3.1"  # This model works with the API
LOOKBACK_DAYS = 60
LLM_TEMPERATURE = 0.3
MAX_TOKENS = 800
GROUND_TRUTH_THRESHOLD = 0.005  # 0.5% move for BUY/SELL


def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Return daily OHLCV data for the requested ticker."""
    data = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data retrieved for ticker {ticker} and period {period}.")

    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in data.columns and isinstance(data[col], pd.DataFrame):
            data[col] = data[col].squeeze("columns")

    data = data.dropna(how="all")
    data.index = pd.to_datetime(data.index)
    return data


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Augment OHLCV data with technical indicators."""
    if df.empty:
        raise ValueError("Input price DataFrame is empty.")

    data = df.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    data["SMA_10"] = ta.trend.SMAIndicator(close=close, window=10).sma_indicator()
    data["SMA_20"] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
    data["SMA_50"] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    data["SMA_100"] = ta.trend.SMAIndicator(close=close, window=100).sma_indicator()
    data["SMA_200"] = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    data["EMA_12"] = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    data["EMA_26"] = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()
    data["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd = ta.trend.MACD(close=close)
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()
    data["MACD_Hist"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    data["BB_Upper"] = bb.bollinger_hband()
    data["BB_Middle"] = bb.bollinger_mavg()
    data["BB_Lower"] = bb.bollinger_lband()

    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    data["Stoch_K"] = stoch.stoch()
    data["Stoch_D"] = stoch.stoch_signal()

    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
    data["OBV"] = obv.on_balance_volume()

    data["Daily_Return"] = close.pct_change()
    data = data.dropna()

    return data


def build_llm_input(df: pd.DataFrame, lookback_days: int = LOOKBACK_DAYS) -> str:
    """Create a textual snapshot of recent price action and indicators."""
    if df.empty:
        raise ValueError("Cannot build LLM input from empty DataFrame.")

    window = df.tail(lookback_days)

    display_cols = [
        "Close", "SMA_10", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "BB_Upper",
        "BB_Middle", "BB_Lower", "Stoch_K", "Stoch_D", "OBV",
        "Volume", "Daily_Return",
    ]

    available_cols = [col for col in display_cols if col in window.columns]
    snapshot = window[available_cols].copy()
    snapshot = snapshot.round({col: 4 for col in available_cols})

    lines = []
    for idx, row in snapshot.iterrows():
        line = (
            f"{idx.date().isoformat()} | "
            f"Close={float(row.get('Close', float('nan'))):.2f}, "
            f"SMA10={float(row.get('SMA_10', float('nan'))):.2f}, "
            f"SMA50={float(row.get('SMA_50', float('nan'))):.2f}, "
            f"SMA200={float(row.get('SMA_200', float('nan'))):.2f}, "
            f"RSI14={float(row.get('RSI_14', float('nan'))):.1f}, "
            f"MACD={float(row.get('MACD', float('nan'))):.3f}"
        )
        lines.append(line)

    price_change_pct = (window["Close"].iloc[-1] / window["Close"].iloc[0] - 1) * 100
    daily_returns = window["Daily_Return"].dropna()
    annual_vol = daily_returns.std() * np.sqrt(252) * 100 if not daily_returns.empty else 0.0

    summary = dedent(f"""
        Recent performance summary:
        - Lookback window: {len(window)} trading days
        - Net close change: {price_change_pct:.2f}%
        - Annualized volatility (est.): {annual_vol:.2f}%
    """).strip()

    return f"{summary}\n\nDaily snapshots (most recent last):\n" + "\n".join(lines)


def build_llm_prompt(ticker: str, market_snapshot: str, lookback_days: int = LOOKBACK_DAYS) -> str:
    """Compose the instruction payload for the LLM."""
    return dedent(f"""
        You are an expert quantitative analyst supporting an automated daily stock trading system.
        Evaluate the provided market context for ticker {ticker}.

        Data characteristics:
        - Frequency: daily candles (one decision per trading day).
        - Horizon: predict the next trading day's closing price behavior only.
        - Goal: produce BUY, SELL, HOLD confidence scores that sum to ~1.

        Market context extracted from the last {lookback_days} trading days:

        {market_snapshot}

        Instructions:
        - Analyze trends, momentum, volatility, and mean-reversion signals from the data.
        - Determine whether the next day's closing price is likely to rise sharply (BUY), fall sharply (SELL), or stay relatively neutral (HOLD).
        - Return a strict JSON object with keys: buy_confidence, sell_confidence, hold_confidence, next_day_view, explanation.
        - Confidence values must be floats between 0 and 1 and collectively sum to approximately 1.
        - Set next_day_view to BUY, SELL, or HOLD depending on the dominant signal.
        - Do not include any additional text outside the JSON object.
    """).strip()


def get_llm_client() -> ChatOpenAI:
    """Instantiate and cache the LangChain ChatOpenAI client."""
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")

    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=90,
    )


def invoke_llm(prompt: str) -> str:
    """Send the prompt to the LLM and return the raw text response."""
    client = get_llm_client()
    messages = [
        SystemMessage(content="You are a disciplined trading assistant. Follow instructions exactly and respond with strict JSON."),
        HumanMessage(content=prompt),
    ]
    response = client.invoke(messages)
    return response.content


def parse_llm_decision(raw_text: str) -> Dict[str, Any]:
    """Parse the LLM JSON payload and enforce expected structure."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            print(f"DEBUG: Raw response (first 500 chars): {raw_text[:500]}")
            raise ValueError("LLM response is not valid JSON.") from None
        payload = json.loads(match.group())

    required_keys = {"buy_confidence", "sell_confidence", "hold_confidence", "next_day_view"}
    missing = required_keys - payload.keys()
    if missing:
        raise ValueError(f"LLM response missing keys: {sorted(missing)}")

    confidences = {}
    for key in ["buy_confidence", "sell_confidence", "hold_confidence"]:
        value = float(payload[key])
        value = max(0.0, min(1.0, value))
        confidences[key] = value

    total = sum(confidences.values())
    if total <= 0:
        raise ValueError("Confidence scores sum to zero.")
    confidences = {k: v / total for k, v in confidences.items()}

    return {
        "buy_confidence": confidences["buy_confidence"],
        "sell_confidence": confidences["sell_confidence"],
        "hold_confidence": confidences["hold_confidence"],
        "next_day_view": str(payload.get("next_day_view", "HOLD")).upper().strip(),
    }


def generate_llm_training_data(
    ticker: str = "AAPL",
    days: int = 60,
    output_path: str = None,
    delay: float = 0.5
) -> pd.DataFrame:
    """
    Generate LLM training data for the specified ticker and number of days.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of historical days to generate predictions for
        output_path: Path to save CSV (default: llm_training_data_{ticker}.csv)
        delay: Delay between API calls in seconds
    
    Returns:
        DataFrame with LLM predictions and ground truth
    """
    if output_path is None:
        output_path = f"../datasets/llm_predictions_{ticker.lower()}.csv"
    
    print(f"Generating LLM training data for {ticker} over the last {days} days...")
    print(f"This will make {days} API calls with {delay}s delay each.")
    print(f"Estimated time: {days * (delay + 2) / 60:.1f} minutes\n")
    
    # Fetch full history
    full_data = fetch_stock_data(ticker, period="2y")
    full_data = compute_indicators(full_data)
    
    valid_dates = full_data.index
    if len(valid_dates) < days + 20:
        raise ValueError(f"Not enough data. Need at least {days + 20} days, got {len(valid_dates)}")
    
    # Target the last 'days' decision points (excluding very last day for ground truth)
    target_indices = range(len(valid_dates) - days - 1, len(valid_dates) - 1)
    
    dataset = []
    
    for i, idx in enumerate(target_indices):
        current_date = valid_dates[idx]
        next_date = valid_dates[idx + 1]
        
        # Slice data to simulate "past" knowledge only
        past_data = full_data.loc[:current_date]
        
        try:
            # Generate LLM prediction
            market_snapshot = build_llm_input(past_data, lookback_days=LOOKBACK_DAYS)
            prompt = build_llm_prompt(ticker, market_snapshot, lookback_days=LOOKBACK_DAYS)
            
            raw_response = invoke_llm(prompt)
            decision = parse_llm_decision(raw_response)
            
            # Compute ground truth from actual next-day return
            next_return = full_data.loc[next_date]['Daily_Return']
            
            if next_return > GROUND_TRUTH_THRESHOLD:
                ground_truth = "BUY"
            elif next_return < -GROUND_TRUTH_THRESHOLD:
                ground_truth = "SELL"
            else:
                ground_truth = "HOLD"
            
            match = "✓" if decision['next_day_view'] == ground_truth else "✗"
            # Print every 5th entry to reduce output spam, plus first and last
            if (i + 1) % 5 == 0 or i == 0 or i == days - 1:
                print(f"[{i+1}/{days}] {current_date.date()} -> LLM: {decision['next_day_view']:4} | Truth: {ground_truth:4} | Ret: {next_return:+.2%} {match}")
            
            dataset.append({
                "Date": current_date.date(),
                "Buy_Conf": round(decision["buy_confidence"], 2),
                "Sell_Conf": round(decision["sell_confidence"], 2),
                "Hold_Conf": round(decision["hold_confidence"], 2),
                "LLM_View": decision["next_day_view"],
                "Ground_Truth": ground_truth,
                "Next_Return": next_return
            })
            
        except Exception as e:
            # Print errors every 5th entry to reduce spam
            if (i + 1) % 5 == 0 or i == 0 or i == days - 1:
                print(f"[{i+1}/{days}] {current_date.date()} -> ERROR: {e}")
        
        # Rate limit delay
        time.sleep(delay)
    
    if dataset:
        df_out = pd.DataFrame(dataset)
        df_out.to_csv(output_path, index=False)
        
        # Calculate accuracy
        accuracy = (df_out['LLM_View'] == df_out['Ground_Truth']).mean()
        
        print(f"\n{'='*60}")
        print(f"Successfully saved {len(df_out)} rows to {output_path}")
        print(f"LLM Accuracy: {accuracy:.2%}")
        print(f"{'='*60}")
        
        return df_out
    else:
        print("No data generated.")
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LLM training data for stock predictions")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker (default: AAPL)")
    parser.add_argument("--days", type=int, default=60, help="Number of days to generate (default: 60)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls in seconds")
    
    args = parser.parse_args()
    
    generate_llm_training_data(
        ticker=args.ticker,
        days=args.days,
        output_path=args.output,
        delay=args.delay
    )

