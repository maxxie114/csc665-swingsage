# CSC665 SwingSage: Agentic Stock Analysis System

## Overview
SwingSage is an advanced, agentic system designed to automate stock trading decisions. By combining deep learning (LSTM) for price trend prediction with Large Language Models (LLM) for market sentiment and indicator analysis, the system aims to make robust Buy/Sell/Hold decisions and autonomously execute trades via the Alpaca API.

The project is currently composed of three specialized modules that will eventually merge into a single agentic operational flow.

## Modules

### 1. `combine.py`
This module acts as the decision fusion engine. It provides mechanisms to combine trading signals from different sources (e.g., the LSTM model and the LLM agent).
- **Functionality**: Implements a `VectorCombiner` using PyTorch to learn optimal weights for merging signal vectors [Buy, Sell, Hold].
- **Key Features**: 
    - Learnable `alpha` and `beta` parameters for weighted averaging.
    - Support for both simple linear combination and a trainable neural network module.

### 2. `llm_stock_analysis.ipynb`
This notebook houses the LLM-driven quantitative analyst.
- **Functionality**: Fetches historical stock data and calculates a comprehensive suite of technical indicators (RSI, MACD, Bollinger Bands, etc.). It then feeds a textual snapshot of the market state to a Large Language Model (e.g., DeepSeek via OpenRouter) to generate a trading confidence vector.
- **Key Features**:
    - Automated extraction of technical indicators using the `ta` library.
    - Prompt engineering for financial decision-making.
    - structured JSON output enforcing Buy/Sell/Hold confidence scores.

### 3. `LSTM_Training_REGRESS_Import.ipynb`
This notebook implements the deep learning component focused on price trend forecasting.
- **Functionality**: Trains and runs an LSTM (Long Short-Term Memory) neural network to predict future stock prices based on historical sequences.
- **Key Features**:
    - Data preprocessing using `MinMaxScaler` and `yfinance`.
    - LSTM architecture for time-series regression.
    - Logic to convert predicted price trends into discrete Buy/Sell/Hold signals based on percentage thresholds.

## Future Roadmap: The Unified Agent
The next phase of development involves integrating these three modules into a cohesive, single-notebook or single-script agent. This unified system will:
1. **Perceive**: Automatically fetch live market data.
2. **Analyze**: Run parallel analysis streams—quantitative (LSTM) and qualitative/technical (LLM).
3. **Decide**: Fuse these signals using the logic from `combine.py` to form a final high-confidence decision.
4. **Act**: Execute the trade orders directly using the Alpaca API (to be implemented).

## Dependency Management
This project uses **uv** for fast and reliable dependency management. All required packages are listed in `pyproject.toml`.

## Deployment & Running Instructions

### 1. Install Dependencies
First, ensure you have [uv](https://github.com/astral-sh/uv) installed. Then, sync the project dependencies:
```bash
uv sync
```

### 2. Running the Modules
Each module can be run independently using `uv run`.

#### Running the Combiner Script
To run the decision fusion engine:
```bash
uv run combine.py
```

#### Running the Notebooks
To interact with the analysis or training notebooks, you need to launch Jupyter.

**Option A: Launch Jupyter Lab**
```bash
uv run jupyter lab
```
Then open `llm_stock_analysis.ipynb` or `LSTM_Training_REGRESS_Import.ipynb` from the interface.

**Option B: Execute a Notebook directly (headless)**
If you want to just execute the notebook from the command line:
```bash
uv run jupyter nbconvert --to notebook --execute llm_stock_analysis.ipynb
```
