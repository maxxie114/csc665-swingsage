# CSC665 SwingSage

Agentic stock analysis system that fuses an LSTM price forecaster, an LLM quantitative analyst, a learnable combiner, and Alpaca-powered execution. SwingSage includes a FastAPI server, a trading dashboard frontend, and multiple research notebooks to trace every prediction.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Notebooks & Modules](#notebooks--modules)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Workflows](#workflows)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [License](#license)

## Features
- **Unified Agent Notebook** (`auto_stock_trader_agent.ipynb`)
    - Loads pretrained `lstm_model.keras`, `models/scaler.json`, and `models/combiner_weights_90day.pt`.
    - Generates LSTM and LLM vectors, combines them, optionally executes trades, and can launch a FastAPI server.
    - Built around `uv` so a single `uv sync` satisfies every dependency.
- **LSTM Training** (`LSTM_Training_REGRESS_Import.ipynb` & `LSTM_Training_REGRESS_V2.ipynb`)
    - Colab-friendly notebooks that train the regression model and derive buy/sell/hold signals.
- **LLM Decisioning** (`llm_stock_analysis.ipynb`)
    - Fetches OHLCV data via yfinance, engineers indicators, crafts prompts, and parses strict JSON responses from OpenRouter.
- **Combiner Toolkit** (`combiner/`)
    - PyTorch `VectorCombiner` (`combine.py`) with dataset generators for aligning LLM/LSTM predictions.
- **Frontend Dashboard** (`frontend/`)
    - Flask app + static dashboard to view predictions and interact with Alpaca.
- **Datasets & Models**
    - Curated prediction CSVs (`datasets/`) and weight files (`models/`) tracked via git.

## Project Structure
```
csc665-swingsage/
├── auto_stock_trader_agent.ipynb   # Unified agent (LSTM + LLM + combiner + Alpaca + FastAPI)
├── llm_stock_analysis.ipynb        # Standalone LLM decision module
├── LSTM_Training_REGRESS_*.ipynb   # LSTM training & inference notebooks
├── combiner/
│   ├── combine.py                  # PyTorch combiner model
│   ├── generate_llm_data.py        # Build LLM prediction datasets
│   └── generate_lstm_predictions.py# Export LSTM vectors
├── datasets/                       # CSVs for combiner training
├── frontend/
│   ├── app.py                      # Flask dashboard + API bridge
│   ├── templates/index.html        # UI layout
│   └── static/                     # CSS/JS/assets
├── models/
│   ├── combiner_weights_30day.pt
│   ├── combiner_weights_90day.pt
│   ├── lstm_model.keras
│   └── scaler.json
├── PLANNER.md                      # Step-by-step execution plan
├── README.md                       # (this file)
├── pyproject.toml                  # uv dependency list
└── uv.lock                         # Resolved dependency versions
```

## Notebooks & Modules
| Component | Role | How to Run |
|-----------|------|------------|
| `auto_stock_trader_agent.ipynb` | LSTM + LLM + combiner + Alpaca + FastAPI | `uv run jupyter lab` → open notebook, or `uv run jupyter nbconvert --execute auto_stock_trader_agent.ipynb` |
| `llm_stock_analysis.ipynb` | LLM-only module (indicators + LangChain) | Same as above |
| `LSTM_Training_REGRESS_Import.ipynb` | Regression LSTM training/inference | Same as above (Colab friendly) |
| `combiner/combine.py` | Trainable fusion of vectors | `uv run python combiner/combine.py` |
| `frontend/app.py` | Flask dashboard for Alpaca | `cd frontend && uv run python app.py` |

## Getting Started
### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Alpaca Paper Trading API keys
- OpenRouter API key (DeepSeek/other LLMs)

### Install Dependencies
```bash
uv sync
```
This command installs every dependency defined in `pyproject.toml`, covering notebooks, FastAPI server, frontend backend, and research scripts.

### Launch Jupyter Lab
```bash
uv run jupyter lab
```
Open the notebook you wish to execute (e.g., `auto_stock_trader_agent.ipynb`).

### Run Unified Agent Headlessly
```bash
uv run jupyter nbconvert --to notebook --execute auto_stock_trader_agent.ipynb
```

### Run Combiner Script
```bash
uv run python combiner/combine.py
```

### Run Frontend Dashboard
```bash
cd frontend
uv run python app.py
```
Then visit `http://localhost:5000`.

## Configuration
Create `.env` in the repository root:
```
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=deepseek/deepseek-chat-v3.1
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ENABLE_SERVER=false
DRY_RUN=true
DEFAULT_TICKER=AAPL
DEFAULT_PERIOD=5y
LOOKBACK_DAYS=60
LLM_LOG_PATH=agent_run_log.json
USE_GPU=false
TRADE_QUANTITY=1
```
Additional optional variables: `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `ENABLE_SERVER`, `DRY_RUN` per environment, override `USE_GPU` to opt into GPU acceleration, and tweak `TRADE_QUANTITY` for consistent fills.

## Workflows
### Unified Agent Notebook
1. **Setup Cell** – imports, env vars, logging.
2. **Data Utils** – `fetch_stock_data`, `compute_indicators` (handles yfinance MultiIndex columns).
3. **LSTM Section** – loads `lstm_model.keras` + scaler, predicts next close, generates soft vector.
4. **LLM Section** – indicator snapshot, prompt, LangChain call, strict JSON parser.
5. **Combiner Section** – loads `models/combiner_weights_90day.pt`, merges vectors.
6. **Trading Section** – Alpaca order helper with `DRY_RUN` guard.
7. **Orchestration** – `run_agent()` prints JSON summary and optionally executes trade.
8. **FastAPI Section** – `ENABLE_SERVER=true` spawns `/health`, `/decision`, `/trade` endpoints.
9. **Tests Section** – smoke tests for parser + combiner plus handy `uv` commands.

### Frontend Dashboard
- Python/Flask backend with endpoints under `/api/*`.
- Uses Alpaca for account, positions, orders, history, quotes, trades.
- Integrates OpenRouter (Kimi K2) for conversational assistance.
- Static assets stored in `frontend/static/` with AMOLED-themed UI.

### Combiner Training
- `datasets/combined_llm_lstm_aapl_*.csv` provide aligned target data.
- `combiner/generate_*` scripts regenerate datasets when needed.
- `combine.py` trains weights, prints accuracy comparisons, and saves PT files.

## Testing
- Notebook smoke tests (`run_smoke_tests()` in `auto_stock_trader_agent.ipynb`).
- Combiner training script reports loss/accuracy each 100 epochs.
- Frontend has `/api/test` to verify Alpaca connectivity.
- Manual tests: run `uv run python combiner/combine.py` and ensure weights save; in dashboard, hit `/api/account`/`/api/orders` endpoints.

## Roadmap
- [ ] Expand FastAPI endpoints to stream LLM/LSTM vectors to frontend.
- [ ] Build CI workflow for linting and notebook execution.
- [ ] Integrate real-time WebSocket updates in dashboard.
- [ ] Package agent as a deployable service with Docker.

## License
Apache 2.0 (placeholder—update as needed).
