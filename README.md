# CSC665 SwingSage

SwingSage is a collaborative trading co-pilot that fuses quantitative forecasting and language-model reasoning into one agentic workflow. A multi-layer LSTM tracks price structure, DeepSeek-based LLM prompts interpret engineered indicators, and a learnable combiner reconciles the signals before Alpaca paper trading executes guarded orders. FastAPI, a lightweight Flask dashboard, and Jupyter notebooks round out the toolkit so every prediction is transparent, testable, and deployable on both laptops and Google Colab.

**Project Team:** 
- Max Xie
- Joseph Alhambra
- Atharva Walawalker

## Contents
- [Repository Map](#repository-map)
- [Module Overview](#module-overview)
- [Deployment Guide](#deployment-guide)
- [Configuration](#configuration)
- [Operational Notes](#operational-notes)
- [Roadmap](#roadmap)

## Repository Map

```
csc665-swingsage/
├── .env.example
├── README.md
├── auto_stock_trader_agent.ipynb
├── llm_stock_analysis.ipynb
├── LSTM_Training_REGRESS_Import.ipynb
├── LSTM_Training_REGRESS_V2.ipynb
├── combiner/
│   ├── combine.py
│   ├── generate_llm_data.py
│   └── generate_lstm_predictions.py
├── datasets/
│   ├── combined_llm_lstm_aapl_30day.csv
│   ├── combined_llm_lstm_aapl_90day.csv
│   ├── llm_predictions_aapl.csv
│   └── llm_predictions_nvda.csv
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   ├── templates/index.html
│   └── static/
│       ├── favicon.png
│       ├── logo.png
│       ├── script.js
│       └── style.css
├── models/
│   ├── combiner_weights_30day.pt
│   ├── combiner_weights_90day.pt
│   ├── lstm_model.keras
│   └── scaler.json
├── run_cloudflare.sh
├── cloudflared
├── pyproject.toml
└── uv.lock
```

- Root-level helpers such as [.env.example](.env.example) and [.python-version](.python-version) scaffold environment configs and tooling.
- [combiner/](combiner/) holds PyTorch utilities for generating data and training the VectorCombiner that blends LLM and LSTM outputs.
- [datasets/](datasets/) aggregates CSV artifacts that align predictions across modules for replay and supervised learning.
- [frontend/](frontend/) delivers the Flask-based dashboard along with Vercel metadata and static assets.
- [models/](models/) stores pretrained weights and scaling parameters required by the notebooks and API.
- Operational helpers such as [run_cloudflare.sh](run_cloudflare.sh) manage Cloudflare tunnels for notebook exposure.
- Documentation and research artifacts include [csc_665_final_report_team1.pdf](csc_665_final_report_team1.pdf) alongside notebook commentary.

## Module Overview

| Module | Primary Files | Responsibilities | Highlights |
| --- | --- | --- | --- |
| Unified Agent | [auto_stock_trader_agent.ipynb](auto_stock_trader_agent.ipynb) | Runs the full pipeline from yfinance ingestion through Alpaca order placement and optional FastAPI hosting. | Includes smoke tests, guardrails, and notebook-friendly logging for reproducibility. |
| LSTM Forecaster | [LSTM_Training_REGRESS_V2.ipynb](LSTM_Training_REGRESS_V2.ipynb), [models/lstm_model.keras](models/lstm_model.keras), [models/scaler.json](models/scaler.json) | Trains and serves a multi-layer LSTM to regress next-day close and produce soft BUY/SELL/HOLD vectors. | Uses MinMax scaling and 100-day windows; tuned for AAPL but extensible to other tickers. |
| LLM Decisioning | [llm_stock_analysis.ipynb](llm_stock_analysis.ipynb), [combiner/generate_llm_data.py](combiner/generate_llm_data.py) | Crafts indicator-driven prompts, calls DeepSeek via OpenRouter, and stores normalized probabilities. | Swapped from DeepSeek v3.2 to v3.1 for latency; enforces strict JSON responses with LangChain. |
| Signal Combiner | [combiner/combine.py](combiner/combine.py), [datasets/combined_llm_lstm_aapl_90day.csv](datasets/combined_llm_lstm_aapl_90day.csv), [models/combiner_weights_90day.pt](models/combiner_weights_90day.pt) | Learns alpha/beta weights that blend LLM and LSTM outputs into a final action vector. | Training uses Negative Log-Likelihood loss and confirms 46.67% accuracy over a 90-day replay. |
| Frontend & API Bridge | [frontend/app.py](frontend/app.py), [frontend/templates/index.html](frontend/templates/index.html), [frontend/static/script.js](frontend/static/script.js) | Provides a dashboard for positions, orders, and agent recommendations with Alpaca integration. | Compatible with Vercel deployment; reads API base from environment to point at Colab or local tunnel. |
| Deployment Utilities | [run_cloudflare.sh](run_cloudflare.sh), [cloudflared](cloudflared) | Enables secure exposure of notebook FastAPI services through Cloudflare tunnels. | Script handles named tunnel tokens so the FastAPI host remains stable. |
| Research Artifacts | [csc_665_final_report_team1.pdf](csc_665_final_report_team1.pdf) | Archives project documentation for traceability. | Mirrors the narrative captured in the final report and README. |

## Deployment Guide

### Local Prerequisites
- Install Python 3.12 (pyenv honors [.python-version](.python-version)).
- Install uv (`pip install uv`) or use the binary from Astral, then run `uv self update`.
- Request Alpaca paper credentials and an OpenRouter API key with DeepSeek access.

### Environment Setup
- Review the keys in [.env.example](.env.example) and export matching environment variables (for example, `export OPENROUTER_API_KEY=...`).
- Configure equivalent values for the dashboard using the names listed in [frontend/.env.example](frontend/.env.example) through your shell or deployment platform.
- Install dependencies once with:

```bash
uv sync
```

### Running the Unified Agent Notebook Locally
- Launch Jupyter Lab:

```bash
uv run jupyter lab
```

- Open [auto_stock_trader_agent.ipynb](auto_stock_trader_agent.ipynb), run Cell 1 through Cell 9 sequentially, confirm `DRY_RUN` remains true until live testing, and re-run the FastAPI cell after configuration edits.
- For headless execution or CI smoke tests, use:

```bash
uv run jupyter nbconvert --to notebook --execute auto_stock_trader_agent.ipynb
```

### Starting the FastAPI Service from the Notebook Artefact
- Ensure the environment variable `ENABLE_SERVER` is set to `true` before running the FastAPI cell.
- Execute the FastAPI cell (Cell 10) in [auto_stock_trader_agent.ipynb](auto_stock_trader_agent.ipynb); it binds to `http://0.0.0.0:8000` and surfaces `/health`, `/decision`, and `/trade` endpoints.

### Running the Flask Frontend Locally
- Open the project root, then run:

```bash
cd frontend
uv run python app.py
```

- Set `API_BASE_URL` in [frontend/.env](frontend/.env) to the FastAPI host (e.g., `http://localhost:8000` or a Cloudflare tunnel URL).

### Exposing the Notebook Server via Cloudflare Tunnel
- Review `.env.example` to identify required environment variables and set them explicitly inside the notebook, for example:

```python
import os
os.environ["DRY_RUN"] = "true"
os.environ["ENABLE_SERVER"] = "true"
os.environ["USE_GPU"] = "false"
# repeat for API keys shown in .env.example
```
- From your machine or Colab terminal, run:

```bash
chmod +x run_cloudflare.sh
./run_cloudflare.sh
```

- The helper script relies on a Cloudflare **named tunnel** token, so it never falls back to temporary trycloudflare.com links.

### Deploying the Frontend to Vercel
- Confirm [frontend/vercel.json](frontend/vercel.json) routes `/api/*` to the Flask backend when hosted.
- Use `vercel deploy` from the `frontend` directory or push to a connected Git repository; Vercel reads requirements from [frontend/requirements.txt](frontend/requirements.txt).
- Populate Vercel project environment variables with Alpaca keys and `NEXT_PUBLIC_API_URL` targeting the FastAPI tunnel.

### Running in Google Colab
- Start a fresh Colab runtime with GPU disabled unless explicitly needed for experimentation.
- Clone the repository inside `/content` and copy model assets to the runtime root:

```bash
%cd /content
!git clone https://github.com/<your-org>/csc665-swingsage.git
%cd csc665-swingsage
!cp -r models /content/models
!cp pyproject.toml /content/pyproject.toml
```

- Inspect `.env.example` for the required keys and set them directly in the notebook runtime, for example:

```python
import os
os.environ["DRY_RUN"] = "true"
os.environ["ENABLE_SERVER"] = "true"
os.environ["USE_GPU"] = "false"
# add OpenRouter and Alpaca keys as shown in .env.example
```
- Install Python dependencies with uv’s Colab-safe command:

```bash
!uv pip install --system -r pyproject.toml
```

- Open [auto_stock_trader_agent.ipynb](auto_stock_trader_agent.ipynb) inside Colab, execute Cells 2–10 sequentially, and confirm the FastAPI log stream cell updates live.
- Copy [run_cloudflare.sh](run_cloudflare.sh) into `/content/run_cloudflare.sh`, inject your Cloudflare token, and execute it to expose the FastAPI server externally.
- Point the deployed frontend or local browser to the stable hostname configured for your named tunnel (e.g., `https://swingsage.qubemc.com`).

### Notebook Server File Execution Outside Jupyter
- To run the FastAPI service without the notebook UI, convert the notebook to a script and execute via uv:

```bash
uv run jupyter nbconvert --to script auto_stock_trader_agent.ipynb --output agent_server
uv run python agent_server.py
```

- Ensure the script runs in a shell where the required environment variables are already exported.

### Frontend plus Notebook End-to-End Sanity Test
- Start the notebook FastAPI service locally or in Colab.
- Launch the frontend as described above.
- Visit `/health` through the browser network tab to verify connectivity, then trigger a `/decision` call from the dashboard to confirm the full LSTM → LLM → combiner path executes without error.

## Configuration
- Export environment variables that mirror [.env.example](.env.example); the snippet below annotates each value with its purpose:

```dotenv
# OpenRouter API token used for DeepSeek requests
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY

# Base URL for OpenRouter; change only when targeting a proxy
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Default LLM identifier served by OpenRouter
OPENROUTER_MODEL=deepseek/deepseek-chat-v3.1

# Safety model that approves requests before they reach the main assistant
OPENROUTER_GUARD_MODEL=meta-llama/llama-guard-4-12b

# Controls LLM randomness (0 = deterministic, 1 = exploratory)
LLM_TEMPERATURE=0.3

# Maximum tokens returned by each LLM response
LLM_MAX_TOKENS=800

# Enable or disable the guard model (leave true for production)
ENABLE_CHAT_GUARD=true

# Alpaca paper trading API credentials and host
ALPACA_API_KEY=YOUR_ALPACA_API_KEY
ALPACA_SECRET_KEY=YOUR_ALPACA_SECRET_KEY
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Agent guardrails and default market configuration
ENABLE_SERVER=false      # Toggle FastAPI server launch inside the notebook
DRY_RUN=true             # Prevent live trades; set false only for verified paper trading
DEFAULT_TICKER=AAPL      # Fallback ticker when the user does not supply one
DEFAULT_PERIOD=5y        # Historical window fetched from yfinance
LOOKBACK_DAYS=60         # Number of days summarized for the LLM prompt

# Optional logging file name used when persisting LLM outputs
LLM_LOG_PATH=agent_run_log.json

# Hardware and trade sizing controls
USE_GPU=false            # Set true to allow TensorFlow to use available GPUs
TRADE_QUANTITY=1         # Default number of shares per trade when orders are placed
```

- Configure frontend secrets using the variable names listed in [frontend/.env.example](frontend/.env.example) through your shell, `.env`-compatible tooling of choice, or hosting provider.
- Google Colab users should keep `DRY_RUN=true` and `ENABLE_SERVER=true` by default, flipping the flags only after manual verification.

## Operational Notes
- Smoke tests in [auto_stock_trader_agent.ipynb](auto_stock_trader_agent.ipynb) validate JSON parsing and combiner math; rerun them after model updates.
- [combiner/combine.py](combiner/combine.py) prints accuracy each hundred epochs and overwrites weights in [models/](models/); archive old PT files before retraining if you need history.
- [frontend/app.py](frontend/app.py) exposes `/api/test` to validate Alpaca credentials and connection status.

## Contribution & Licensing
- **How to contribute:** Fork the repository, create a feature branch off `main`, run notebook smoke tests (`run_smoke_tests()` in [auto_stock_trader_agent.ipynb](auto_stock_trader_agent.ipynb)), ensure `uv sync` succeeds without lockfile changes unless dependencies are updated, and open a pull request describing testing evidence.
- **Code of conduct:** Follow respectful collaboration norms; seek review before merging, and avoid committing secrets (use [.env.example](.env.example) patterns).
- **License:** Apache License 2.0 — contributions are accepted under the same terms.

## Roadmap
- Expand FastAPI endpoints to stream intermediate vectors to the dashboard.
- Automate notebook execution checks in CI via `uv run jupyter nbconvert`.
- Explore Dockerizing the notebook server plus frontend for single-command deployment.
- Integrate sentiment features and additional tickers into the LLM dataset generator.
