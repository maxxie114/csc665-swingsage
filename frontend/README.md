# SwingSage Trading Dashboard

The SwingSage frontend is a Flask application that proxies Alpaca paper trading data, surfaces blended LLM/LSTM predictions, and now enforces finance-only guardrails via the OpenRouter Llama Guard model.

## Highlights
- Live Alpaca account, position, order, and quote views.
- Combined DeepSeek + LSTM insights rendered inline with each chat reply.
- Paper-trade confirmations with optional auto-trade hooks.
- Finance-scoped guardrail that rejects off-topic chat prompts while allowing trades, portfolio lookups, and crypto questions.

## Prerequisites
- Python 3.12 (pyenv honors the project root `.python-version`).
- [`uv`](https://docs.astral.sh/uv/) for dependency management (recommended) or `pip`.
- Alpaca paper trading API keys.
- OpenRouter API key with access to `deepseek/deepseek-chat-v3.1` and `meta-llama/llama-guard-4-12b`.
- URL for the FastAPI agent (local notebook or tunnel) supplying `get_comprehensive_stock_data`.

## Setup

### 1. Install dependencies
```bash
cd frontend
uv sync
```
> Prefer `uv sync`; fall back to `pip install -r requirements.txt` only if uv is unavailable.

### 2. Configure environment variables
Copy the template and fill each value:
```bash
cp .env.example .env
```

Key fields:
- `OPENROUTER_API_KEY`: OpenRouter token (required for chat + guard).
- `OPENROUTER_MODEL`: main assistant model (`deepseek/deepseek-chat-v3.1` by default).
- `OPENROUTER_GUARD_MODEL`: safety firewall (`meta-llama/llama-guard-4-12b`).
- `ENABLE_CHAT_GUARD`: `true` to keep the guard active (recommended).
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`: Alpaca paper credentials.
- `FASTAPI_SERVER_URL`: URL of the notebook-hosted agent (`http://localhost:8000` or tunnel).

### 3. Launch the server
```bash
uv run python app.py
```

If uv is unavailable:
```bash
python app.py
```

Visit http://localhost:5000 once the server reports `Running on http://127.0.0.1:5000/`.

## Guardrail Behavior
- Requests unrelated to stocks, equities, ETFs, options, or crypto are blocked with category-specific feedback (e.g., `S6 (Specialized Advice)`).
- In-scope prompts such as “buy 1 share of NVDA” and “current BTC quote” are explicitly allowed even when the guard flags Specialized Advice.
- Frontend chat displays guard denials in-line with a red banner so users understand why input was rejected.

## Endpoint Reference
- `GET /api/test` – sanity-check Alpaca credentials.
- `GET /api/account` – fetch account summary.
- `GET /api/positions` – list open positions.
- `GET /api/orders` – recent order history.
- `GET /api/history` – portfolio performance series.
- `GET /api/quote/<symbol>` – latest quote; falls back to trades when data API is unavailable.
- `GET /api/prediction/<symbol>` – combined AI forecast (LLM + LSTM + combiner).
- `POST /api/chat` – guarded natural-language assistant.
- `POST /api/chat/confirm` – execute staged trades (paper only).

## Frontend Layout
```
frontend/
├── app.py              # Flask API + guardrail orchestration
├── requirements.txt    # Python dependencies for uv/pip
├── templates/
│   └── index.html      # Single-page dashboard
└── static/
    ├── script.js       # UI logic, chat client, charts
    └── style.css       # AMOLED-inspired theme
```

## Styling Notes
- Primary background: `#000000` (AMOLED black).
- Accent: `#C4F000` (SwingSage neon green) with orange-red for warnings.
- Fonts: system sans-serif body, JetBrains Mono for numerics/code.
- Guard denials render with a red border for immediate visibility.

## Troubleshooting
- `403` with guard message: prompt was out of scope; adjust to finance/crypto topics.
- `403` with “Guardrail service unavailable”: verify `OPENROUTER_API_KEY`, guard model access, and network connectivity.
- “AI assistant is not available”: ensure the FastAPI agent is running and `FASTAPI_SERVER_URL` points to it.

