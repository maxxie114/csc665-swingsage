# SwingSage Trading Dashboard

A sleek trading dashboard built with Flask, connecting to Alpaca's paper trading API and displaying AI model predictions.

## Features

- Real-time portfolio tracking
- AI prediction display (LLM + LSTM combined)
- Quick trade execution
- Position and order monitoring
- AMOLED black theme with purple and neon green accents

## Setup

### 1. Install Dependencies

```bash
cd frontend
pip install -r requirements.txt
```

Or with uv:
```bash
uv pip install -r requirements.txt
```

### 2. Configure Alpaca API Keys

Add the following to your `.env` file in the project root:

```
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Get your free paper trading API keys from [Alpaca](https://alpaca.markets/).

### 3. Run the Dashboard

```bash
python app.py
```

Or with uv:
```bash
uv run python app.py
```

Open http://localhost:5000 in your browser.

## Project Structure

```
frontend/
  app.py              # Flask backend
  requirements.txt    # Python dependencies
  templates/
    index.html        # Main dashboard template
  static/
    style.css         # AMOLED black theme
    script.js         # Frontend JavaScript
```

## Design

The dashboard uses:
- AMOLED Black (#000000) as the primary background
- Purple (#9945FF) as the accent color
- Robinhood Neon Green (#00DC82) for positive values and buy signals
- Red (#FF3B5C) for negative values and sell signals
- DM Sans for clean, modern typography
- JetBrains Mono for numerical data

## API Endpoints

- GET /api/account - Get account info
- GET /api/positions - Get all positions
- GET /api/orders - Get recent orders
- GET /api/history - Get portfolio history
- GET /api/quote/SYMBOL - Get latest quote
- GET /api/prediction/SYMBOL - Get AI prediction
- POST /api/trade - Place a trade

