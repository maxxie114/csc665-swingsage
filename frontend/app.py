"""
SwingSage Trading Dashboard Backend
Connects to Alpaca Paper Trading API and serves predictions
"""

import os
import json
import re
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Try loading .env from multiple possible locations
env_paths = ['../.env', '.env', os.path.join(os.path.dirname(__file__), '..', '.env')]
for path in env_paths:
    if os.path.exists(path):
        load_dotenv(path)
        break
else:
    load_dotenv()  # Try default location

app = Flask(__name__)
CORS(app)

# Alpaca API Configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# OpenRouter API Configuration for Kimi K2 Chat
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
KIMI_MODEL = "moonshotai/kimi-k2-0905"  # Kimi K2 model on OpenRouter

# Debug: Print if keys are loaded (without showing full key)
if ALPACA_API_KEY:
    print(f"✓ Alpaca API Key loaded: {ALPACA_API_KEY[:10]}...")
else:
    print("✗ Alpaca API Key NOT loaded")
if ALPACA_SECRET_KEY:
    print(f"✓ Alpaca Secret Key loaded: {ALPACA_SECRET_KEY[:10]}...")
else:
    print("✗ Alpaca Secret Key NOT loaded")
if OPENROUTER_API_KEY:
    print(f"✓ OpenRouter API Key loaded: {OPENROUTER_API_KEY[:10]}...")
else:
    print("✗ OpenRouter API Key NOT loaded")

HEADERS = {
    'APCA-API-KEY-ID': ALPACA_API_KEY,
    'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
    'Content-Type': 'application/json'
}


def validate_alpaca_config():
    """Validate Alpaca API configuration"""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return {'error': 'Alpaca API keys not configured. Please add ALPACA_API_KEY and ALPACA_SECRET_KEY to .env file'}
    return None


def alpaca_request(endpoint, method='GET', data=None):
    """Make a request to Alpaca API"""
    # Check config first
    config_error = validate_alpaca_config()
    if config_error:
        return config_error
    
    url = f"{ALPACA_BASE_URL}{endpoint}"
    try:
        if method == 'GET':
            response = requests.get(url, headers=HEADERS, timeout=10)
        elif method == 'POST':
            response = requests.post(url, headers=HEADERS, json=data, timeout=10)
        elif method == 'DELETE':
            response = requests.delete(url, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            error_text = response.text
            try:
                error_json = response.json()
                error_text = error_json.get('message', error_text)
            except:
                pass
            return {'error': error_text, 'status': response.status_code}
    except requests.exceptions.Timeout:
        return {'error': 'Request timeout - Alpaca API not responding'}
    except requests.exceptions.ConnectionError:
        return {'error': 'Connection error - Cannot reach Alpaca API'}
    except Exception as e:
        return {'error': f'API Error: {str(e)}'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/test')
def test_alpaca():
    """Test Alpaca API connection"""
    config_error = validate_alpaca_config()
    if config_error:
        return jsonify(config_error), 400
    
    # Try to get account to verify connection
    account = alpaca_request('/v2/account')
    if 'error' in account:
        return jsonify({
            'connected': False,
            'error': account['error'],
            'status': account.get('status', 'unknown')
        }), 400
    
    return jsonify({
        'connected': True,
        'account_id': account.get('account_number', 'N/A'),
        'status': account.get('status', 'N/A')
    })


@app.route('/api/account')
def get_account():
    """Get account information"""
    account = alpaca_request('/v2/account')
    if 'error' in account:
        return jsonify(account), 400
    return jsonify(account)


@app.route('/api/positions')
def get_positions():
    """Get all positions"""
    positions = alpaca_request('/v2/positions')
    if 'error' in positions:
        return jsonify(positions), 400
    # Ensure it's always an array
    if not isinstance(positions, list):
        return jsonify([])
    return jsonify(positions)


@app.route('/api/orders')
def get_orders():
    """Get recent orders"""
    orders = alpaca_request('/v2/orders?status=all&limit=20')
    if 'error' in orders:
        return jsonify(orders), 400
    # Ensure it's always an array
    if not isinstance(orders, list):
        return jsonify([])
    return jsonify(orders)


@app.route('/api/history')
def get_history():
    """Get portfolio history"""
    end = datetime.now()
    start = end - timedelta(days=30)
    history = alpaca_request(f'/v2/account/portfolio/history?period=1M&timeframe=1D')
    return jsonify(history)


@app.route('/api/quote/<symbol>')
def get_quote(symbol):
    """Get latest quote for a symbol"""
    config_error = validate_alpaca_config()
    if config_error:
        return jsonify(config_error), 400
    
    # Ensure symbol is uppercase
    symbol = symbol.upper()
    
    # Try Market Data API first
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            # If Market Data API fails, try Trading API's latest trade endpoint
            print(f"Market Data API returned {response.status_code}: {response.text}")
            # Fallback to latest trade from Trading API
            trade_url = f"{ALPACA_BASE_URL}/v2/stocks/{symbol}/trades/latest"
            trade_response = requests.get(trade_url, headers=HEADERS, timeout=10)
            if trade_response.status_code == 200:
                trade_data = trade_response.json()
                # Format as quote-like response
                return jsonify({
                    'quote': {
                        'ap': trade_data.get('trade', {}).get('p', 0),
                        'bp': trade_data.get('trade', {}).get('p', 0),
                    },
                    'symbol': symbol
                })
            else:
                error_text = response.text
                try:
                    error_json = response.json()
                    error_text = error_json.get('message', error_text)
                except:
                    pass
                return jsonify({'error': f'Market Data API: {error_text}', 'status': response.status_code}), response.status_code
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timeout'}), 408
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Connection error'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bars/<symbol>')
def get_bars(symbol):
    """Get historical bars for a symbol"""
    end = datetime.now()
    start = end - timedelta(days=30)
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars?timeframe=1Day&start={start.strftime('%Y-%m-%d')}&end={end.strftime('%Y-%m-%d')}"
    try:
        response = requests.get(url, headers=HEADERS)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/trade', methods=['POST'])
def place_trade():
    """Place a trade"""
    data = request.json
    order = {
        'symbol': data.get('symbol', 'AAPL'),
        'qty': data.get('qty', 1),
        'side': data.get('side', 'buy'),
        'type': 'market',
        'time_in_force': 'day'
    }
    result = alpaca_request('/v2/orders', method='POST', data=order)
    return jsonify(result)


# Stock name to ticker mapping (case-insensitive matching)
STOCK_NAMES = {
    'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
    'amazon': 'AMZN', 'meta': 'META', 'facebook': 'META', 'nvidia': 'NVDA', 'netflix': 'NFLX',
    'disney': 'DIS', 'coca cola': 'KO', 'coke': 'KO', 'pepsi': 'PEP', 'walmart': 'WMT',
    'jpmorgan': 'JPM', 'jp morgan': 'JPM', 'bank of america': 'BAC', 'bofa': 'BAC',
    'visa': 'V', 'mastercard': 'MA', 'master card': 'MA',
    'intel': 'INTC', 'amd': 'AMD', 'oracle': 'ORCL', 'salesforce': 'CRM',
    'uber': 'UBER', 'lyft': 'LYFT', 'spotify': 'SPOT', 'twitter': 'TWTR', 'x': 'TWTR',
    'snapchat': 'SNAP', 'snap': 'SNAP', 'zoom': 'ZM', 'palantir': 'PLTR',
    'goldman sachs': 'GS', 'goldman': 'GS', 'morgan stanley': 'MS', 'morgan': 'MS',
    'boeing': 'BA', 'general motors': 'GM', 'gm': 'GM', 'ford': 'F',
    'exxon': 'XOM', 'chevron': 'CVX', 'shell': 'SHEL', 'bp': 'BP'
}


def parse_natural_language(message):
    """
    Parse natural language message to extract trading intent or information query.
    Returns: {
        'symbol': str or None,
        'action': 'buy' or 'sell' or None,
        'quantity': int (default 1),
        'query_type': 'trade' or 'info' or None,
        'info_type': 'price' or 'close' or '52week' or 'high' or 'low' or None
    }
    """
    message_lower = message.lower()
    
    # Detect information queries
    query_type = None
    info_type = None
    
    price_patterns = ['current price', 'price', 'what is the price', 'how much is', 'cost']
    close_patterns = ['yesterday', 'yesterday\'s close', 'previous close', 'last close', 'closed at']
    high_patterns = ['52 week high', '52-week high', 'year high', 'all time high', 'highest']
    low_patterns = ['52 week low', '52-week low', 'year low', 'all time low', 'lowest']
    
    if any(pattern in message_lower for pattern in price_patterns):
        query_type = 'info'
        info_type = 'price'
    elif any(pattern in message_lower for pattern in close_patterns):
        query_type = 'info'
        info_type = 'close'
    elif any(pattern in message_lower for pattern in high_patterns):
        query_type = 'info'
        info_type = '52week_high'
    elif any(pattern in message_lower for pattern in low_patterns):
        query_type = 'info'
        info_type = '52week_low'
    elif any(word in message_lower for word in ['what', 'tell me', 'show me', 'information', 'data', 'stats']):
        query_type = 'info'
        info_type = 'general'
    
    # Extract quantity (look for numbers before "share", "stock", "shares", "stocks")
    quantity = 1
    qty_patterns = [
        r'(\d+)\s*(?:share|stock|shares|stocks)',
        r'(?:share|stock|shares|stocks)\s*(?:of\s*)?(\d+)',
        r'(\d+)\s*(?:of\s*)?(?:share|stock|shares|stocks)'
    ]
    for pattern in qty_patterns:
        match = re.search(pattern, message_lower)
        if match:
            try:
                quantity = int(match.group(1))
                break
            except:
                pass
    
    # Extract action (buy or sell) - only if not an info query
    action = None
    if query_type != 'info':
        buy_patterns = ['buy', 'purchase', 'get', 'acquire', 'want to buy', 'plan to buy', 'should buy']
        sell_patterns = ['sell', 'dump', 'get rid of', 'want to sell', 'plan to sell', 'should sell']
        
        for pattern in buy_patterns:
            if pattern in message_lower:
                action = 'buy'
                query_type = 'trade'
                break
        
        if action is None:
            for pattern in sell_patterns:
                if pattern in message_lower:
                    action = 'sell'
                    query_type = 'trade'
                    break
    
    # Extract symbol
    symbol = None
    
    # First, try stock names (more reliable than ticker regex)
    for name, ticker in STOCK_NAMES.items():
        if name in message_lower:
            symbol = ticker
            break
    
    # If no stock name found, try to find ticker symbols (uppercase 1-5 letter codes)
    if symbol is None:
        # Common words to exclude (expanded list)
        common_words = {
            'I', 'A', 'THE', 'TO', 'OF', 'AND', 'FOR', 'IN', 'ON', 'AT', 'BY', 'AS', 'IS',
            'IT', 'BE', 'OR', 'AN', 'MY', 'ME', 'WE', 'HE', 'SHE', 'HIS', 'HER', 'OUR',
            'THEY', 'THEM', 'THIS', 'THAT', 'THESE', 'THOSE', 'WHAT', 'WHEN', 'WHERE',
            'WHY', 'HOW', 'WHO', 'WHICH', 'CAN', 'WILL', 'WOULD', 'SHOULD', 'COULD',
            'MAY', 'MIGHT', 'MUST', 'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID', 'GET',
            'GOT', 'BUY', 'SELL', 'SHARE', 'SHARES', 'STOCK', 'STOCKS', 'PRICE', 'COST'
        }
        
        # Find all potential tickers
        ticker_matches = re.findall(r'\b([A-Z]{1,5})\b', message.upper())
        for potential_ticker in ticker_matches:
            # Validate it's not a common word and looks like a real ticker
            if potential_ticker not in common_words and len(potential_ticker) >= 2:
                # Prefer longer tickers (more likely to be real)
                if symbol is None or len(potential_ticker) > len(symbol):
                    symbol = potential_ticker
    
    return {
        'symbol': symbol,
        'action': action,
        'quantity': quantity,
        'query_type': query_type,
        'info_type': info_type
    }


@app.route('/api/chat', methods=['POST'])
def chat_assistant():
    """AI chat assistant with function calling - LLM decides when to use tools"""
    data = request.json
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'Message is required'}), 400
    
    try:
        kimi_client = get_kimi_client()
        if not kimi_client:
            return jsonify({
                'type': 'error',
                'message': 'AI assistant is not available. Please check API configuration.'
            }), 500
        
        system_prompt = """You are a helpful AI trading assistant for SwingSage. You help users with:
- Stock information queries (prices, 52-week highs/lows, volume, etc.)
- Trading decisions and analysis using AI predictions
- Executing trades (buy/sell orders)

You have access to a comprehensive tool: get_comprehensive_stock_data
This tool gives you ALL available information about a stock in one call:
- Current price (latest trade)
- Yesterday's close
- 52-week high/low
- Trading volume
- AI predictions (LLM, LSTM, combined signals, confidence)
- Current bid/ask prices

**IMPORTANT**: When a user asks about a stock:
1. ALWAYS use get_comprehensive_stock_data first - it gives you everything you need
2. Based on the user's question, decide what information to present:
   - If they ask "what is the price?" → return the current_price
   - If they ask "what do you think?" → return predictions and analysis
   - If they ask "tell me about X" → return relevant info based on context
3. Be smart about what to show - only present what's relevant to their question
4. For stock names like "Apple", "Tesla", "Microsoft", convert them to tickers (AAPL, TSLA, MSFT) before calling tools

Be concise, friendly, and helpful. Use the comprehensive data to give intelligent, context-aware responses."""
        
        # Use LangChain message format
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        # Initial LLM call
        try:
            response = kimi_client.invoke(messages)
        except Exception as e:
            error_msg = str(e)
            print(f"LLM invocation error: {error_msg}")
            
            # Check if it's an authentication error
            if "401" in error_msg or "User not found" in error_msg or "unauthorized" in error_msg.lower():
                # Try without tools first to test basic API connectivity
                print("Authentication error detected. Testing API key without tools...")
                try:
                    client_no_tools = get_kimi_client(with_tools=False)
                    if client_no_tools:
                        test_response = client_no_tools.invoke([
                            SystemMessage(content="You are a helpful assistant."),
                            HumanMessage(content="Say hello")
                        ])
                        print("✓ API key works without tools")
                        # If it works without tools, the issue is with function calling
                        return jsonify({
                            'type': 'error',
                            'message': 'API key is valid but function calling may not be supported for this model. Please try a simpler query or check OpenRouter documentation.'
                        }), 500
                except Exception as e2:
                    print(f"✗ API key test failed: {e2}")
                    return jsonify({
                        'type': 'error',
                        'message': f'Invalid API key or authentication failed. Please check your OPENROUTER_API_KEY in the .env file. Error: {str(e2)}'
                    }), 401
            
            return jsonify({
                'type': 'error',
                'message': f"Error communicating with AI assistant: {error_msg}"
            }), 500
        
        
        # Handle tool calls
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check if response has tool calls
            tool_calls = getattr(response, 'tool_calls', []) or []
            
            if not tool_calls:
                break
            
            print(f"🔧 Tool calls detected: {len(tool_calls)}")
            
            # Add AI response to messages
            messages.append(response)
            
            # Execute tool calls
            for tool_call in tool_calls:
                # Handle different tool call formats
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('args', {})
                    tool_call_id = tool_call.get('id', f"call_{tool_name}_{iteration}")
                else:
                    # Handle object format
                    tool_name = getattr(tool_call, 'name', '')
                    tool_args = getattr(tool_call, 'args', {})
                    tool_call_id = getattr(tool_call, 'id', f"call_{tool_name}_{iteration}")
                
                try:
                    print(f"🔧 Executing tool: {tool_name} with args: {tool_args}")
                    if tool_name == 'get_comprehensive_stock_data':
                        result = get_comprehensive_stock_data.invoke(tool_args)
                        print(f"✅ Tool result: {result}")
                    elif tool_name == 'get_stock_info_tool' or tool_name == 'get_stock_info':
                        result = get_stock_info_tool.invoke(tool_args)
                        print(f"✅ Tool result: {result}")
                    elif tool_name == 'get_prediction_tool' or tool_name == 'get_prediction':
                        result = get_prediction_tool.invoke(tool_args)
                        print(f"✅ Tool result: {result}")
                    elif tool_name == 'get_current_price_tool' or tool_name == 'get_current_price':
                        result = get_current_price_tool.invoke(tool_args)
                        print(f"✅ Tool result: {result}")
                    elif tool_name == 'place_trade_tool':
                        # For trades, we need special handling - return confirmation request
                        symbol = tool_args.get('symbol', '').upper()
                        side = tool_args.get('side', '').lower()
                        quantity = tool_args.get('quantity', 1)
                        
                        # Get prediction and price for confirmation
                        prediction = get_prediction_data(symbol)
                        quote_data = get_quote_data(symbol)
                        price = 0
                        if quote_data and 'quote' in quote_data:
                            price = quote_data['quote'].get('ap', quote_data['quote'].get('bp', 0))
                        
                        return jsonify({
                            'type': 'trade_confirmation',
                            'message': f"I'm ready to {side} {quantity} share{'s' if quantity > 1 else ''} of {symbol}. Please confirm to proceed.",
                            'symbol': symbol,
                            'action': side,
                            'quantity': quantity,
                            'price': price,
                            'estimated_cost': price * quantity if price > 0 else 0,
                            'prediction': prediction
                        })
                    else:
                        result = {'error': f'Unknown tool: {tool_name}'}
                    
                    # Add tool result to messages
                    messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id))
                    
                except Exception as e:
                    print(f"Tool execution error: {e}")
                    messages.append(ToolMessage(
                        content=json.dumps({'error': str(e)}),
                        tool_call_id=tool_call_id
                    ))
            
            # Get next LLM response with tool results
            response = kimi_client.invoke(messages)
        
        # Final response - format based on content
        final_message = response.content if hasattr(response, 'content') else str(response)
        
        # Check if we have stock info or predictions in the conversation
        symbol = None
        prediction = None
        stock_info = None
        
        # Try to extract symbol and data from tool results
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    tool_result = json.loads(msg.content)
                    if 'symbol' in tool_result:
                        symbol = tool_result['symbol']
                    if 'llm' in tool_result or 'combined' in tool_result:
                        prediction = tool_result
                    if 'current_price' in tool_result and '52week_high' in tool_result:
                        stock_info = tool_result
                except:
                    pass
        
        # Determine response type
        if prediction and symbol:
            quote_data = get_quote_data(symbol) if symbol else None
            price = 0
            if quote_data and 'quote' in quote_data:
                price = quote_data['quote'].get('ap', quote_data['quote'].get('bp', 0))
            
            return jsonify({
                'type': 'prediction',
                'message': final_message,
                'symbol': symbol,
                'prediction': prediction,
                'price': price
            })
        elif stock_info and symbol:
            return jsonify({
                'type': 'stock_info',
                'message': final_message,
                'symbol': symbol,
                'info': stock_info,
                'info_type': 'general'
            })
        else:
            return jsonify({
                'type': 'text',
                'message': final_message
            })
            
    except Exception as e:
        print(f"Chat assistant error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'type': 'error',
            'message': f"Error: {str(e)}"
        }), 500


@app.route('/api/chat/confirm', methods=['POST'])
def confirm_trade():
    """Confirm and execute a trade from chat"""
    data = request.json
    symbol = data.get('symbol')
    action = data.get('action')  # 'buy' or 'sell'
    quantity = data.get('quantity', 1)
    
    if not symbol or not action:
        return jsonify({'error': 'Symbol and action are required'}), 400
    
    # Execute trade using internal function
    try:
        order = {
            'symbol': symbol,
            'qty': quantity,
            'side': action,
            'type': 'market',
            'time_in_force': 'day'
        }
        trade_result = alpaca_request('/v2/orders', method='POST', data=order)
        
        if 'error' in trade_result:
            return jsonify({
                'success': False,
                'message': f"Trade failed: {trade_result['error']}",
                'result': trade_result
            }), 400
        
        return jsonify({
            'success': True,
            'message': f"✅ Trade executed! {action.capitalize()} {quantity} share{'s' if quantity > 1 else ''} of {symbol}",
            'result': trade_result
        })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f"Error executing trade: {str(e)}"
        }), 500


def get_prediction_data(symbol):
    """Get model prediction data (helper function)"""
    import random
    
    # Simulated prediction based on our combiner model
    predictions = {
        'AAPL': {
            'llm': {'buy': 0.45, 'sell': 0.25, 'hold': 0.30, 'signal': 'BUY'},
            'lstm': {'buy': 0.20, 'sell': 0.60, 'hold': 0.20, 'signal': 'SELL'},
            'combined': {'buy': 0.28, 'sell': 0.50, 'hold': 0.22, 'signal': 'SELL'},
            'confidence': 0.50,
            'model_weights': {'llm': 0.14, 'lstm': 0.86}
        }
    }
    
    if symbol.upper() in predictions:
        return predictions[symbol.upper()]
    else:
        # Generate random prediction for other symbols
        signals = ['BUY', 'SELL', 'HOLD']
        signal = random.choice(signals)
        return {
            'llm': {'buy': 0.33, 'sell': 0.33, 'hold': 0.34, 'signal': signal},
            'lstm': {'buy': 0.33, 'sell': 0.33, 'hold': 0.34, 'signal': signal},
            'combined': {'buy': 0.33, 'sell': 0.33, 'hold': 0.34, 'signal': signal},
            'confidence': 0.33,
            'model_weights': {'llm': 0.14, 'lstm': 0.86}
        }


def get_quote_data(symbol):
    """Get quote data (helper function)"""
    config_error = validate_alpaca_config()
    if config_error:
        return None
    
    symbol = symbol.upper()
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            # Fallback to latest trade
            trade_url = f"{ALPACA_BASE_URL}/v2/stocks/{symbol}/trades/latest"
            trade_response = requests.get(trade_url, headers=HEADERS, timeout=10)
            if trade_response.status_code == 200:
                trade_data = trade_response.json()
                return {
                    'quote': {
                        'ap': trade_data.get('trade', {}).get('p', 0),
                        'bp': trade_data.get('trade', {}).get('p', 0),
                    },
                    'symbol': symbol
                }
    except:
        pass
    return None


def get_bars_data(symbol, days=365):
    """Get historical bars data (helper function)"""
    config_error = validate_alpaca_config()
    if config_error:
        return None
    
    symbol = symbol.upper()
    end = datetime.now()
    start = end - timedelta(days=days)
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars?timeframe=1Day&start={start.strftime('%Y-%m-%d')}&end={end.strftime('%Y-%m-%d')}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


# Tool definitions for LLM function calling
@tool
def get_comprehensive_stock_data(symbol: str) -> dict:
    """Get ALL available stock information including current price, historical data, AI predictions, and trading signals.
    Use this tool whenever the user asks about a stock - it provides complete context for you to answer any question.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')
    
    Returns:
        Dictionary with comprehensive stock data including:
        - current_price: Latest trade price
        - yesterday_close: Previous day's closing price
        - 52week_high: 52-week high price
        - 52week_low: 52-week low price
        - volume: Trading volume
        - predictions: AI model predictions (LLM, LSTM, combined signals, confidence)
        - quote_data: Current bid/ask prices
    """
    symbol = symbol.upper()
    
    # Get all available data
    stock_info = get_stock_info(symbol)
    prediction = get_prediction_data(symbol)
    quote_data = get_quote_data(symbol)
    
    # Get latest trade price (most accurate current price)
    latest_trade_price = 0
    config_error = validate_alpaca_config()
    if not config_error:
        try:
            trade_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
            response = requests.get(trade_url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                trade_data = response.json()
                if 'trade' in trade_data and 'p' in trade_data['trade']:
                    latest_trade_price = trade_data['trade']['p']
        except:
            pass
    
    # Use latest trade price if available, otherwise use from stock_info
    if latest_trade_price > 0:
        stock_info['current_price'] = latest_trade_price
    
    # Combine all data
    comprehensive_data = {
        'symbol': symbol,
        'current_price': stock_info.get('current_price', 0),
        'yesterday_close': stock_info.get('yesterday_close', 0),
        '52week_high': stock_info.get('52week_high', 0),
        '52week_low': stock_info.get('52week_low', 0),
        'volume': stock_info.get('volume', 0),
        'predictions': prediction,
        'quote': quote_data.get('quote', {}) if quote_data else {}
    }
    
    return comprehensive_data


# Keep individual tools for backward compatibility, but LLM should prefer comprehensive tool
@tool
def get_stock_info_tool(symbol: str) -> dict:
    """Get comprehensive stock information including current price, yesterday's close, 52-week high/low, and volume.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')
    
    Returns:
        Dictionary with stock information including current_price, yesterday_close, 52week_high, 52week_low, and volume
    """
    return get_stock_info(symbol)


@tool
def get_prediction_tool(symbol: str) -> dict:
    """Get AI model predictions for a stock including LLM, LSTM, and combined signals with confidence scores.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')
    
    Returns:
        Dictionary with prediction data including llm, lstm, combined signals, and confidence
    """
    return get_prediction_data(symbol)


@tool
def get_current_price_tool(symbol: str) -> dict:
    """Get the current market price for a stock using the latest trade price.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')
    
    Returns:
        Dictionary with current price information
    """
    symbol = symbol.upper()
    
    # Try to get latest trade price first (most accurate)
    config_error = validate_alpaca_config()
    if not config_error:
        try:
            trade_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
            response = requests.get(trade_url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                trade_data = response.json()
                if 'trade' in trade_data and 'p' in trade_data['trade']:
                    price = trade_data['trade']['p']
                    return {
                        'symbol': symbol,
                        'current_price': price,
                        'price_available': True,
                        'source': 'latest_trade'
                    }
        except:
            pass
    
    # Fallback to quote data (ask/bid price)
    quote_data = get_quote_data(symbol)
    price = 0
    if quote_data and 'quote' in quote_data:
        quote = quote_data['quote']
        # Use ask price, or bid price, or average if both available
        ask_price = quote.get('ap', 0)
        bid_price = quote.get('bp', 0)
        if ask_price > 0 and bid_price > 0:
            price = (ask_price + bid_price) / 2  # Midpoint
        elif ask_price > 0:
            price = ask_price
        elif bid_price > 0:
            price = bid_price
    
    return {
        'symbol': symbol,
        'current_price': price,
        'price_available': price > 0,
        'source': 'quote' if price > 0 else 'unavailable'
    }


@tool
def place_trade_tool(symbol: str, side: str, quantity: int) -> dict:
    """Place a trade order (buy or sell) for a stock.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')
        side: 'buy' or 'sell'
        quantity: Number of shares to trade
    
    Returns:
        Dictionary with trade execution result
    """
    order = {
        'symbol': symbol.upper(),
        'qty': quantity,
        'side': side.lower(),
        'type': 'market',
        'time_in_force': 'day'
    }
    result = alpaca_request('/v2/orders', method='POST', data=order)
    return result


def get_openrouter_tools():
    """Get tools in OpenRouter format"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_stock_info",
                "description": "Get comprehensive stock information including current price, yesterday's close, 52-week high/low, and volume",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_prediction",
                "description": "Get AI model predictions for a stock including LLM, LSTM, and combined signals with confidence scores",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_price",
                "description": "Get the current market price for a stock",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT')"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }
    ]


def call_openrouter_api(messages, tools=None):
    """Call OpenRouter API directly"""
    if not OPENROUTER_API_KEY:
        return None
    
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://swingsage.local",
        "X-Title": "SwingSage Trading Assistant"
    }
    
    payload = {
        "model": KIMI_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    if tools:
        payload["tools"] = tools
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"OpenRouter API HTTP error: {e}")
        if e.response:
            print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"OpenRouter API error: {e}")
        return None


def get_kimi_client(with_tools=True):
    """Get Kimi K2 chat client using OpenRouter LangChain integration
    Following exact format from: https://openrouter.ai/docs/guides/community/langchain
    """
    # Reload environment to ensure we have the latest API key
    from dotenv import load_dotenv
    env_paths = ['../.env', '.env', os.path.join(os.path.dirname(__file__), '..', '.env')]
    for path in env_paths:
        if os.path.exists(path):
            load_dotenv(path, override=True)
            break
    
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    
    if not api_key:
        print("✗ OpenRouter API Key is missing")
        return None
    
    print(f"✓ Using OpenRouter API Key: {api_key[:10]}...")
    
    # Create tools list - comprehensive tool first so LLM prefers it
    tools = [get_comprehensive_stock_data, get_stock_info_tool, get_prediction_tool, get_current_price_tool, place_trade_tool]
    
    # Use OpenRouter's recommended LangChain configuration
    # See: https://openrouter.ai/docs/guides/community/langchain
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=KIMI_MODEL,
            temperature=0.7,
            max_tokens=1000,
            default_headers={
                "HTTP-Referer": "https://swingsage.local",  # Optional: for OpenRouter analytics
                "X-Title": "SwingSage Trading Assistant"  # Optional: for OpenRouter analytics
            }
        )
        
        # Bind tools to the client if requested
        if with_tools:
            try:
                return llm.bind_tools(tools)
            except Exception as e:
                print(f"Warning: Could not bind tools: {e}")
                print("Falling back to client without tools")
                return llm
        
        return llm
    except Exception as e:
        print(f"Error creating ChatOpenAI client: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_stock_info(symbol):
    """Get comprehensive stock information"""
    info = {
        'symbol': symbol.upper(),
        'current_price': 0,
        'yesterday_close': 0,
        '52week_high': 0,
        '52week_low': 0,
        'volume': 0
    }
    
    # Get current quote
    quote_data = get_quote_data(symbol)
    if quote_data and 'quote' in quote_data:
        quote = quote_data['quote']
        info['current_price'] = quote.get('ap', quote.get('bp', 0))
    
    # Get bars for historical data
    bars_data = get_bars_data(symbol, days=365)
    if bars_data and 'bars' in bars_data:
        bars = bars_data['bars']
        
        # Handle different response formats
        if isinstance(bars, dict):
            # Format: {'AAPL': [...]}
            if symbol in bars:
                bars = bars[symbol]
            else:
                # Try with any key
                bars = list(bars.values())[0] if bars else []
        elif isinstance(bars, list):
            # Format: [...]
            bars = bars
        else:
            bars = []
        
        if bars and len(bars) > 0:
            # Yesterday's close (second to last bar, or last if market is closed)
            # Sort by timestamp if needed
            if len(bars) >= 2:
                yesterday_bar = bars[-2] if len(bars) > 1 else bars[-1]
                info['yesterday_close'] = yesterday_bar.get('c', yesterday_bar.get('close', 0))
            elif len(bars) == 1:
                info['yesterday_close'] = bars[0].get('c', bars[0].get('close', 0))
            
            # 52-week high and low
            highs = []
            lows = []
            for bar in bars:
                high = bar.get('h', bar.get('high', 0))
                low = bar.get('l', bar.get('low', 0))
                if high > 0:
                    highs.append(high)
                if low > 0:
                    lows.append(low)
            
            if highs:
                info['52week_high'] = max(highs)
            if lows:
                info['52week_low'] = min(lows)
            
            # Latest volume
            if bars:
                info['volume'] = bars[-1].get('v', bars[-1].get('volume', 0))
    
    return info


@app.route('/api/prediction/<symbol>')
def get_prediction(symbol):
    """Get model prediction for a symbol"""
    prediction = get_prediction_data(symbol)
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='127.0.0.1')

