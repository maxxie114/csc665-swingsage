// SwingSage Trading Dashboard
// Frontend JavaScript

const API_BASE = '';

// State
let currentSymbol = 'AAPL';
let tradeSide = 'buy';
let predictionMode = 'view';
let portfolioChart = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    setupEventListeners();
    setupPredictionControls();
    setupChatListeners();
    await testAlpacaConnection();
    await refreshData();
}

async function testAlpacaConnection() {
    try {
        const response = await fetch(`${API_BASE}/api/test`);
        const result = await response.json();
        
        if (!result.connected) {
            console.error('Alpaca connection failed:', result.error);
            showConnectionError(result.error);
        } else {
            console.log('Alpaca connected successfully:', result);
            hideConnectionError();
        }
    } catch (error) {
        console.error('Error testing Alpaca:', error);
        showConnectionError(`Network error: ${error.message}`);
    }
}

function showConnectionError(message) {
    // Create or update error banner
    let errorBanner = document.getElementById('alpacaErrorBanner');
    if (!errorBanner) {
        errorBanner = document.createElement('div');
        errorBanner.id = 'alpacaErrorBanner';
        errorBanner.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: #FF5000;
            color: white;
            padding: 16px;
            text-align: center;
            z-index: 1000;
            font-weight: 600;
            font-size: 14px;
        `;
        document.body.prepend(errorBanner);
    }
    errorBanner.textContent = `⚠️ Alpaca API Error: ${message}`;
}

function hideConnectionError() {
    const errorBanner = document.getElementById('alpacaErrorBanner');
    if (errorBanner) {
        errorBanner.remove();
    }
}

function switchView(view) {
    // Hide all views
    document.querySelectorAll('.view-content').forEach(v => {
        v.style.display = 'none';
    });
    
    // Show selected view
    const viewElement = document.getElementById(`${view}View`);
    if (viewElement) {
        viewElement.style.display = 'block';
    }
    
    // Update page title
    const titles = {
        'dashboard': { title: 'Dashboard', subtitle: 'Account Overview' },
        'portfolio': { title: 'Portfolio', subtitle: 'Stocks & Trading' },
        'predictions': { title: 'Predictions', subtitle: 'AI Model Analysis' }
    };
    
    if (titles[view]) {
        document.getElementById('pageTitle').textContent = titles[view].title;
        document.getElementById('pageSubtitle').textContent = titles[view].subtitle;
    }
}

function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const view = item.dataset.view;
            switchView(view);
            
            // Update active nav item
            document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
        });
    });

    // Symbol search
    const searchInput = document.getElementById('symbolSearch');
    searchInput.addEventListener('keyup', (e) => {
        if (e.key === 'Enter') {
            currentSymbol = searchInput.value.toUpperCase();
            document.getElementById('tradeSymbol').value = currentSymbol;
            refreshData();
        }
    });

    // Trade tabs
    document.querySelectorAll('.trade-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.trade-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            tradeSide = tab.dataset.side;
            updateTradeButton();
        });
    });

    // Trade symbol input
    document.getElementById('tradeSymbol').addEventListener('input', (e) => {
        currentSymbol = e.target.value.toUpperCase();
        updateTradeButton();
    });

    // Quantity input
    document.getElementById('tradeQty').addEventListener('input', updateEstimatedCost);
}

function setupPredictionControls() {
    const modeButtons = document.querySelectorAll('[data-prediction-mode]');
    const quantityGroup = document.getElementById('autoTradeQuantityGroup');
    const runButton = document.getElementById('runPredictionBtn');

    modeButtons.forEach(button => {
        button.addEventListener('click', () => {
            predictionMode = button.dataset.predictionMode;
            modeButtons.forEach(btn => btn.classList.toggle('active', btn === button));

            if (quantityGroup) {
                quantityGroup.style.display = predictionMode === 'trade' ? 'flex' : 'none';
            }
        });
    });

    if (runButton) {
        runButton.addEventListener('click', async () => {
            const qtyInput = document.getElementById('autoTradeQty');
            const quantity = qtyInput ? Math.max(parseInt(qtyInput.value, 10) || 1, 1) : 1;

            await fetchPrediction(currentSymbol, {
                autoTrade: predictionMode === 'trade',
                quantity
            });
        });
    }
}

async function refreshData() {
    showLoading();
    await Promise.all([
        fetchAccount(),
        fetchPositions(),
        fetchOrders(),
        fetchPrediction(currentSymbol, { autoTrade: false }),
        fetchQuote(currentSymbol),
        fetchPortfolioHistory()
    ]);
    hideLoading();
}

let currentPrice = 0;

async function fetchQuote(symbol) {
    try {
        const response = await fetch(`${API_BASE}/api/quote/${symbol}`);
        const quote = await response.json();
        
        if (quote.error) {
            console.error('Quote error:', quote.error);
            return;
        }
        
        // Extract price from Alpaca quote response
        if (quote.quotes && quote.quotes[symbol]) {
            const quoteData = quote.quotes[symbol];
            currentPrice = parseFloat(quoteData.ap) || parseFloat(quoteData.bp) || 0;
        } else if (quote.quote) {
            currentPrice = parseFloat(quote.quote.ap) || parseFloat(quote.quote.bp) || 0;
        }
        
        if (currentPrice > 0) {
            document.getElementById('currentPrice').textContent = formatCurrency(currentPrice);
            updateEstimatedCost();
        } else {
            document.getElementById('currentPrice').textContent = 'Loading...';
        }
    } catch (error) {
        console.error('Error fetching quote:', error);
    }
}

async function fetchAccount() {
    try {
        const response = await fetch(`${API_BASE}/api/account`);
        const account = await response.json();
        
        if (account.error) {
            console.error('Account error:', account.error);
            showError('account', account.error);
            return;
        }
        
        const equity = parseFloat(account.equity);
        const lastEquity = parseFloat(account.last_equity);
        const change = equity - lastEquity;
        const changePercent = (change / lastEquity) * 100;
        
        document.getElementById('portfolioValue').textContent = formatCurrency(equity);
        document.getElementById('portfolioChange').textContent = formatPercent(changePercent);
        document.getElementById('portfolioChange').className = `stat-change ${changePercent >= 0 ? 'positive' : 'negative'}`;
        
        document.getElementById('todayPL').textContent = formatCurrency(change);
        document.getElementById('todayPL').className = change >= 0 ? '' : 'negative';
        document.getElementById('todayPLPercent').textContent = formatPercent(changePercent);
        document.getElementById('todayPLPercent').className = `stat-change ${changePercent >= 0 ? 'positive' : 'negative'}`;
        
        document.getElementById('buyingPower').textContent = formatCurrency(parseFloat(account.buying_power));
    } catch (error) {
        console.error('Error fetching account:', error);
        showError('account', `Network error: ${error.message}`);
    }
}

function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = `Error: ${message}`;
        element.style.color = 'var(--red-primary)';
    }
}

async function fetchPositions() {
    try {
        const response = await fetch(`${API_BASE}/api/positions`);
        const positions = await response.json();
        
        const positionsList = document.getElementById('positionsList');
        const positionCount = document.getElementById('positionCount');
        const positionsCountBadge = document.getElementById('positionsCount');
        
        if (positions.error) {
            positionsList.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                    <p style="color: var(--red-primary);">Error: ${positions.error}</p>
                </div>
            `;
            positionCount.textContent = '0';
            positionsCountBadge.textContent = '0';
            return;
        }
        
        if (!Array.isArray(positions) || positions.length === 0) {
            positionsList.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/>
                    </svg>
                    <p>No open positions</p>
                </div>
            `;
            positionCount.textContent = '0';
            positionsCountBadge.textContent = '0';
            return;
        }
        
        positionCount.textContent = positions.length.toString();
        positionsCountBadge.textContent = positions.length.toString();
        
        positionsList.innerHTML = positions.map(pos => {
            const pl = parseFloat(pos.unrealized_pl);
            const plPercent = parseFloat(pos.unrealized_plpc) * 100;
            return `
                <div class="position-item">
                    <div class="position-info">
                        <div class="position-symbol">${pos.symbol}</div>
                        <div class="position-qty">${pos.qty} shares @ ${formatCurrency(parseFloat(pos.avg_entry_price))}</div>
                    </div>
                    <div class="position-value">
                        <div class="position-price">${formatCurrency(parseFloat(pos.market_value))}</div>
                        <div class="position-pl ${pl >= 0 ? 'positive' : 'negative'}">
                            ${formatCurrency(pl)} (${formatPercent(plPercent)})
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('Error fetching positions:', error);
    }
}

async function fetchOrders() {
    try {
        const response = await fetch(`${API_BASE}/api/orders`);
        const orders = await response.json();
        
        const ordersList = document.getElementById('ordersList');
        
        if (orders.error) {
            ordersList.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                    <p style="color: var(--red-primary);">Error: ${orders.error}</p>
                </div>
            `;
            return;
        }
        
        if (!Array.isArray(orders) || orders.length === 0) {
            ordersList.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                        <path d="M14 2v6h6"/>
                        <path d="M16 13H8"/>
                        <path d="M16 17H8"/>
                        <path d="M10 9H8"/>
                    </svg>
                    <p>No recent orders</p>
                </div>
            `;
            return;
        }
        
        ordersList.innerHTML = orders.slice(0, 5).map(order => {
            const statusClass = order.status === 'filled' ? 'positive' : 
                               order.status === 'canceled' ? 'negative' : 'neutral';
            return `
                <div class="order-item">
                    <div class="position-info">
                        <div class="position-symbol">${order.symbol}</div>
                        <div class="position-qty">${order.side.toUpperCase()} ${order.qty} @ ${order.type}</div>
                    </div>
                    <div class="position-value">
                        <div class="position-pl ${statusClass}">${order.status.toUpperCase()}</div>
                    </div>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('Error fetching orders:', error);
    }
}

async function fetchPrediction(symbol, options = {}) {
    const { autoTrade = false, quantity = 1 } = options;

    try {
        let url = `${API_BASE}/api/prediction/${symbol}`;
        const params = new URLSearchParams();

        if (autoTrade) {
            params.append('auto_trade', 'true');
            params.append('quantity', Math.max(quantity, 1).toString());
        }

        if (params.toString()) {
            url = `${url}?${params.toString()}`;
        }

        const response = await fetch(url);
        const prediction = await response.json();

        document.getElementById('predictionSymbol').textContent = symbol;

        const signal = (prediction?.combined?.signal || 'HOLD').toUpperCase();
        const signalDisplay = document.getElementById('signalDisplay');
        signalDisplay.className = `signal-display ${signal.toLowerCase()}`;
        signalDisplay.querySelector('.signal-text').textContent = signal;

        const confidence = Math.max(0, Math.min(1, prediction.confidence || 0));
        document.getElementById('confidenceValue').textContent = `${(confidence * 100).toFixed(0)}%`;
        document.getElementById('confidenceFill').style.width = `${confidence * 100}%`;

        const weights = prediction.model_weights || {};
        const llmWeightValue = weights.alpha ?? weights.llm;
        const lstmWeightValue = weights.beta ?? weights.lstm;
        document.getElementById('llmWeight').textContent = weightDisplay(llmWeightValue);
        document.getElementById('lstmWeight').textContent = weightDisplay(lstmWeightValue);

        const llmSignal = document.getElementById('llmSignal');
        const lstmSignal = document.getElementById('lstmSignal');

        const llmData = prediction.llm || {};
        const lstmData = prediction.lstm || {};

        llmSignal.textContent = (llmData.signal || 'HOLD').toUpperCase();
        llmSignal.style.color = getSignalColor(llmSignal.textContent);
        llmSignal.style.background = getSignalBg(llmSignal.textContent);

        lstmSignal.textContent = (lstmData.signal || 'HOLD').toUpperCase();
        lstmSignal.style.color = getSignalColor(lstmSignal.textContent);
        lstmSignal.style.background = getSignalBg(lstmSignal.textContent);

        const combined = prediction.combined || { buy: 0.33, sell: 0.33, hold: 0.34 };
        updateProbabilityBar('buy', combined.buy || 0);
        updateProbabilityBar('sell', combined.sell || 0);
        updateProbabilityBar('hold', combined.hold || 0);

        updateLlmExplanation(llmData.explanation);
        renderTradeResult(prediction, symbol);

    } catch (error) {
        console.error('Error fetching prediction:', error);
    }
}

function weightDisplay(value) {
    if (value === undefined || value === null || Number.isNaN(value)) {
        return '—';
    }
    return `${(Math.max(0, Math.min(1, value)) * 100).toFixed(0)}%`;
}

function updateProbabilityBar(kind, value) {
    const clamped = Math.max(0, Math.min(1, value || 0));
    const bar = document.getElementById(`${kind}Prob`);
    const label = document.getElementById(`${kind}ProbValue`);

    if (bar) {
        bar.style.width = `${clamped * 100}%`;
    }

    if (label) {
        label.textContent = `${(clamped * 100).toFixed(0)}%`;
    }
}

function updateLlmExplanation(explanation) {
    const container = document.getElementById('llmExplanationContainer');
    const textEl = document.getElementById('llmExplanation');

    if (!container || !textEl) {
        return;
    }

    if (explanation && explanation.trim().length > 0) {
        textEl.textContent = explanation.trim();
        container.style.display = 'block';
    } else {
        textEl.textContent = '';
        container.style.display = 'none';
    }
}

function renderTradeResult(prediction, symbol) {
    const tradeEl = document.getElementById('tradeResult');
    if (!tradeEl) {
        return;
    }

    if (!prediction.auto_trade) {
        tradeEl.style.display = 'none';
        tradeEl.className = 'trade-result';
        tradeEl.textContent = '';
        return;
    }

    const tradeInfo = prediction.trade || {};
    const quantity = tradeInfo.quantity || prediction.trade_quantity || 1;
    const side = (tradeInfo.side || (prediction.combined?.signal ?? 'hold')).toString().toUpperCase();

    let message = 'Automated trade executed.';
    let styleClass = 'trade-result warning';

    if (tradeInfo.status === 'submitted') {
        message = `Order submitted: ${side} ${quantity} ${symbol.toUpperCase()}.`;
        styleClass = 'trade-result success';
    } else if (tradeInfo.status === 'failed') {
        const reason = tradeInfo.error || 'Unknown error';
        message = `Trade failed: ${reason}`;
        styleClass = 'trade-result error';
    } else if (tradeInfo.status === 'skipped') {
        message = tradeInfo.reason || 'Automated trade skipped.';
        styleClass = 'trade-result warning';
    }

    tradeEl.style.display = 'block';
    tradeEl.className = styleClass;
    tradeEl.textContent = message;
}

async function fetchPortfolioHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/history`);
        const history = await response.json();

        const timestamps = history.timestamp || history.time || [];
        const equities = Array.isArray(history.equity) ? history.equity : [];

        if (!equities.length) {
            return;
        }

        const labels = [];
        const dataPoints = [];

        equities.forEach((value, index) => {
            const numeric = Number(value);
            if (Number.isNaN(numeric)) {
                return;
            }

            dataPoints.push(numeric);
            labels.push(formatHistoryTimestamp(timestamps[index]));
        });

        if (!dataPoints.length) {
            return;
        }

        drawPortfolioChart(labels, dataPoints);
    } catch (error) {
        console.error('Error fetching portfolio history:', error);
    }
}

function formatHistoryTimestamp(value) {
    if (value === undefined || value === null) {
        return '';
    }

    let date;

    if (typeof value === 'number') {
        date = new Date(value.toString().length === 10 ? value * 1000 : value);
    } else {
        const numeric = Number(value);
        if (!Number.isNaN(numeric) && value.toString().trim() !== '') {
            date = new Date(value.length === 10 ? numeric * 1000 : numeric);
        } else {
            date = new Date(value);
        }
    }

    if (Number.isNaN(date.getTime())) {
        return '';
    }

    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function drawPortfolioChart(labels, dataPoints) {
    const canvas = document.getElementById('equityChart');
    if (!canvas || typeof Chart === 'undefined' || !dataPoints.length) {
        return;
    }

    const context = canvas.getContext('2d');

    if (portfolioChart) {
        portfolioChart.data.labels = labels;
        portfolioChart.data.datasets[0].data = dataPoints;
        portfolioChart.update();
        return;
    }

    const gradient = context.createLinearGradient(0, 0, 0, 240);
    gradient.addColorStop(0, 'rgba(80, 112, 255, 0.35)');
    gradient.addColorStop(1, 'rgba(80, 112, 255, 0)');

    portfolioChart = new Chart(context, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Equity',
                    data: dataPoints,
                    borderColor: '#5070FF',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.25,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: (ctx) => `Equity: ${formatCurrency(ctx.parsed.y)}`
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: 'var(--text-secondary)',
                        maxRotation: 0,
                        autoSkip: true
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    ticks: {
                        color: 'var(--text-secondary)',
                        callback: (value) => formatShortCurrency(value)
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                }
            }
        }
    });
}

async function executeTrade() {
    const symbol = document.getElementById('tradeSymbol').value.toUpperCase();
    const qty = parseInt(document.getElementById('tradeQty').value);
    
    if (!symbol || qty < 1) {
        alert('Please enter a valid symbol and quantity');
        return;
    }
    
    const btn = document.getElementById('tradeBtn');
    btn.disabled = true;
    btn.textContent = 'Executing...';
    
    try {
        const response = await fetch(`${API_BASE}/api/trade`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: symbol,
                qty: qty,
                side: tradeSide
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            alert(`Trade failed: ${result.error}`);
        } else {
            alert(`Order placed successfully!\n${tradeSide.toUpperCase()} ${qty} ${symbol}`);
            refreshData();
        }
    } catch (error) {
        console.error('Trade error:', error);
        alert('Trade failed. Please check your Alpaca API credentials.');
    } finally {
        btn.disabled = false;
        updateTradeButton();
    }
}

function updateTradeButton() {
    const symbol = document.getElementById('tradeSymbol').value.toUpperCase();
    const btn = document.getElementById('tradeBtn');
    
    btn.textContent = `${tradeSide === 'buy' ? 'Buy' : 'Sell'} ${symbol}`;
    btn.className = `btn btn-trade ${tradeSide}`;
}

function updateEstimatedCost() {
    const qty = parseInt(document.getElementById('tradeQty').value) || 0;
    if (currentPrice > 0) {
        document.getElementById('estimatedCost').textContent = formatCurrency(qty * currentPrice);
    } else {
        document.getElementById('estimatedCost').textContent = 'Loading...';
        // Try to fetch quote if we don't have it
        const symbol = document.getElementById('tradeSymbol').value.toUpperCase();
        if (symbol) {
            fetchQuote(symbol);
        }
    }
}

// Utility functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function formatPercent(value) {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
}

function formatShortCurrency(value) {
    const absValue = Math.abs(value);

    if (absValue >= 1_000_000_000) {
        return `$${(value / 1_000_000_000).toFixed(1)}B`;
    }

    if (absValue >= 1_000_000) {
        return `$${(value / 1_000_000).toFixed(1)}M`;
    }

    if (absValue >= 1_000) {
        return `$${(value / 1_000).toFixed(1)}K`;
    }

    return formatCurrency(value);
}

function getSignalColor(signal) {
    switch (signal) {
        case 'BUY': return '#00C805'; // Robinhood green
        case 'SELL': return '#FF5000'; // Robinhood red
        case 'HOLD': return '#8E8E93'; // Robinhood gray
        default: return '#8E8E93';
    }
}

function getSignalBg(signal) {
    switch (signal) {
        case 'BUY': return 'rgba(0, 200, 5, 0.1)';
        case 'SELL': return 'rgba(255, 80, 0, 0.1)';
        case 'HOLD': return 'rgba(142, 142, 147, 0.1)';
        default: return 'rgba(142, 142, 147, 0.1)';
    }
}

function showLoading() {
    // Add loading state if needed
}

function hideLoading() {
    // Remove loading state if needed
}

// Chat Assistant Functions
let chatOpen = false;
let chatMinimized = false;
let chatWidth = 400; // Default width
let isResizing = false;

function toggleChatWindow() {
    const chatWindow = document.getElementById('chatWindow');
    const chatFloatButton = document.getElementById('chatFloatButton');
    const main = document.querySelector('.main');
    
    chatOpen = !chatOpen;
    
    if (chatOpen) {
        chatWindow.style.display = 'flex';
        chatFloatButton.classList.add('hidden');
        main.classList.add('chat-open');
        updateChatWidth();
    } else {
        chatWindow.style.display = 'none';
        chatFloatButton.classList.remove('hidden');
        main.classList.remove('chat-open');
        main.style.marginRight = '';
    }
}

function minimizeChatWindow() {
    const chatWindow = document.getElementById('chatWindow');
    const main = document.querySelector('.main');
    chatMinimized = !chatMinimized;
    
    if (chatMinimized) {
        chatWindow.classList.add('minimized');
        main.style.marginRight = '50px';
    } else {
        chatWindow.classList.remove('minimized');
        updateChatWidth();
    }
}

function updateChatWidth() {
    const chatWindow = document.getElementById('chatWindow');
    const main = document.querySelector('.main');
    
    if (chatOpen && !chatMinimized) {
        chatWindow.style.width = `${chatWidth}px`;
        main.style.marginRight = `${chatWidth}px`;
        document.documentElement.style.setProperty('--chat-width', `${chatWidth}px`);
    }
}

function scrollChatToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatMarkdown(text) {
    if (!text) return '';
    
    // Escape HTML first
    let formatted = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    // Store code blocks with placeholders
    const codeBlocks = [];
    let codeBlockIndex = 0;
    formatted = formatted.replace(/```([\s\S]*?)```/g, (match, code) => {
        const placeholder = `__CODEBLOCK_${codeBlockIndex}__`;
        codeBlocks[codeBlockIndex] = `<pre class="chat-code-block"><code>${code}</code></pre>`;
        codeBlockIndex++;
        return placeholder;
    });
    
    // Store inline code with placeholders
    const inlineCodes = [];
    let inlineCodeIndex = 0;
    formatted = formatted.replace(/`([^`\n]+)`/g, (match, code) => {
        const placeholder = `__INLINECODE_${inlineCodeIndex}__`;
        inlineCodes[inlineCodeIndex] = `<code class="chat-inline-code">${code}</code>`;
        inlineCodeIndex++;
        return placeholder;
    });
    
    // Convert bold (**text** or __text__)
    formatted = formatted.replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/__([^_]+?)__/g, '<strong>$1</strong>');
    
    // Convert italic (*text* or _text_) - only single asterisks/underscores
    formatted = formatted.replace(/\*([^*\n]+?)\*/g, '<em>$1</em>');
    formatted = formatted.replace(/_([^_\n]+?)_/g, '<em>$1</em>');
    
    // Convert quotes (lines starting with >)
    formatted = formatted.replace(/^&gt; (.+)$/gm, '<blockquote class="chat-quote">$1</blockquote>');
    
    // Convert bullet points (- or + at start of line)
    formatted = formatted.replace(/^[\-\+] (.+)$/gm, '<li>$1</li>');
    // Convert * at start of line (bullet point, not formatting)
    formatted = formatted.replace(/^\* (.+)$/gm, '<li>$1</li>');
    
    // Wrap consecutive list items in <ul>
    formatted = formatted.replace(/(<li>.*?<\/li>(?:<br>?)+)+/g, (match) => {
        return '<ul class="chat-list">' + match.replace(/<br>/g, '') + '</ul>';
    });
    
    // Convert numbered lists
    formatted = formatted.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
    
    // Restore inline codes
    inlineCodes.forEach((code, index) => {
        formatted = formatted.replace(`__INLINECODE_${index}__`, code);
    });
    
    // Restore code blocks
    codeBlocks.forEach((code, index) => {
        formatted = formatted.replace(`__CODEBLOCK_${index}__`, code);
    });
    
    // Convert line breaks to <br>
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}

function addChatMessage(content, isUser = false, type = 'text', data = null) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${isUser ? 'user' : 'ai'}`;
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    let messageHTML = '';
    
    if (type === 'text') {
        const formattedContent = formatMarkdown(content);
        messageHTML = `
            <div class="message-content">${formattedContent}</div>
            <div class="message-time">${time}</div>
        `;
    } else if (type === 'prediction') {
        const prediction = data.prediction;
        const symbol = data.symbol;
        const price = data.price;
        
        const formattedContent = formatMarkdown(content);
        messageHTML = `
            <div class="message-content">${formattedContent}</div>
            <div class="chat-prediction-card">
                <h4>📊 ${symbol} Analysis</h4>
                <div class="chat-prediction-row">
                    <span class="chat-prediction-label">LLM:</span>
                    <span class="chat-prediction-value ${prediction.llm.signal.toLowerCase()}">${prediction.llm.signal} (${(prediction.llm.buy * 100).toFixed(0)}% buy, ${(prediction.llm.sell * 100).toFixed(0)}% sell, ${(prediction.llm.hold * 100).toFixed(0)}% hold)</span>
                </div>
                <div class="chat-prediction-row">
                    <span class="chat-prediction-label">LSTM:</span>
                    <span class="chat-prediction-value ${prediction.lstm.signal.toLowerCase()}">${prediction.lstm.signal} (${(prediction.lstm.buy * 100).toFixed(0)}% buy, ${(prediction.lstm.sell * 100).toFixed(0)}% sell, ${(prediction.lstm.hold * 100).toFixed(0)}% hold)</span>
                </div>
                <div class="chat-prediction-row">
                    <span class="chat-prediction-label">Combined:</span>
                    <span class="chat-prediction-value ${prediction.combined.signal.toLowerCase()}">${prediction.combined.signal} (${(prediction.combined.buy * 100).toFixed(0)}% buy, ${(prediction.combined.sell * 100).toFixed(0)}% sell, ${(prediction.combined.hold * 100).toFixed(0)}% hold)</span>
                </div>
                <div class="chat-prediction-row">
                    <span class="chat-prediction-label">Confidence:</span>
                    <span class="chat-prediction-value">${(prediction.confidence * 100).toFixed(0)}%</span>
                </div>
                ${price > 0 ? `<div class="chat-prediction-row">
                    <span class="chat-prediction-label">Current Price:</span>
                    <span class="chat-prediction-value">$${price.toFixed(2)}</span>
                </div>` : ''}
            </div>
            <div class="message-time">${time}</div>
        `;
    } else if (type === 'stock_info') {
        const symbol = data.symbol;
        const info = data.info;
        const infoType = data.info_type;
        
        let infoHTML = '';
        if (info) {
            infoHTML = `
                <div class="chat-prediction-card">
                    <h4>📈 ${symbol} Information</h4>
                    ${info.current_price > 0 ? `
                    <div class="chat-prediction-row">
                        <span class="chat-prediction-label">Current Price:</span>
                        <span class="chat-prediction-value" style="color: var(--green-primary);">$${info.current_price.toFixed(2)}</span>
                    </div>` : ''}
                    ${info.yesterday_close > 0 ? `
                    <div class="chat-prediction-row">
                        <span class="chat-prediction-label">Yesterday's Close:</span>
                        <span class="chat-prediction-value">$${info.yesterday_close.toFixed(2)}</span>
                    </div>` : ''}
                    ${info['52week_high'] > 0 ? `
                    <div class="chat-prediction-row">
                        <span class="chat-prediction-label">52-Week High:</span>
                        <span class="chat-prediction-value" style="color: var(--green-primary);">$${info['52week_high'].toFixed(2)}</span>
                    </div>` : ''}
                    ${info['52week_low'] > 0 ? `
                    <div class="chat-prediction-row">
                        <span class="chat-prediction-label">52-Week Low:</span>
                        <span class="chat-prediction-value" style="color: var(--red-primary);">$${info['52week_low'].toFixed(2)}</span>
                    </div>` : ''}
                    ${info.volume > 0 ? `
                    <div class="chat-prediction-row">
                        <span class="chat-prediction-label">Volume:</span>
                        <span class="chat-prediction-value">${info.volume.toLocaleString()}</span>
                    </div>` : ''}
                </div>
            `;
        }
        
        const formattedContent = formatMarkdown(content);
        messageHTML = `
            <div class="message-content">${formattedContent}</div>
            ${infoHTML}
            <div class="message-time">${time}</div>
        `;
    } else if (type === 'trade_confirmation') {
        const symbol = data.symbol;
        const action = data.action;
        const quantity = data.quantity;
        const price = data.price;
        const estimatedCost = data.estimated_cost;
        const prediction = data.prediction;
        
        let predictionHTML = '';
        if (prediction) {
            predictionHTML = `
                <div class="chat-prediction-card" style="margin-bottom: 12px;">
                    <h4>📊 ${symbol} Analysis</h4>
                    <div class="chat-prediction-row">
                        <span class="chat-prediction-label">LLM:</span>
                        <span class="chat-prediction-value ${prediction.llm.signal.toLowerCase()}">${prediction.llm.signal}</span>
                    </div>
                    <div class="chat-prediction-row">
                        <span class="chat-prediction-label">LSTM:</span>
                        <span class="chat-prediction-value ${prediction.lstm.signal.toLowerCase()}">${prediction.lstm.signal}</span>
                    </div>
                    <div class="chat-prediction-row">
                        <span class="chat-prediction-label">Combined:</span>
                        <span class="chat-prediction-value ${prediction.combined.signal.toLowerCase()}">${prediction.combined.signal} (${(prediction.confidence * 100).toFixed(0)}% confidence)</span>
                    </div>
                </div>
            `;
        }
        
        const formattedContent = formatMarkdown(content);
        messageHTML = `
            <div class="message-content">${formattedContent}</div>
            ${predictionHTML}
            <div class="chat-confirmation">
                <div style="margin-bottom: 8px; font-size: 12px; color: var(--text-secondary);">
                    ${action === 'buy' ? 'Buy' : 'Sell'} ${quantity} share${quantity > 1 ? 's' : ''} of ${symbol}
                    ${price > 0 ? ` at ~$${price.toFixed(2)}` : ''}
                    ${estimatedCost > 0 ? ` (Est. ${estimatedCost.toFixed(2)})` : ''}
                </div>
                <div class="chat-confirmation-buttons">
                    <button class="chat-confirm-btn cancel" onclick="cancelTrade()">Cancel</button>
                    <button class="chat-confirm-btn ${action}" onclick="confirmTrade('${symbol}', ${quantity}, '${action}')">
                        Confirm ${action === 'buy' ? 'Buy' : 'Sell'}
                    </button>
                </div>
            </div>
            <div class="message-time">${time}</div>
        `;
    } else if (type === 'loading') {
        messageHTML = `
            <div class="chat-loading">
                <div class="chat-loading-dot"></div>
                <div class="chat-loading-dot"></div>
                <div class="chat-loading-dot"></div>
                <span>${content}</span>
            </div>
        `;
    }
    
    messageDiv.innerHTML = messageHTML;
    chatMessages.appendChild(messageDiv);
    scrollChatToBottom();
    
    return messageDiv;
}

function removeLoadingMessage() {
    const chatMessages = document.getElementById('chatMessages');
    const loadingMessages = chatMessages.querySelectorAll('.chat-loading');
    loadingMessages.forEach(msg => msg.parentElement.remove());
}

async function sendChatMessage(message) {
    if (!message.trim()) return;
    
    // Add user message
    addChatMessage(message, true);
    
    // Add loading message
    const loadingDiv = addChatMessage('Analyzing your request...', false, 'loading');
    
    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Remove loading
        removeLoadingMessage();
        
        // Handle response based on type
        if (data.type === 'text') {
            addChatMessage(data.message, false);
        } else if (data.type === 'stock_info') {
            addChatMessage(data.message, false, 'stock_info', {
                symbol: data.symbol,
                info: data.info,
                info_type: data.info_type
            });
        } else if (data.type === 'prediction') {
            addChatMessage(data.message, false, 'prediction', {
                symbol: data.symbol,
                prediction: data.prediction,
                price: data.price
            });
        } else if (data.type === 'trade_confirmation') {
            addChatMessage(data.message, false, 'trade_confirmation', {
                symbol: data.symbol,
                action: data.action,
                quantity: data.quantity,
                price: data.price,
                estimated_cost: data.estimated_cost,
                prediction: data.prediction
            });
        } else if (data.type === 'error') {
            addChatMessage(`Error: ${data.message}`, false);
        }
    } catch (error) {
        removeLoadingMessage();
        addChatMessage(`Error: ${error.message}`, false);
    }
}

async function confirmTrade(symbol, quantity, side) {
    // Add loading message
    const loadingDiv = addChatMessage('Executing trade...', false, 'loading');
    
    try {
        const response = await fetch(`${API_BASE}/api/chat/confirm`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: symbol,
                quantity: quantity,
                action: side
            })
        });
        
        const data = await response.json();
        
        // Remove loading
        removeLoadingMessage();
        
        if (data.success) {
            addChatMessage(data.message, false);
            // Refresh portfolio data
            setTimeout(() => {
                refreshData();
            }, 1000);
        } else {
            addChatMessage(data.message, false);
        }
    } catch (error) {
        removeLoadingMessage();
        addChatMessage(`Error executing trade: ${error.message}`, false);
    }
}

function cancelTrade() {
    addChatMessage('Trade cancelled.', false);
}

// Initialize chat event listeners
function setupChatListeners() {
    const chatFloatButton = document.getElementById('chatFloatButton');
    const chatWindow = document.getElementById('chatWindow');
    const chatClose = document.getElementById('chatClose');
    const chatMinimize = document.getElementById('chatMinimize');
    const chatResizeHandle = document.getElementById('chatResizeHandle');
    const chatInput = document.getElementById('chatInput');
    const chatSend = document.getElementById('chatSend');
    
    // Floating button click
    if (chatFloatButton) {
        chatFloatButton.addEventListener('click', toggleChatWindow);
    }
    
    // Close button
    if (chatClose) {
        chatClose.addEventListener('click', toggleChatWindow);
    }
    
    // Minimize button
    if (chatMinimize) {
        chatMinimize.addEventListener('click', (e) => {
            e.stopPropagation();
            minimizeChatWindow();
        });
    }
    
    // Click on minimized window to expand
    if (chatWindow) {
        chatWindow.addEventListener('click', (e) => {
            if (chatMinimized && !e.target.closest('.chat-window-controls')) {
                minimizeChatWindow();
            }
        });
    }
    
    // Resize handle
    if (chatResizeHandle) {
        let startX, startWidth;
        
        chatResizeHandle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startX = e.clientX;
            startWidth = chatWidth;
            chatResizeHandle.classList.add('resizing');
            document.body.style.cursor = 'ew-resize';
            document.body.style.userSelect = 'none';
            
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            
            const diff = startX - e.clientX; // Reverse because we're resizing from left
            const newWidth = Math.max(300, Math.min(800, startWidth + diff));
            chatWidth = newWidth;
            updateChatWidth();
        });
        
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                chatResizeHandle.classList.remove('resizing');
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            }
        });
    }
    
    // Send button
    if (chatSend) {
        chatSend.addEventListener('click', () => {
            const message = chatInput.value.trim();
            if (message) {
                sendChatMessage(message);
                chatInput.value = '';
            }
        });
    }
    
    // Input enter key
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const message = chatInput.value.trim();
                if (message) {
                    sendChatMessage(message);
                    chatInput.value = '';
                }
            }
        });
    }
}

