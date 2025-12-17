#!/bin/bash

# 1. Install Cloudflared if missing
if [ ! -f cloudflared ]; then
    echo "⬇️  Downloading Cloudflared..."
    wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    mv cloudflared-linux-amd64 cloudflared
    chmod +x cloudflared
fi

# 2. Kill old tunnels to prevent conflicts
pkill cloudflared

# 3. Start the Tunnel (pointing to port 8000)
echo "🚀 Starting Cloudflare Tunnel on Port 8000..."
# We run it in the background but keep the logs flowing to a file
nohup ./cloudflared tunnel --url http://localhost:8000 > tunnel.log 2>&1 &

# 4. Loop to find and display the URL
echo "⏳ Searching for URL..."
found_url=""
while [ -z "$found_url" ]; do
    if [ -f tunnel.log ]; then
        found_url=$(grep -o 'https://.*\.trycloudflare\.com' tunnel.log | head -n 1)
    fi
    sleep 1
done

echo ""
echo "========================================================"
echo "✅  YOUR PUBLIC URL:  $found_url"
echo "========================================================"
echo "👉 Copy this URL to Vercel."
echo "👉 Now run the FastAPI cell in your notebook!"
echo "   (Press Ctrl+C to stop the tunnel)"

# 5. Keep the script running to keep the tunnel alive
tail -f tunnel.log