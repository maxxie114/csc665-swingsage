#!/bin/bash

###############################################################################
# Cloudflare Named Tunnel Launcher for Colab
#
# GOAL:
#   - Expose your FastAPI app running in Colab on http://localhost:8000
#   - Via a stable URL like https://swingsage.qubemc.com
#   - Using a named Cloudflare Tunnel (NO more random trycloudflare.com links)
#
# OVERVIEW OF WHAT YOU NEED TO DO (ONE-TIME SETUP OUTSIDE COLAB)
# ---------------------------------------------------------------------------
# 0. PREREQUISITES
#    - You own the domain: qubemc.com
#    - qubemc.com is already using Cloudflare for DNS (you mentioned it is)
#    - You have a Cloudflare account and can log into the dashboard
#
# 1. CREATE A NAMED TUNNEL IN CLOUDFLARE DASHBOARD
#    (This is done ONCE in the browser, NOT in Colab)
#
#    a) Log in to Cloudflare dashboard.
#    b) Go to:
#         - "Zero Trust" (or "Access") → "Tunnels" (sometimes under "Networks")
#       or search for "Tunnels" in the Cloudflare UI.
#    c) Create a new tunnel, e.g. name it:
#         swingsage-tunnel
#    d) Cloudflare will show you a command to run cloudflared with a token,
#       something like:
#         cloudflared tunnel --no-autoupdate run --token <LONG_TOKEN_STRING>
#
#       COPY that <LONG_TOKEN_STRING> — this is your TUNNEL TOKEN.
#
# 2. CONFIGURE THE PUBLIC HOSTNAME FOR THE TUNNEL
#    (Still in Cloudflare Zero Trust dashboard)
#
#    a) In the "Tunnels" list, click your new tunnel (e.g. swingsage-tunnel).
#    b) Find the section "Public hostnames" (or "Add public hostname").
#    c) Add a new hostname:
#         - Hostname:  swingsage.qubemc.com
#         - Type:      HTTP
#         - URL:       http://localhost:8000
#
#       This tells Cloudflare:
#       "When someone visits https://swingsage.qubemc.com, forward that
#        traffic through this tunnel to http://localhost:8000 on the
#        machine running cloudflared."
#
#    d) Cloudflare will automatically create a DNS record for
#       swingsage.qubemc.com that points to the tunnel.
#
#    Now, all tunnel routing logic lives in Cloudflare:
#      swingsage.qubemc.com -> your named tunnel -> localhost:8000
#
# 3. USE THIS SCRIPT IN COLAB
#    (This is what you're editing now)
#
#    a) Paste your TUNNEL TOKEN (from step 1d) into the variable below:
#         CLOUDFLARE_TUNNEL_TOKEN="..."
#
#    b) Upload this script to Colab (e.g. /content/run_cloudflare.sh) and:
#         !chmod +x run_cloudflare.sh
#         !bash run_cloudflare.sh
#
#    c) In another Colab cell, run your FastAPI app on port 8000, e.g.:
#
#         import nest_asyncio, uvicorn
#         from fastapi import FastAPI
#
#         nest_asyncio.apply()
#         app = FastAPI()
#
#         @app.get("/health")
#         def health():
#             return {"status": "ok"}
#
#         uvicorn.run(app, host="0.0.0.0", port=8000)
#
#    d) Your backend will then be reachable at:
#         https://swingsage.qubemc.com
#
# 4. CONNECTING FROM VERCEL FRONTEND
#
#    - In your Vercel project settings, set an env var, e.g.:
#         NEXT_PUBLIC_API_URL="https://swingsage.qubemc.com"
#
#    - In your frontend code, use that env var as the base URL.
#
# 5. IMPORTANT NOTES / CAVEATS
#
#    - Colab is still ephemeral:
#        * When the runtime shuts down, the tunnel + FastAPI die.
#        * But the PUBLIC URL stays the SAME; it just stops responding until
#          you rerun this script + your FastAPI cell.
#
#    - This script assumes:
#        * FastAPI runs on http://localhost:8000 inside Colab.
#        * Cloudflare tunnel for swingsage.qubemc.com is properly configured
#          to forward to http://localhost:8000.
#
###############################################################################

#########################
# USER CONFIG SECTION   #
#########################

# The stable hostname you configured in Cloudflare for this tunnel:
PUBLIC_HOSTNAME="swingsage.qubemc.com"

# Where your FastAPI app is listening INSIDE Colab:
# (This must match the "URL" you set in the tunnel's "Public hostname" config)
LOCAL_URL="http://localhost:8000"

# 🔐 Tunnel token from Cloudflare Zero Trust → Tunnels → <your tunnel> → "Run tunnel"
#    Example format: CLOUDFLARE_TUNNEL_TOKEN="eyJhIjoi...<long_string>..."
#    DO NOT COMMIT THIS TOKEN TO PUBLIC REPOS.
CLOUDFLARE_TUNNEL_TOKEN="PUT_YOUR_TUNNEL_TOKEN_HERE"

#########################
# SCRIPT STARTS HERE    #
#########################

# 0. Basic sanity check: ensure user has set the token
if [ "$CLOUDFLARE_TUNNEL_TOKEN" = "PUT_YOUR_TUNNEL_TOKEN_HERE" ]; then
    echo "❌ ERROR: You must edit this script and set CLOUDFLARE_TUNNEL_TOKEN."
    echo "   Get it from Cloudflare Zero Trust → Tunnels → your tunnel → 'Run tunnel' command."
    exit 1
fi

# 1. Install Cloudflared if missing (local copy in the current directory)
if [ ! -f cloudflared ]; then
    echo "⬇️  Downloading Cloudflared (Linux AMD64)..."
    wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    mv cloudflared-linux-amd64 cloudflared
    chmod +x cloudflared
else
    echo "✅ cloudflared binary already present."
fi

# 2. Kill old tunnels to prevent conflicts
#    This is useful if you re-run the script multiple times in the same Colab session.
echo "🧹 Stopping any existing cloudflared processes..."
pkill cloudflared || echo "   (No existing cloudflared process found.)"

# 3. Start the named Cloudflare Tunnel using the token
#
#    NOTE:
#      - We do NOT pass --url here. Named tunnels with tokens rely on the
#        Cloudflare-side config (public hostname → localhost:port) that you
#        set up in the dashboard.
#      - The token fully identifies the tunnel + account + config.
#
echo "🚀 Starting Cloudflare Tunnel for ${PUBLIC_HOSTNAME}..."
echo "   This will forward ${PUBLIC_HOSTNAME} → ${LOCAL_URL}"
echo "   (as configured in Cloudflare Zero Trust)."

# Run cloudflared in the background and send logs to tunnel.log
nohup ./cloudflared tunnel --no-autoupdate run --token "$CLOUDFLARE_TUNNEL_TOKEN" > tunnel.log 2>&1 &

# 4. Basic check to see if cloudflared started
sleep 5

if pgrep -f "cloudflared tunnel" > /dev/null; then
    echo ""
    echo "========================================================"
    echo "✅  Cloudflare Tunnel is RUNNING"
    echo "🌐  PUBLIC URL:  https://${PUBLIC_HOSTNAME}"
    echo "🎯  LOCAL TARGET: ${LOCAL_URL}"
    echo "========================================================"
    echo "👉 Point your Vercel frontend (e.g. NEXT_PUBLIC_API_URL)"
    echo "   to: https://${PUBLIC_HOSTNAME}"
    echo ""
    echo "📜 Tail the logs with:  !tail -f tunnel.log"
    echo "   (in a separate cell, if you want to debug)"
else
    echo ""
    echo "❌ ERROR: cloudflared does not appear to be running."
    echo "   Check tunnel.log for details:"
    echo "   !sed -n '1,120p' tunnel.log"
    exit 1
fi

# 5. Keep the script attached to the log output so you can see what's happening
echo ""
echo "📡 Streaming Cloudflare Tunnel logs (Ctrl+C to stop viewing logs)."
tail -f tunnel.log
