#!/bin/bash

# Configuration
REMOTE_USER="anton"
REMOTE_HOST="abakus"
REMOTE_DEST="~/AuViMi"
PORT=8000

# 1. Sync code to server
echo "🚀 Syncing code to abakus..."
rsync -avz -e "ssh -o ClearAllForwardings=yes" \
           --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
           ./ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEST}/

# 2. Local Cleanup (Mac side)
echo "🧹 Cleaning up local port ${PORT}..."
lsof -ti:${PORT} | xargs kill -9 2>/dev/null || true

# 3. Remote Cleanup (kill old server)
echo "🧹 Cleaning up remote port ${PORT}..."
ssh -o "ClearAllForwardings=yes" ${REMOTE_USER}@${REMOTE_HOST} "fuser -k ${PORT}/tcp 2>/dev/null || true"

# 4. Remote Setup (ensure venv and deps are ready)
echo "📦 Ensuring dependencies are installed..."
ssh -o "ClearAllForwardings=yes" ${REMOTE_USER}@${REMOTE_HOST} "
    cd ${REMOTE_DEST}
    [ -d .venv ] || python3 -m venv .venv
    source .venv/bin/activate
    command -v uv &>/dev/null || pip install uv &>/dev/null
    uv pip install -e . &>/dev/null || pip install -e . &>/dev/null
"

# 5. Start server with tunnel
echo ""
echo "============================================"
echo "🌐 Starting server with SSH tunnel..."
echo "============================================"
echo ""
echo "💡 In another terminal, run:"
echo "   python3 client.py --host localhost"
echo ""
echo "(Ignore the 8888 warning below - it's from your SSH config)"
echo ""

# NOTE: We do NOT use ClearAllForwardings here because it would kill our -L tunnel!
# The 8888 warning is harmless.
ssh -L ${PORT}:localhost:${PORT} ${REMOTE_USER}@${REMOTE_HOST} \
    "cd ${REMOTE_DEST} && source .venv/bin/activate && python3 server.py $*"
