#!/bin/bash

# Configuration
REMOTE_USER="anton"
REMOTE_HOST="abakus"
REMOTE_DEST="~/AuViMi"
LOCAL_DEST="./pulled_movies"

echo "📂 Creating local directory: ${LOCAL_DEST}"
mkdir -p "${LOCAL_DEST}"

echo "🔄 Pulling transformed movies from abakus..."

# This command pulls only the .mp4 files from the sessions and results folders,
# maintaining the timestamped directory structure.
# --include='*/'        : include all directories (so we can find the files inside)
# --include='*.mp4'     : include all MP4 files
# --exclude='*'         : exclude everything else (the thousands of .jpg frames)

# 1. Pull from server sessions (transformed.mp4 and original.mp4)
echo "🎬 Checking sessions/..."
rsync -avz -e "ssh" \
    --include='*/' \
    --include='*.mp4' \
    --exclude='*' \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEST}/sessions/ \
    "${LOCAL_DEST}/sessions/"

# 2. Pull from host results (mirror.mp4 and input.mp4)
echo "🎬 Checking results/..."
rsync -avz -e "ssh" \
    --include='*/' \
    --include='*.mp4' \
    --exclude='*' \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEST}/results/ \
    "${LOCAL_DEST}/results/"

echo ""
echo "✨ Done! Your movies are in: ${LOCAL_DEST}"
ls -R "${LOCAL_DEST}"
