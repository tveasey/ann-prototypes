#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# You can change this to any recent version you need
ES_VERSION="9.2.0"
# ---------------------

# 1. Determine Architecture and Download
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
  ES_ARCH="darwin-x86_64"
  echo "Detected Intel architecture (x86_64)."
elif [ "$ARCH" = "arm64" ]; then
  ES_ARCH="darwin-aarch64"
  echo "Detected Apple Silicon architecture (arm64)."
else
  echo "❌ Error: Unsupported architecture: $ARCH"
  exit 1
fi

FILENAME="elasticsearch-${ES_VERSION}-${ES_ARCH}.tar.gz"
URL="https://artifacts.elastic.co/downloads/elasticsearch/$FILENAME"
DIR_NAME="elasticsearch-${ES_VERSION}"

# 2. Download if the file doesn't already exist
if [ -f "$FILENAME" ]; then
    echo "⏩ $FILENAME already exists. Skipping download."
else
    echo "Downloading $URL..."
    curl -fO $URL # -f fails fast on 404 errors
    curl https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-${ES_VERSION}-${ES_ARCH}.tar.gz.sha512 | shasum -a 512 -c -
    echo "✅ Download complete."
fi

# 3. Install (Extract)
# Extract if the directory doesn't already exist
if [ -d "$DIR_NAME" ]; then
    echo "⏩ $DIR_NAME already exists. Skipping extraction."
else
    echo "Extracting $FILENAME..."
    tar -xzf $FILENAME
    echo "✅ Extraction complete."
fi

# 4. Start Elasticsearch
echo "🚀 Starting Elasticsearch..."
echo "----------------------------------------------------------------"
echo "‼️ IMPORTANT ‼️"
echo "On the first run, Elasticsearch will generate a password for the 'elastic' user, "
echo "an HTTP certificate fingerprint and an enrollment token for Kibana. This is critical!"
echo ""
echo "👉 LOOK for lines in the output below like:"
echo "    - ℹ️  \"Password for the elastic user (reset with `bin/elasticsearch-reset-password -u elastic`):\n<YOUR_PASSWORD>\""
echo "    - ℹ️  \"HTTP CA certificate SHA-256 fingerprint:\n<YOUR_FINGERPRINT>:\""
echo "    - ℹ️  \"Configure Kibana to use this cluster:\n..."
echo ""
echo "Add them to the .env file as ELASTIC_PASSWORD and CERT_FINGERPRINT."
echo "To stop Elasticsearch, press Ctrl+C in this terminal."
echo "----------------------------------------------------------------"
sleep 5

# Change into the directory and run
cd $DIR_NAME
./bin/elasticsearch