#!/bin/bash
# Thanks ChatGPT!

# Check if a directory is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <download-directory>"
    exit 1
fi

# Create the directory if it doesn't exist
DOWNLOAD_DIR="$1"
mkdir -p "$DOWNLOAD_DIR"

# Define the list of files to download
URLS=(
	"LICENSE.txt"
	"README.md"
	"USE_POLICY.md"
	"config.json"
	"model-00001-of-00002.safetensors"
	"model-00002-of-00002.safetensors"
	"model.safetensors.index.json"
	"special_tokens_map.json"
	"tokenizer.json"
	"tokenizer_config.json"
)


# Download each file to the specified directory
for URL in "${URLS[@]}"; do
    wget -nc -P "$DOWNLOAD_DIR" "$URL"
done
