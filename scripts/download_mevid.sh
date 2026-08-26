#!/bin/bash

# Download script for MEVID dataset

# Set project root
PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0")")")
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT=$(pwd) # Fallback if realpath is not available
fi

# Auto-detect Google Drive across different environments, or use provided argument
if [ ! -z "$1" ]; then
    TARGET_DIR="$1"
elif [ -d "/content/drive/MyDrive" ]; then
    # Google Colab environment
    TARGET_DIR="/content/drive/MyDrive/mevid"
else
    # Try to find Google Drive on macOS natively
    MAC_GDRIVE=$(ls -d "$HOME/Library/CloudStorage"/GoogleDrive-*/"My Drive" 2>/dev/null | head -n 1)
    if [ ! -z "$MAC_GDRIVE" ] && [ -d "$MAC_GDRIVE" ]; then
        TARGET_DIR="$MAC_GDRIVE/mevid"
    elif [ -d "/Volumes/GoogleDrive/My Drive" ]; then
        TARGET_DIR="/Volumes/GoogleDrive/My Drive/mevid"
    elif [ -d "/g/My Drive" ]; then
        TARGET_DIR="/g/My Drive/mevid" # Windows Git Bash common path
    else
        echo "Could not auto-detect Google Drive. Defaulting to local directory."
        TARGET_DIR="$PROJECT_ROOT/data/mevid"
    fi
fi

echo "Using target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Create a symlink in the project root so local code still works seamlessly
if [ "$TARGET_DIR" != "$PROJECT_ROOT/data/mevid" ]; then
    rm -rf "$PROJECT_ROOT/data/mevid" 2>/dev/null
    mkdir -p "$PROJECT_ROOT/data"
    ln -s "$TARGET_DIR" "$PROJECT_ROOT/data/mevid"
    echo "Created symlink: $PROJECT_ROOT/data/mevid -> $TARGET_DIR"
fi

# URLs
URL_ANNOTATIONS="https://mevadata-public-01.s3.amazonaws.com/mevid-annotations/mevid-v1-annotation-data.zip"
URL_BBOX_TRAIN="https://mevadata-public-01.s3.amazonaws.com/mevid-annotations/mevid-v1-bbox-train.tgz"
URL_BBOX_TEST="https://mevadata-public-01.s3.amazonaws.com/mevid-annotations/mevid-v1-bbox-test.tgz"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Determine download command
if command_exists wget; then
    DOWNLOAD_CMD="wget -q --show-progress -O"
elif command_exists curl; then
    DOWNLOAD_CMD="curl -L --progress-bar -o"
else
    echo "Error: Neither wget nor curl is available. Please install one of them."
    exit 1
fi

echo "Target directory: $TARGET_DIR"
cd "$TARGET_DIR" || exit 1

# Download Annotations
if [ ! -d "mevid-v1-annotation-data" ]; then
    echo "Downloading annotations..."
    $DOWNLOAD_CMD mevid-v1-annotation-data.zip "$URL_ANNOTATIONS"
    if [ $? -eq 0 ]; then
        echo "Extracting annotations..."
        unzip -q mevid-v1-annotation-data.zip
        rm mevid-v1-annotation-data.zip
    else
        echo "Error downloading annotations."
        exit 1
    fi
else
    echo "Annotations already exist. Skipping download."
fi

# Download Train bbox
if [ ! -d "bbox_train" ]; then
    echo "Downloading train bbox..."
    $DOWNLOAD_CMD mevid-v1-bbox-train.tgz "$URL_BBOX_TRAIN"
    if [ $? -eq 0 ]; then
        echo "Extracting train bbox..."
        tar -xzf mevid-v1-bbox-train.tgz
        rm mevid-v1-bbox-train.tgz
    else
        echo "Error downloading train bbox."
        exit 1
    fi
else
    echo "Train bbox already exists. Skipping download."
fi

# Download Test bbox
if [ ! -d "bbox_test" ]; then
    echo "Downloading test bbox..."
    $DOWNLOAD_CMD mevid-v1-bbox-test.tgz "$URL_BBOX_TEST"
    if [ $? -eq 0 ]; then
        echo "Extracting test bbox..."
        tar -xzf mevid-v1-bbox-test.tgz
        rm mevid-v1-bbox-test.tgz
    else
        echo "Error downloading test bbox."
        exit 1
    fi
else
    echo "Test bbox already exists. Skipping download."
fi

echo ""
echo "=== Dataset Statistics ==="
if [ -d "bbox_train" ]; then
    echo "Number of training images: $(find bbox_train -type f -name '*.jpg' | wc -l | tr -d ' ')"
fi
if [ -d "bbox_test" ]; then
    echo "Number of test images: $(find bbox_test -type f -name '*.jpg' | wc -l | tr -d ' ')"
fi
echo "Done!"
