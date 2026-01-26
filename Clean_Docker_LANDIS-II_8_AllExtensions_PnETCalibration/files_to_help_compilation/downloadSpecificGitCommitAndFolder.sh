#!/bin/bash

# Check if all required arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <github_repo_url> <commit_hash> <folder_path>"
    exit 1
fi

# Assign arguments to variables
REPO_URL=$1
COMMIT_HASH=$2
FOLDER_PATH=$3

# Extract repository owner and name from URL
# Handles both https://github.com/owner/repo and https://github.com/owner/repo.git
REPO_OWNER=$(echo $REPO_URL | sed -E 's|https?://github.com/([^/]+)/.*|\1|')
REPO_NAME=$(basename -s .git $REPO_URL)

# Create temporary directory for download
TEMP_DIR="${REPO_NAME}_temp"
mkdir -p $TEMP_DIR

echo "Downloading commit $COMMIT_HASH from $REPO_OWNER/$REPO_NAME..."

# Download specific commit as archive with retry logic
MAX_RETRIES=5
RETRY_COUNT=0
SUCCESS=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -L -f -o "$TEMP_DIR/archive.zip" \
        "https://github.com/${REPO_OWNER}/${REPO_NAME}/archive/${COMMIT_HASH}.zip"; then
        SUCCESS=true
        echo "Download successful"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "Download failed. Retry $RETRY_COUNT of $MAX_RETRIES..."
        sleep 3
    fi
done

if [ "$SUCCESS" = false ]; then
    echo "Failed to download after $MAX_RETRIES attempts"
    rm -rf $TEMP_DIR
    exit 1
fi

# Extract archive
echo "Extracting archive..."
unzip -q "$TEMP_DIR/archive.zip" -d $TEMP_DIR

# Find extracted directory (GitHub names it as REPO_NAME-COMMIT_HASH)
EXTRACTED_DIR=$(find $TEMP_DIR -maxdepth 1 -type d -name "${REPO_NAME}-*" | head -n 1)

if [ -z "$EXTRACTED_DIR" ]; then
    echo "Error: Could not find extracted directory"
    rm -rf $TEMP_DIR
    exit 1
fi

# Create target directory
mkdir -p $REPO_NAME

# Check if we should keep everything or just a specific folder
if [ "$FOLDER_PATH" = "/" ]; then
    echo "Extracting entire repository..."
    cp -r "$EXTRACTED_DIR"/* "$REPO_NAME/"
    echo "Successfully extracted all files from commit $COMMIT_HASH"
elif [ -d "$EXTRACTED_DIR/$FOLDER_PATH" ]; then
    cp -r "$EXTRACTED_DIR/$FOLDER_PATH" "$REPO_NAME/"
    echo "Successfully extracted $FOLDER_PATH from commit $COMMIT_HASH"
else
    echo "Warning: Folder path $FOLDER_PATH not found in commit"
    echo "Extracting entire repository instead..."
    cp -r "$EXTRACTED_DIR"/* "$REPO_NAME/"
fi

# Cleanup
rm -rf $TEMP_DIR

echo "Download and extraction completed"
