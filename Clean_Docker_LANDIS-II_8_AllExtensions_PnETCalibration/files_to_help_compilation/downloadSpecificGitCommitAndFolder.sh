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

# Extract repository name from URL
REPO_NAME=$(basename -s .git $REPO_URL)

# Create and enter the repository directory
mkdir $REPO_NAME
cd $REPO_NAME

# Initialize a new Git repository
git init

# Add the remote repository
git remote add origin $REPO_URL

# Enable sparse checkout
git config core.sparseCheckout true

# Specify the folder to checkout
echo "$FOLDER_PATH" > .git/info/sparse-checkout

# Fetch the specific commit
git fetch --depth 1 origin $COMMIT_HASH

# Checkout the specified commit
git checkout $COMMIT_HASH

echo "Sparse checkout completed for $FOLDER_PATH from commit $COMMIT_HASH"
