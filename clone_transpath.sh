#!/bin/bash

# Script to clone TransPath repository and download pre-trained models
# TransPath: Transformer-based Unsupervised Contrastive Learning for Histopathological Image Classification

# Configuration
REPO_URL="https://github.com/Xiyue-Wang/TransPath.git"
GIT_DIR="."
REPO_NAME="TransPath"
TARGET_DIR="${GIT_DIR}/${REPO_NAME}"

# Pre-trained model Google Drive file IDs
CTRANSPATH_MODEL_ID="1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX"
CTRANSPATH_MODEL_NAME="ctranspath.pth"
MOCov3_MODEL_ID="13d_SHy9t9JCwp_MsU2oOUZ5AvI6tsC-K"
MOCov3_MODEL_NAME="vit_small.pth.tar"
TRANSPATH_MODEL_ID="1dhysqcv_Ct_A96qOF8i6COTK3jLb56vx"
TRANSPATH_MODEL_NAME="transpath.pth"

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed or not in PATH"
    exit 1
fi

# Function to download Google Drive file using gdown
download_gdrive_file() {
    local file_id=$1
    local output_file=$2
    local output_path="${TARGET_DIR}/${output_file}"
    
    if [ -f "${output_path}" ]; then
        echo "Model ${output_file} already exists, skipping download."
        return 0
    fi
    
    # Check if gdown is available
    if ! command -v gdown &> /dev/null; then
        echo "gdown is not installed. Installing gdown..."
        pip install gdown -q
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install gdown. Please install it manually: pip install gdown"
            return 1
        fi
    fi
    
    echo "Downloading ${output_file} from Google Drive..."
    gdown "https://drive.google.com/uc?id=${file_id}" -O "${output_path}"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded ${output_file}"
        return 0
    else
        echo "Warning: Failed to download ${output_file}. You may need to download it manually."
        echo "  URL: https://drive.google.com/file/d/${file_id}/view?usp=sharing"
        return 1
    fi
}

# Check if target directory already exists
if [ -d "${TARGET_DIR}" ]; then
    echo "Directory ${TARGET_DIR} already exists."
    read -p "Do you want to update it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating existing repository..."
        cd "${TARGET_DIR}" || exit 1
        git pull
        if [ $? -eq 0 ]; then
            echo "Repository updated successfully."
        else
            echo "Error: failed to update repository."
            exit 1
        fi
    else
        echo "Skipping clone/update."
        exit 0
    fi
else
    # Clone the repository
    echo "Cloning ${REPO_NAME} into ${TARGET_DIR}..."
    cd "${GIT_DIR}" || exit 1
    git clone "${REPO_URL}" "${REPO_NAME}"
    
    if [ $? -eq 0 ]; then
        echo "Successfully cloned ${REPO_NAME} into ${TARGET_DIR}"
    else
        echo "Error: failed to clone repository."
        exit 1
    fi
fi

# Download pre-trained models
echo ""
echo "Downloading pre-trained models..."
echo "================================"

# Download CTransPath model (recommended)
echo ""
echo "1. Downloading CTransPath model (recommended)..."
download_gdrive_file "${CTRANSPATH_MODEL_ID}" "${CTRANSPATH_MODEL_NAME}"

# Download MoCo v3 model
echo ""
echo "2. Downloading MoCo v3 model..."
download_gdrive_file "${MOCov3_MODEL_ID}" "${MOCov3_MODEL_NAME}"

# Download TransPath model
echo ""
echo "3. Downloading TransPath model..."
download_gdrive_file "${TRANSPATH_MODEL_ID}" "${TRANSPATH_MODEL_NAME}"

echo ""
echo "Done."
echo ""
echo "Note: If any downloads failed, you can download them manually from:"
echo "  - CTransPath: https://drive.google.com/file/d/${CTRANSPATH_MODEL_ID}/view?usp=sharing"
echo "  - MoCo v3: https://drive.google.com/file/d/${MOCov3_MODEL_ID}/view?usp=sharing"
echo "  - TransPath: https://drive.google.com/file/d/${TRANSPATH_MODEL_ID}/view?usp=sharing"

