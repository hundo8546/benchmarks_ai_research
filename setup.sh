#!/bin/bash
set -e

echo "=== [1/8] Installing pip dependencies ==="
pip install --upgrade torch torchvision
pip install \
  pandas \
  numpy \
  decord \
  opencv-python \
  Pillow \
  tqdm \
  scikit-learn \
  joblib \
  transformers \
  huggingface_hub \
  bitsandbytes \
  accelerate \
  py7zr \
  matplotlib \
  scipy \
  rich \
  ipykernel \
  typer

echo ""
echo "=== [2/8] Creating required directories ==="
mkdir -p /workspace/data/genbuster-mini/extracted
mkdir -p /workspace/benchmarks_ai_research/CNNDetection/checkpoints
mkdir -p /workspace/benchmarks_ai_research/routing

echo ""
echo "=== [3/8] Setting PYTHONPATH ==="
export PYTHONPATH=$PYTHONPATH:/workspace/benchmarks_ai_research/CNNDetection
echo 'export PYTHONPATH=$PYTHONPATH:/workspace/benchmarks_ai_research/CNNDetection' >> ~/.bashrc

echo ""
echo "=== [4/8] HuggingFace login ==="
read -p "Enter HuggingFace token: " HF_TOKEN
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

echo ""
echo "=== [5/8] Downloading GenBuster dataset ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='l8cv/GenBuster-200K-mini',
    repo_type='dataset',
    local_dir='/workspace/data/genbuster-mini'
)
"

echo ""
echo "=== [6/8] Extracting GenBuster dataset ==="
python -c "
import py7zr, os
archive = '/workspace/data/genbuster-mini/GenBuster-200K-mini.7z'
if os.path.exists(archive):
    py7zr.SevenZipFile(archive, mode='r').extractall(path='/workspace/data/genbuster-mini/extracted')
    print('Extraction complete')
else:
    print('WARNING: Archive not found at', archive)
"

echo ""
echo "=== [7/8] Downloading CNNDetection weights ==="
wget -q --show-progress \
  -O /workspace/benchmarks_ai_research/CNNDetection/checkpoints/blur_jpg_prob0.1.pth \
  "https://www.dropbox.com/s/h7tkpcgiwuftb6g/blur_jpg_prob0.1.pth?dl=1"

wget -q --show-progress \
  -O /workspace/benchmarks_ai_research/CNNDetection/checkpoints/blur_jpg_prob0.5.pth \
  "https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1"

echo ""
echo "=== [8/8] Verifying imports ==="
python -c "
import torch, torchvision, pandas, numpy, cv2, tqdm, sklearn, joblib, PIL, py7zr
import matplotlib, scipy, rich, typer
from decord import VideoReader, cpu
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    CLIPProcessor,
    CLIPModel,
)
from huggingface_hub import snapshot_download
print('All imports OK')
print('Torch version:       ', torch.__version__)
print('Transformers version:', __import__('transformers').__version__)
print('CUDA available:      ', torch.cuda.is_available())
"

echo ""
echo "=== Setup complete ==="