#!/usr/bin/env bash
set -e      # stop at the first error instead of carrying on

# system libs: headless rendering for the sim + ffmpeg for video
apt-get update && apt-get install -y libegl1 libgl1 ffmpeg

# torch for CUDA 12.8, installed BEFORE lerobot so lerobot doesn't pick its own
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# lerobot + the sim + the transformers pin
pip install lerobot gym-aloha 'transformers>=4.57.1,<5.0'

# the two checks that matter
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
nproc       # how many CPU cores you got, for --num_workers
