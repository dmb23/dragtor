#!/bin/bash

echo "Installing dependencies..."

# Install FFmpeg
brew install ffmpeg

# Install PyTorch (optimized for macOS)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/macos
