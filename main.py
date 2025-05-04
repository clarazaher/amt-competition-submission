#!/opt/homebrew/bin/python3
"""
Name: main.py
Purpose: Transcribes MP3 files into MIDI files through the PyTorch implementation of MT3.
         Also downloads the model from Google Drive if needed.
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"

import argparse
import os
import subprocess
import sys
from inference import InferenceHandler

def download_model_from_drive(model_path):
    # For Google Drive, convert the shareable link to a direct download URL.
    # The file id for the model is "1dFXGiHdyEPVy9_KA853zRfWHg8zMCZEr"
    # url = "https://docs.google.com/uc?id=1dFXGiHdyEPVy9_KA853zRfWHg8zMCZEr&export=download&confirm=t"
    url = "https://drive.usercontent.google.com/download?id=1dFXGiHdyEPVy9_KA853zRfWHg8zMCZEr&export=download&authuser=0&confirm=t&uuid=68d01d44-7b6f-4a5f-b654-ed352b1a6ebf&at=APcmpoxShTD2RP3FeWuKdyu_Z9DE:1746390935350"
    try:
        subprocess.run(["curl", "-L", url, "-o", model_path], check=True)
    except subprocess.CalledProcessError:
        print("Failed to download model using curl.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP3 files into MIDI using MT3")
    parser.add_argument('-i', '--input', required=True, help='Path to input MP3 file')
    parser.add_argument('-o', '--output', required=True, help='Path to output MIDI file')
    args = parser.parse_args()

    pretrained_dir = './pretrained'
    os.makedirs(pretrained_dir, exist_ok=True)
    model_path = os.path.join(pretrained_dir, 'mt3.pth')

    if not os.path.exists(model_path):
        download_model_from_drive(model_path)
    else:
        print(f"{model_path} already exists. Skipping model download.")

    handler = InferenceHandler(pretrained_dir)
    handler.inference(args.input, args.output)

if __name__ == "__main__":
    main()
