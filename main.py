#!/opt/homebrew/bin/python3
"""
Name: main.py
Purpose: Transcribes MP3 files into MIDI files through the PyTorch implementation of MT3.
         Also pulls assets from Git LFS in the pretrained directory if needed.
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"

import argparse
import os
import subprocess
import sys
from inference import InferenceHandler

def pull_from_github_lfs():
    try:
        subprocess.run(
            ["git", "lfs", "pull"],
            cwd="https://github.com/clarazaher/amt-competition-submission.git",
            check=True,
        )
    except subprocess.CalledProcessError:
        print(f"Failed to pull from Git LFS.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Transcribe MP3 files into MIDI using MT3")
    parser.add_argument('-i', '--input', required=True, help='Path to input MP3 file')
    parser.add_argument('-o', '--output', required=True, help='Path to output MIDI file')
    args = parser.parse_args()

    pretrained_dir = './pretrained'
    model_path = os.path.join(pretrained_dir, 'mt3.pth')
    
    if not os.path.exists(model_path):
        pull_from_github_lfs()
    else:
        print(f"{model_path} already exists. Skipping Git LFS pull.")

    handler = InferenceHandler(pretrained_dir)
    handler.inference(args.input, args.output)

if __name__ == "__main__":
    main()
