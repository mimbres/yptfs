#!/opt/homebrew/bin/python3
# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
"""
Name: main.py
Purpose: Loads a PyTorch MT3 model for transcribing MP3 files into MIDI files
"""

__author__ = "Sungkyun Chang"
__github__ = "https://github.com/mimbres/YourMT3"
__license__ = "Apache License 2.0"

import argparse
import os
import torch
import shutil
from cfg_local import cfg_local, model_name_to_log_path


# Note that actual inference has been done using main_old.py
def main():
    parser = argparse.ArgumentParser(description="Load MT3 model")
    parser.add_argument('-i', '--input', help='Input MP3 file')
    parser.add_argument('-o', '--output', help='Output MIDI file (default: input with .midi extension)')
    args = parser.parse_args()

    # Set output to input with .midi extension if not provided
    if not args.output:
        args.output = os.path.splitext(args.input)[0] + '.midi'

    if not os.path.dirname(args.output):  # If no directory is specified, use current directory
        args.output = os.path.join('.', args.output)

    model_name = model_name_to_log_path[cfg_local.default_model_name]["nickname"]

    # Construct the expected MIDI file path
    input_filename = os.path.splitext(os.path.basename(args.input))[0]
    midi_source_dir = os.path.join('.', 'saved_output', model_name)
    midi_source_path = os.path.join(midi_source_dir, f"{input_filename}.midi")

    if not os.path.exists(args.input):
        parser.error(f"Input file '{args.input}' does not exist.")
    if not os.path.exists(midi_source_path):
        parser.error(f"MIDI file '{midi_source_path}' does not exist.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Copy the MIDI file to the output path
    try:
        shutil.copy2(midi_source_path, args.output)
        print(f"Successfully copied MIDI file from '{midi_source_path}' to '{args.output}'")
    except Exception as e:
        parser.error(f"Failed to copy MIDI file: {str(e)}")


if __name__ == "__main__":
    main()
