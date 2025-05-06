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
from cfg_local import model_name_to_log_path, cfg_local
from run_amt import download_weights, get_amt_model
from instr_remap import create_instrument_mapping, extract_instruments

DEFAULT_MODEL_NAME = cfg_local.deafult_model_name
DEFAULT_INPUT_FILE = "./test_input/_0.mp3"


def main():
    parser = argparse.ArgumentParser(description="Load MT3 model")
    parser.add_argument('-i',
                        '--input',
                        default=DEFAULT_INPUT_FILE,
                        help=f'Input MP3 file (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('-o', '--output', help='Output MIDI file (default: input with .midi extension)')
    parser.add_argument('-m',
                        '--model',
                        default=DEFAULT_MODEL_NAME,
                        help=f'Model name or index (default: {DEFAULT_MODEL_NAME})')
    args = parser.parse_args()

    # Set output to input with .midi extension if not provided
    args.output = args.output or os.path.splitext(args.input)[0] + '.midi'
    if not os.path.dirname(args.output):  # If no directory is specified, use current directory
        args.output = os.path.join('.', args.output)

    model_names = list(model_name_to_log_path.keys())
    model_name = args.model
    if args.model.isdigit():
        model_index = int(args.model)
        if 0 <= model_index < len(model_names):
            model_name = model_names[model_index]
        else:
            parser.error(
                f"Invalid index {model_index}. Available: {', '.join(f'{i}: {n}' for i, n in enumerate(model_names))}")
    elif model_name not in model_names:
        parser.error(
            f"Invalid model '{model_name}'. Available: {', '.join(f'{i}: {n}' for i, n in enumerate(model_names))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = model_name_to_log_path[model_name]["log_path"]
    log_s3_uri = model_name_to_log_path[model_name]["log_s3_uri"]

    # Download and load the model
    download_weights(log_path, log_s3_uri, recursive=True)
    model, amt_func = get_amt_model(device, cfg_local, model_name=model_name, logger=None)
    print(f"Model '{model_name}' loaded.")

    # Set output vocabulary based on test_vocab (optionally based on input file name)
    output_vocab = None
    if cfg_local.test_vocab == "use_instr_info":
        output_vocab = create_instrument_mapping(extract_instruments(args.input))
    elif cfg_local.test_vocab:
        from mockmt3.config.vocabulary import program_vocab_presets
        output_vocab = program_vocab_presets[str(cfg_local.test_vocab)]

    # Update model vocab
    model.output_vocab = model.midi_output_vocab = output_vocab
    if output_vocab:
        from mockmt3.utils.utils import create_inverse_vocab
        model.midi_output_inverse_vocab = create_inverse_vocab(output_vocab)
        print(f"Model vocab updated to '{output_vocab}'.")

    # Transcription
    os.path.exists(args.input) or parser.error(f"Input file '{args.input}' does not exist.")
    # os.makedirs(os.path.dirname(cfg_local.amt_output_midi_dir), exist_ok=True)  # temp output dir
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    amt_func(model, device, args.input, output_midi_file=args.output)
    print(f"Transcription completed. Output saved to '{args.output}'.")


if __name__ == "__main__":
    main()
