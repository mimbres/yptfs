import numpy as np
import os.path
import re


def extract_instruments(input_file):
    # Get the base filename without path and extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Extract numbers separated by underscores
    numbers = re.findall(r'\d+', base_name)

    # Convert to integers
    return [int(num) for num in numbers]


def create_instrument_mapping(instr_info):
    # Original MT3_FULL mapping for reference to get instrument names
    MT3_FULL = {
        "Acoustic Piano": [0, 1, 3, 6, 7],
        "Electric Piano": [2, 4, 5],
        "Chromatic Percussion": np.arange(8, 16).tolist(),
        "Organ": np.arange(16, 24).tolist(),
        "Acoustic Guitar": np.arange(24, 26).tolist(),
        "Clean Electric Guitar": np.arange(26, 29).tolist(),
        "Distorted Electric Guitar": np.arange(29, 32).tolist(),
        "Acoustic Bass": [32, 35],
        "Electric Bass": [33, 34, 36, 37, 38, 39],
        "Violin": [40],
        "Viola": [41],
        "Cello": [42],
        "Contrabass": [43],
        "Orchestral Harp": [46],
        "Timpani": [47],
        "String Ensemble": [48, 49, 44, 45],
        "Synth Strings": [50, 51],
        "Choir and Voice": [52, 53, 54],
        "Orchestra Hit": [55],
        "Trumpet": [56, 59],
        "Trombone": [57],
        "Tuba": [58],
        "French Horn": [60],
        "Brass Section": [61, 62, 63],
        "Soprano/Alto Sax": [64, 65],
        "Tenor Sax": [66],
        "Baritone Sax": [67],
        "Oboe": [68],
        "English Horn": [69],
        "Bassoon": [70],
        "Clarinet": [71],
        "Flute": [73, 72, 74, 75, 76, 77, 78, 79],
        "Synth Lead": np.arange(80, 88).tolist(),
        "Synth Pad": np.arange(88, 96).tolist(),
    }

    # Create a mapping from MIDI program to instrument name
    program_to_name = {}
    for name, programs in MT3_FULL.items():
        for program in programs:
            program_to_name[program] = name

    # Initialize the new mapping dictionary with instrument names from instr_info
    custom_mapping = {}
    for instr in instr_info:
        if not 0 <= instr <= 96:
            continue  # Skip invalid MIDI program numbers
        name = program_to_name.get(instr, f"Unknown_{instr}")
        custom_mapping[name] = []

    # Check if Oboe or Flute is present in instr_info
    has_oboe_or_flute = any(program_to_name.get(instr) in ["Oboe", "Flute"] for instr in instr_info)

    # For each possible MIDI program number (0 to 96) plus 100 and 101
    for program in list(range(97)) + [100, 101]:
        # Special handling for 100 and 101
        if program in [100, 101]:
            if has_oboe_or_flute:
                # Assign to Oboe or Flute if present
                for instr in instr_info:
                    if program_to_name.get(instr) in ["Oboe", "Flute"]:
                        name = program_to_name.get(instr)
                        custom_mapping[name].append(program)
                        break
                continue
            # If no Oboe or Flute, proceed to closest instrument

        # Find the closest instrument from instr_info
        min_distance = float('inf')
        closest_instr = None

        for instr in instr_info:
            if not 0 <= instr <= 96:
                continue  # Skip invalid MIDI program numbers
            distance = abs(program - instr)
            if distance < min_distance:
                min_distance = distance
                closest_instr = instr

        # Assign the program to the closest instrument's category
        if closest_instr is not None:
            name = program_to_name.get(closest_instr, f"Unknown_{closest_instr}")
            custom_mapping[name].append(program)

    # Ensure the representative MIDI program (from instr_info) is the first in the list
    for name, programs in custom_mapping.items():
        # Find the representative MIDI program for this instrument
        representative = None
        for instr in instr_info:
            if name == program_to_name.get(instr, f"Unknown_{instr}"):
                representative = instr
                break
        if representative is not None and representative in programs:
            # Remove the representative and add it to the front
            programs.remove(representative)
            programs.insert(0, representative)
        # Sort the remaining programs
        custom_mapping[name] = [programs[0]] + sorted(programs[1:]) if programs else []

    # Remove empty categories
    custom_mapping = {k: v for k, v in custom_mapping.items() if v}

    return custom_mapping
