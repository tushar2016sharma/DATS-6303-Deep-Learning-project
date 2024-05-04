import os
import json
import music21 as m21
import numpy as np
import torch
import torch.nn.functional as F
import shutil

def get_default_base_path():
    if os.name == 'nt':  # Windows
        return os.getenv('USERPROFILE', 'C:\\Users\\Default')
    else:
        return os.getenv('HOME', '/home/default')

base_path = get_default_base_path()
print(base_path)
KERN_DATASET_PATH = os.path.join(base_path, 'MelodyGenerator', 'Data', 'Songs', 'melodies')
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5, # 8th note
    0.75,
    1.0, # quarter note
    1.5,
    2, # half note
    3,
    4 # whole note
]

def transpose(song):
    major_keys = [m21.key.Key(n) for n in m21.scale.MajorScale().getPitches('C', 'B')]
    minor_keys = [m21.key.Key(n, 'minor') for n in m21.scale.MinorScale().getPitches('C', 'B')]
    all_keys = major_keys + minor_keys

    transposed_songs = []
    for key in all_keys:
        interval = m21.interval.Interval(song.analyze('key').tonic, key.tonic)
        transposed_song = song.transpose(interval)
        transposed_songs.append((transposed_song, key.name))

    return transposed_songs


def prepare_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Removed existing directory: {dir_path}")

    os.makedirs(dir_path)
    print(f"Created new directory: {dir_path}")

def preprocess(dataset_path):
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    prepare_directory(SAVE_DIR)
    for i, song in enumerate(songs):
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        song = transpose(song)
        encoded_song = encode_all_transposed_songs(song)
        save_path = os.path.join(SAVE_DIR, str(i))
        for encoded_song, key in encoded_song:
            save_path = os.path.join(SAVE_DIR, f"{i}_{key}.txt")
            with open(save_path, "w") as fp:
                fp.write(encoded_song)

        if i % 10 == 0:
            print(f"Song {i} out of {len(songs)} processed")




def generate_training_sequences(sequence_length, batch_size):
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    num_sequences = len(int_songs) - sequence_length
    vocabulary_size = len(set(int_songs))

    # Generate data in batches
    for start_idx in range(0, num_sequences, batch_size):
        end_idx = min(start_idx + batch_size, num_sequences)
        batch_inputs = []
        batch_targets = []

        for i in range(start_idx, end_idx):
            batch_inputs.append(int_songs[i:i + sequence_length])
            batch_targets.append(int_songs[i + sequence_length])

        # Convert inputs to one-hot encoding
        batch_inputs = F.one_hot(torch.tensor(batch_inputs), num_classes=vocabulary_size).float()
        batch_targets = torch.tensor(batch_targets)

        yield batch_inputs, batch_targets

def main():
    preprocess(KERN_DATASET_PATH)
    file_path = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(file_path, MAPPING_PATH)

if __name__ == "__main__":
    main()
