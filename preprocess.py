import os
import json
import music21 as m21
import numpy as np
import torch
import torch.nn.functional as F

KERN_DATASET_PATH = "/home/ubuntu/generating-melodies-with-rnn-lstm/3Preprocessing_dataset_for_melody_generation_pt_1/code/deutschl/test/"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

# durations are expressed in quarter length
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

def load_songs_in_kern(dataset_path):
    songs = []
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song):
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0].KeySignature()

    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    tonic = key.tonic
    if key.mode == "major":  # Ionian
        interval = m21.interval.Interval(tonic, m21.pitch.Pitch("C"))
    elif key.mode == "dorian":
        interval = m21.interval.Interval(tonic, m21.pitch.Pitch("D"))
    elif key.mode == "phrygian":
        interval = m21.interval.Interval(tonic, m21.pitch.Pitch("E"))
    elif key.mode == "lydian":
        interval = m21.interval.Interval(tonic, m21.pitch.Pitch("F"))
    elif key.mode == "mixolydian":
        interval = m21.interval.Interval(tonic, m21.pitch.Pitch("G"))
    elif key.mode == "minor":  # Aeolian
        interval = m21.interval.Interval(tonic, m21.pitch.Pitch("A"))
    elif key.mode == "locrian":
        interval = m21.interval.Interval(tonic, m21.pitch.Pitch("B"))

    transposed_song = song.transpose(interval)
    return transposed_song

def encode_song(song, time_step=0.25):
    encoded_song = []
    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song

def adjust_durations(song, duration_factor):
    """
    Adjust the durations of notes and rests in a song by a specified factor.
    A duration_factor of 2.0 would double the duration of all notes and rests,
    while a factor of 0.5 would halve them.
    """
    for note in song.flat.notesAndRests:
        note.duration.quarterLength *= duration_factor
    return song


def preprocess(dataset_path):
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    for i, song in enumerate(songs):
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        song = transpose(song)
        # song = adjust_durations(song, 0.5)
        encoded_song = encode_song(song)
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)
        if i % 10 == 0:
            print(f"Song {i} out of {len(songs)} processed")

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs += song + " " + new_song_delimiter
    songs = songs[:-1]
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    return songs

def create_mapping(songs, mapping_path):
    mappings = {}
    songs = songs.split()
    vocabulary = list(set(songs))
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)

def convert_songs_to_int(songs):
    int_songs = []
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
    songs = songs.split()
    for symbol in songs:
        int_songs.append(mappings[symbol])
    return int_songs

def generate_training_sequences(sequence_length):
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    inputs = []
    targets = []
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    vocabulary_size = len(set(int_songs))
    inputs = F.one_hot(torch.tensor(inputs), num_classes=vocabulary_size).float()
    targets = torch.tensor(targets)
    print(f"There are {len(inputs)} sequences.")
    return inputs, targets

def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    #inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()
