import os
import json
import music21 as m21
import numpy as np
import torch
import torch.nn.functional as F


# THIS SCRIPT IS DESIGNED TO LOAD A TOTAL OF 1700 SONGS ('ERK' FOLDER INSIDE MAIN DATA FOLDER) ONLY

# Constants

def get_default_base_path():
    # Check operating system
    if os.name == 'nt':  # Windows
        # For Windows, typically use the USERPROFILE environment variable
        return os.getenv('USERPROFILE', 'C:\\Users\\Default')
    else:
        # For Unix-like systems (macOS, Linux), typically use the HOME environment variable
        return os.getenv('HOME', '/home/default')
# Set the base path using the function
base_path = get_default_base_path()
print(base_path)


KERN_DATASET_PATH = os.path.join(base_path, 'dataset_erk')

#KERN_DATASET_PATH = os.path.join("melodies_erk")  # path to erk melodies folder on my github folder

folder_name = "dataset_erk" 
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")
else:
    print(f"Folder '{folder_name}' already exists, loading songs from it.")

SAVE_DIR = "dataset_erk"
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

# Load the songs
def load_songs_in_kern(dataset_path):
    """Loads all kern pieces in dataset using music21. 

    :param dataset_path (str): Path to dataset
    :return songs (list of m21 streams): List containing all pieces
    """
    songs = []

    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # consider only kern files
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

# Check for acceptable duration as before
def has_acceptable_durations(song, acceptable_durations):
    """Boolean routine that returns True if piece has all acceptable duration, False otherwise.

    :param song (m21 stream):
    :param acceptable_durations (list): List of acceptable duration in quarter length
    :return (bool):
    """
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True



# MODIFIED 'TRANSPOSE' FUNCTION TO 'NOT' TRANSPOSE ALL THE MELODIES TO A SINGLE KEY 
# SUCH THAT THE MODEL COULD BE EXPOSED TO A WIDER RANGE OF KEYS TO LEARN FROM.
# THUS, IN THIS FUNCTION, WE JUST CHECK IF THE MELODY HAS A MAJOR OR MINOR TONALITY OR NOT.

def bypass_transpose(song):
  """Analyzes the song key and keeps it within 'major' or 'minor' tonality.

  :param song (m21 stream): Piece to analyze
  :return song (m21 stream): Original song (not transposed)
  """

  # Analyze key using music21
  key = song.analyze("key")

  # Check if key is not already major or minor
  if not (key.mode == "major" or key.mode == "minor"):
    print(f"Song {song} has a mode that is not major or minor. Skipping.")
    return None  

  # Keep the song in its original key 
  return song



def encode_song(song, time_step=0.25):
    """Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
    quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
    for representing notes/rests that are carried over into a new time step. Here's a sample encoding:

        ["r", "_", "60", "_", "_", "_", "72" "_"]

    :param song (m21 stream): Piece to encode
    :param time_step (float): Duration of each time step in quarter length
    :return:
    """

    encoded_song = []

    for event in song.flat.notesAndRests:

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # 60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # if it's the first time we see a note/rest, let's encode it. Otherwise, it means we're carrying the same
            # symbol in a new time step
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path):

    # load folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):

        # filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # bypass transpose but check for tonality only
        song = bypass_transpose(song)  

        # encode songs with music time series representation
        encoded_song = encode_song(song)

        # save songs to text file
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
    """Generates a file collating all the encoded songs and adding new piece delimiters.

    :param dataset_path (str): Path to folder containing the encoded songs
    :param file_dataset_path (str): Path to file for saving songs in single file
    :param sequence_length (int): # of time steps to be considered for training
    :return songs (str): String containing all songs in dataset + delimiters
    """

    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    # remove empty space from last character of string
    songs = songs[:-1]

    # save string that contains all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


def create_mapping(songs, mapping_path):
    """Creates a json file that maps the symbols in the song dataset onto integers

    :param songs (str): String with all songs
    :param mapping_path (str): Path where to save mapping
    :return:
    """
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save voabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    int_songs = []

    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # transform songs string to list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


# Function to convert the sequences into time series-compatible sequence (sliding window approach)
def generate_training_sequences(sequence_length):

    # retrieving the sequences
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    inputs, targets = [], []
    num_sequences = len(int_songs) - sequence_length
    
    # iterating over the count of sequences
    for i in range(num_sequences):
        inputs.append(int_songs[i : i + sequence_length])
        targets.append(int_songs[i + sequence_length])
    vocabulary_size = len(set(int_songs))
    
    # conversion into one-hot encodings using torch 
    inputs = F.one_hot(torch.tensor(inputs), num_classes = vocabulary_size).float()
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


