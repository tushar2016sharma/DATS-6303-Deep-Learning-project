import streamlit as st
from melodygenerator import MelodyGenerator
from music21 import environment, stream, note, pitch, tempo
import torch
import subprocess
import os

# Setup page configuration and title
SEQUENCE_LENGTH = 64


# Set device for model computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment for MuseScore path
env = environment.Environment()
env['musescoreDirectPNGPath'] = '/usr/bin/musescore'

# Define global settings
input_size, hidden_size, output_size = 38, 128, 38
model_path = "model.pth"
sound_font_path = '/path/to/your/soundfont.sf2'  # Adjust this to your actual sound font path
# Set title
st.set_page_config(
    page_title="Melody Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# This is a wide-mode music generation app. Enjoy creating melodies!"
    }
)

st.title("Welcome to the AI Music Generator!")

# Set title
 # Ensure this path is correct


def midi_number_to_note_name(midi_number):
    """Converts MIDI number to a music21 note name, returns None if it's a rest placeholder."""
    if midi_number.lower() == 'r':  # Check if the input is 'r' for rest
        return None  # Return None for rests to handle them separately in the calling function
    p = pitch.Pitch()
    p.midi = int(midi_number)  # Convert to integer if it's not 'r'
    return p.nameWithOctave

def text_to_score(text_melody, bpm=120):
    """Converts a space-separated string of MIDI numbers, 'r' for rests, and '_' for continuation into a music21 score."""
    s = stream.Score(id='mainScore')
    p = stream.Part(id='part')
    s.append(p)

    # Set the tempo
    mm = tempo.MetronomeMark(number=bpm)
    s.insert(0, mm)  # Insert the tempo mark at the beginning of the score

    melody_elements = text_melody.split()  # Split the string into components
    last_element = None

    for element in melody_elements:
        if element == '_':
            if last_element and isinstance(last_element, note.Note):
                # Extend the previous note
                last_element.quarterLength += 0.25  # Extend by quarter note duration
            elif last_element and isinstance(last_element, note.Rest):
                # Extend the previous rest
                last_element.quarterLength += 0.25
        elif element == 'r':
            # Add a rest
            rest = note.Rest(quarterLength=0.25)
            p.append(rest)
            last_element = rest
        else:
            note_name = midi_number_to_note_name(element)
            if note_name:
                n = note.Note(note_name, quarterLength=0.25)
                p.append(n)
                last_element = n

    return s




# Then in your Streamlit app
import os



import subprocess

def convert_midi_to_wav(midi_file_path, output_file_path, sound_font):
    command = [
        'fluidsynth',
        '-ni',
        sound_font,
        midi_file_path,
        '-F',
        output_file_path,
        '-r',
        '44100'
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error: {result.stderr.decode()}")
        return False
    return True


# Define path to your sound font file
sound_font_path = '/home/ubuntu/generating-melodies-with-rnn-lstm/9 - Converting Generated Melodies to MIDI//soundfont.sf2'






# Button and functionality to generate melody, image, and play sound
# Initialize session state variables if not already set




from music21 import environment, stream, note, pitch, converter, tempo
import subprocess
# Function to delete files if they exist
env = environment.Environment()
env['musescoreDirectPNGPath'] = '/usr/bin/musescore'
# Set up the environment for MuseScore

def delete_file_if_exists(file_path):
    """Removes the file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)

def save_score_image_with_xvfb(score, file_name="score.png"):
    """Uses xvfb and MuseScore to convert music XML to an image."""
    temp_musicxml = file_name.replace('.png', '.musicxml')
    score.write('musicxml', fp=temp_musicxml)
    delete_file_if_exists(file_name)  # Ensure the file does not exist before creating a new one
    command = ['xvfb-run', '-a', 'musescore', '-o',file_name, temp_musicxml]
    subprocess.run(command, check=True)
    delete_file_if_exists(temp_musicxml)

# Helper function to delete files if they exist
def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Convert MIDI to WAV
def convert_midi_to_wav(midi_file_path, output_file_path, sound_font):
    command = ['fluidsynth', '-ni', sound_font, midi_file_path, '-F', output_file_path, '-r', '44100']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        st.error(f"Error converting MIDI to WAV: {result.stderr.decode()}")
        return False
    return True

# Generate and display sheet music as an image
def save_score_image_with_xvfb(score, file_name="score.png"):
    temp_musicxml = file_name.replace('.png', '.musicxml')
    score.write('musicxml', fp=temp_musicxml)
    delete_file_if_exists(file_name)
    subprocess.run(['xvfb-run', '-a', 'musescore', '-o', file_name, temp_musicxml], check=True)
    delete_file_if_exists(temp_musicxml)

# Handle melody generation and display
def handle_melody_generation(seed_text):
    if seed_text:
        melody_generator = MelodyGenerator(input_size, hidden_size, output_size, model_path=model_path)
        melody_generator.model.to(device)
        melody = melody_generator.generate_melody(seed_text, 500, SEQUENCE_LENGTH, 0.3)
        generated_melody_str = ' '.join([str(m) for m in melody])

        # Save and display melody
        melody_generator.save_melody(melody, file_name="generated_melody.mid")

        convert_midi_to_wav("generated_melody.mid", "generated_melody.wav", sound_font_path)
        if os.path.exists("generated_melody.wav"):
            with open("generated_melody.wav", "rb") as wav_file:
                st.audio(wav_file.read(), format='audio/wav')
        else:
            st.error("Failed to convert MIDI to WAV.")

        # Display and download MIDI
        st.text_area("Generated Melody", generated_melody_str, height=150)
        st.download_button("Download MIDI", data=open("generated_melody.mid", "rb").read(), file_name="generated_melody.mid", mime="audio/midi")

        # Generate score image
        score = text_to_score(generated_melody_str)
        save_score_image_with_xvfb(score)
        st.image("score-1.png", caption="Generated Sheet Music")


# UI for seed melody input
seed_text = st.text_input("Enter a seed melody (notes separated by spaces):", value="67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _")
if st.button('Generate Melody'):
    handle_melody_generation(seed_text)
