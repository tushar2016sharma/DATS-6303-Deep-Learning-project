import json
import numpy as np
import torch
import torch.nn.functional as F
import music21 as m21
from preprocess2 import SEQUENCE_LENGTH, MAPPING_PATH
from train_LSTM_MultiHeadAttention import MusicModel


class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, input_size, hidden_size, output_size, model_path="best_model.pth"):
        """Constructor that initializes PyTorch model"""
        self.model = MusicModel(input_size, hidden_size, output_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            seed = seed[-max_sequence_length:]

            onehot_seed = F.one_hot(torch.tensor(seed), num_classes=len(self._mappings)).float()
            onehot_seed = onehot_seed.unsqueeze(0)

            with torch.no_grad():
                output = self.model(onehot_seed).squeeze(0)
                probabilities = torch.softmax(output, dim=0).numpy()

            output_int = self._sample_with_temperature(probabilities, temperature)
            seed.append(output_int)
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            if output_symbol == "/":
                break

            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilities, temperature, top_k=10):
        probabilities = np.log(probabilities + 1e-10) / temperature
        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))

        # Applying top-k sampling
        top_k = min(top_k, len(probabilities))
        indices_to_remove = probabilities < np.sort(probabilities)[-top_k]
        probabilities[indices_to_remove] = 0
        probabilities /= np.sum(probabilities)

        choices = np.arange(len(probabilities))
        index = np.random.choice(choices, p=probabilities)
        return index

    def _sample_with_temperature(self, probabilities, temperature):
        epsilon = 1e-10
        probabilities = np.log(probabilities + epsilon) / temperature
        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)
        return index

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="melody.mid"):
        """Converts a melody into a MIDI file"""
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            if symbol != "_" or i + 1 == len(melody):
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                    stream.append(m21_event)
                    step_counter = 1
                start_symbol = symbol
            else:
                step_counter += 1

        stream.write(format, file_name)


if __name__ == "__main__":
    with open('mapping.json', 'r') as file:
        data = json.load(file)
        size = len(data)

    print("Number of top-level keys:", size)
    OUTPUT_UNITS = size
    input_size = size
    hidden_size = 128
    output_size = size
    mg = MelodyGenerator(input_size, hidden_size, output_size)
    seed = "60 _ 64 _ 67 _ 60 _ _ 67 64 _ 67 _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
    print(melody)
    mg.save_melody(melody)