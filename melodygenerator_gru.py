import json
import numpy as np
import torch
import torch.nn.functional as F
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
from train_gru import MusicModel


class MelodyGenerator:
    """Class to generate melodies using a pre-trained model based on LSTM or GRU."""

    def __init__(self, input_size, hidden_size, output_size,model_path="model.pth"):
        """Initialize the model and load the pre-trained weights."""
        self.model = MusicModel(input_size, hidden_size, output_size)
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # Set the model to evaluation mode
        except FileNotFoundError:
            print(f"Failed to load the model from {model_path}. Please check the path and try again.")
            raise

        try:
            with open(MAPPING_PATH, "r") as fp:
                self._mappings = json.load(fp)
        except FileNotFoundError:
            print(f"Failed to load mappings from {MAPPING_PATH}. Please check the path and try again.")
            raise

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature=1.0):
        """Generate a melody from a seed sequence."""
        seed = seed.split()
        melody = seed[:]
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

    def _sample_with_temperature(self, probabilities, temperature=1.0):
        """Sample an index from a probability array, adjusted by temperature."""
        probabilities = np.log(probabilities + 1e-10) / temperature
        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        choices = np.arange(len(probabilities))
        index = np.random.choice(choices, p=probabilities)
        return index

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="melody.mid"):
        """Save the generated melody to a MIDI file."""
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
        print(f"Melody saved to {file_name}")


if __name__ == "__main__":
    input_size = 38  # Example size, set appropriately
    hidden_size = 128  # Example size, set appropriately
    output_size = 38  # Example size, set appropriately
    mg = MelodyGenerator(input_size, hidden_size, output_size)
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
    print("Generated Melody:", melody)
    mg.save_melody(melody)
