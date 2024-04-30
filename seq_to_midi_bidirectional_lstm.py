import json
import numpy as np
import torch
import torch.nn.functional as F
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
from train_torch import OUTPUT_UNITS, NUM_UNITS, LSTMModel


class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, model_path="model_new.pt"):
        """Constructor that initialises PyTorch model"""

        self.model_path = model_path
        self.model = LSTMModel(OUTPUT_UNITS, NUM_UNITS)
        self.model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu'))) 
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
                output = self.model(onehot_seed).squeeze(0)           # Get output from the model
                probabilities = torch.softmax(output, dim=0).numpy()  # Apply softmax correctly

            output_int = self._sample_with_temperature(probabilities, temperature)
            seed.append(output_int)
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            if output_symbol == "/":
                break

            melody.append(output_symbol)

        return melody      
      

    def _sample_with_temperature(self, probabilites, temperature):
        """Samples an index from a probability array reapplying softmax using temperature

        :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return index (int): Selected output symbol
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index


    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        """Converts a melody into a MIDI file

        :param melody (list of str):
        :param min_duration (float): Duration of each time step in quarter length
        :param file_name (str): Name of midi file
        :return:
        """

        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    #seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    seed = "59 _ 61 _ 62 _ _ 66 67 _ 69 _ _"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
    print(melody)
    mg.save_melody(melody)




















