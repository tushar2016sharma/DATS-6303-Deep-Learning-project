import json
import numpy as np
import torch
import torch.nn.functional as F
import music21 as m21
from preprocess_final_vae import SEQUENCE_LENGTH, MAPPING_PATH
from train_final_vae import VAE  

class MelodyGenerator:
    def __init__(self, model_path="vae_model.pth"):
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
        
        self.input_dim = len(self._mappings)
        self.model = VAE(self.input_dim, 256, 50, self.input_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self._mappings_inverse = {v: k for k, v in self._mappings.items()}

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        seed = seed.split()
        melody = seed
        seed = [self._mappings.get(symbol, self._mappings["_"]) for symbol in seed]

        # Initialize melody indices with valid boundary checks
        melody_indices = [idx if idx < self.input_dim else self._mappings["_"] for idx in seed]

        for _ in range(num_steps):
            seed_tensor = F.one_hot(torch.tensor([melody_indices]), num_classes=self.input_dim).float()
            
            with torch.no_grad():
                recon_x, _, _ = self.model(seed_tensor)
                probabilities = torch.softmax(recon_x[0], dim=0).numpy()

            output_int = self._sample_with_temperature(probabilities, temperature)

            # Validate output index before appending
            if output_int >= self.input_dim:
                continue
            
            melody_indices.append(output_int)
            melody_indices = melody_indices[-max_sequence_length:]  # Maintain fixed length

            output_symbol = self._mappings_inverse.get(output_int, "_")
            if output_symbol == "/":
                break

        return ' '.join(self._mappings_inverse.get(idx, "_") for idx in melody_indices)


    def _sample_with_temperature(self, probabilities, temperature):
        probabilities = probabilities.flatten()  # Ensure the array is flattened
        probabilities = np.clip(probabilities, 1e-10, None)  # Clip to avoid log(0)
        probabilities = np.exp(np.log(probabilities) / temperature)
        probabilities /= np.sum(probabilities)  # Normalize
        return np.random.choice(len(probabilities), p=probabilities)

    def save_melody(self, melody, step_duration=1, format="midi", file_name="generated_melody1.mid"):
        stream = m21.stream.Stream()
        for symbol in melody.split():
            if symbol.isdigit():
                pitch = m21.pitch.Pitch()
                pitch.midi = int(symbol)
                note = m21.note.Note(pitch.nameWithOctave, quarterLength=step_duration)
                stream.append(note)
            elif symbol == "r":
                rest = m21.note.Rest(quarterLength=step_duration)
                stream.append(rest)
        stream.write(format, file_name)

if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "65 _ _ _ 67 _ _ _ 69 _ _ _ 70 _ _ _ 72 _ _ _ 74 _ _ _ 76 _ _ _ 77"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
    print("Generated Melody:", melody)
    mg.save_melody(melody)
