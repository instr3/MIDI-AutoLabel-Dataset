from mir.io.feature_io_base import *
from pretty_midi_fix import UglyMIDI

class MidiIO(FeatureIO):
    def read(self, filename, entry):
        midi_data = UglyMIDI(filename)
        return midi_data

    def write(self, data, filename, entry):
        data.write(filename)

    def visualize(self, data, filename, entry, override_sr):
        data.write(filename)

    def get_visualize_extention_name(self):
        return "mid"
