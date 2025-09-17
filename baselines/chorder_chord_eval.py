import chorder
from miditoolkit import MidiFile
from pretty_midi import PrettyMIDI
from baselines.exported_midi_chord_recognition.complex_chord import get_scale_and_suffix

QUALITY_MAPPING = {
    'M': ':maj',
    'm': ':min',
    'o': ':dim',
    '+': ':aug',
    '7': ':7',
    'M7': ':maj7',
    'm7': ':min7',
    'o7': ':dim7',
    '/o7': ':hdim7',
    'sus2': ':sus2',
    'sus4': ':sus4',
}

RELATIVE_BASS_TEXT = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', '#5', '6', 'b7', '7']

def standardize_chord_label(chord_struct):
    simple_text = chord_struct.simple_text()
    if simple_text.startswith('N'):
        return 'N'
    scale, suffix = chord_struct.simple_text().split('_')
    scale_id, _ = get_scale_and_suffix(scale)
    bass_id, _ = get_scale_and_suffix(chord_struct.bass())
    suffix = QUALITY_MAPPING[suffix]
    result = scale + suffix
    if bass_id != scale_id:
        relative_bass = (bass_id - scale_id) % 12
        result += '/' + RELATIVE_BASS_TEXT[relative_bass]
    return result

def extract_chord_chorder(midi_path):
    midi_obj = MidiFile(midi_path)
    midi_beats = PrettyMIDI(midi_path).get_beats()
    decoder = chorder.Dechorder()
    result = decoder.dechord(midi_obj, scale=None)
    lab_result = []
    prev_chord_label = None
    prev_chord_onset = None
    for i in range(len(result) + 1):
        if i >= len(midi_beats):
            chord_label = None
            beat_time = midi_beats[-1]
        else:
            chord_struct = result[i]
            chord_label = standardize_chord_label(chord_struct)
            beat_time = midi_beats[i]
        if chord_label != prev_chord_label:
            if beat_time != prev_chord_onset and prev_chord_label is not None:
                lab_result.append((prev_chord_onset, beat_time, prev_chord_label))
            prev_chord_label = chord_label
            prev_chord_onset = beat_time
    # for onset, offset, chord in lab_result:
    #     print(f'{onset:.2f}\t{offset:.2f}\t{chord}')
    return lab_result

if __name__ == '__main__':
    extract_chord_chorder('../input/RM-P003.SMF_SYNC_BPM110.MID')