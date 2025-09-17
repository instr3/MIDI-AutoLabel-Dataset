import os
from settings import NOTTINGHAM_DATASET_PATH
import shutil
from pretty_midi_fix import UglyMIDI

output_dir = os.path.join(NOTTINGHAM_DATASET_PATH, 'processed')


COLLECTIONS = {}

def preprocess_collection_files():
    files = os.listdir(os.path.join(NOTTINGHAM_DATASET_PATH, 'ABC_cleaned'))
    for file in files:
        if not file.endswith('.abc'):
            continue
        collection_name = file[:-4]
        f = open(os.path.join(NOTTINGHAM_DATASET_PATH, 'ABC_cleaned', file), 'r')
        lines = [line.strip() for line in f.readlines() if line.strip()]
        f.close()
        collection_dict = {}
        current_id = None
        for line in lines:
            if line.startswith('X:'):
                if len(lines) > 0 and current_id is not None:
                    collection_dict[current_id]['raw'] = '\n'.join(lines)
                current_id = line[2:].strip()
                collection_dict[current_id] = {}
                lines = []
            elif line.startswith('K:') and current_id is not None:
                key = line[2:].strip()
                if 'key' in collection_dict[current_id] and collection_dict[current_id]['key'] != key:
                    # Multiple keys, mark as unknown
                    collection_dict[current_id]['key'] = 'unknown'
                else:
                    collection_dict[current_id]['key'] = key
            lines.append(line)
        COLLECTIONS[collection_name] = collection_dict

def validate_key_label(midi_path):
    name = os.path.basename(midi_path)[:-4]
    # split 'ashover19' into 'ashover' and '19'
    p = len(name) - 1
    while p >= 0 and name[p].isdigit():
        p -= 1
    collection_name = name[:p + 1]
    collection_id = name[p + 1:]
    data = COLLECTIONS[collection_name][collection_id]
    key = data['key']
    if key == 'unknown':
        return None
    else:
        if key in ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']:
            key += ' major'
        elif key in ['Cm', 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm', 'Gbm', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm']:
            key = key[:-1] + ' minor'
        else:
            print(key)
            assert False
        return key

def is_polyphonic_song(midi_path):
    try:
        midi = UglyMIDI(midi_path)
    except:
        print(f'Could not load {midi_path}')
        return False
    piano_roll = midi.instruments[0].get_piano_roll(fs=4)  # 4 frames per second (16th notes)
    # Check if any time frame has more than one note
    polyphonic = (piano_roll > 0).sum(axis=0) > 1
    return polyphonic.any()


def create_key_dataset():
    preprocess_collection_files()
    key_path = os.path.join(output_dir, 'keys')
    os.makedirs(key_path, exist_ok=True)
    for file in os.listdir(os.path.join(NOTTINGHAM_DATASET_PATH, 'MIDI')):
        if not file.endswith('.mid'):
            continue
        key = validate_key_label(file)
        if key is not None:
            # Copy the file to the output directory
            f = open(os.path.join(key_path, file[:-4] + '.lab'), 'w')
            f.write(key)
            f.close()

def create_chord_dataset():
    preprocess_collection_files()
    chord_path = os.path.join(output_dir, 'chords')
    os.makedirs(chord_path, exist_ok=True)
    for file in os.listdir(os.path.join(NOTTINGHAM_DATASET_PATH, 'MIDI')):
        if not file.endswith('.mid'):
            continue
        if is_polyphonic_song(os.path.join(NOTTINGHAM_DATASET_PATH, 'MIDI', file)):
            print(file)


if __name__ == '__main__':
    create_chord_dataset()
