from pretty_midi_fix import UglyMIDI
from settings import LA_DATASET_PATH
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

FILE_LIST = []

def analyze_tempo_changes(midi_path):
    try:
        midi = UglyMIDI(midi_path)
    except:
        return
    tempo_changes, tempi = midi.get_tempo_changes()
    return len(tempo_changes) >= 3
        

def analyze_key_changes(midi_path):
    try:
        midi = UglyMIDI(midi_path)
    except:
        return
    if midi.get_end_time() > 3600:
        # Too long
        return False
    if len(midi.key_signature_changes) >= 2:
        return True
    if len(midi.key_signature_changes) == 1:
        key_number = midi.key_signature_changes[0].key_number
        if key_number >= 12: # minor keys
            return True
    return False

def process_file(file):
    tempo_result = 0 # 1 if analyze_tempo_changes(file) else 0
    key_result = 1 if analyze_key_changes(file) else 0
    return tempo_result, key_result


if __name__ == '__main__':
    for folder in os.listdir(os.path.join(LA_DATASET_PATH, 'MIDIs')):
        FILE_LIST.extend([os.path.join(LA_DATASET_PATH, 'MIDIs', folder, file)
            for file in os.listdir(os.path.join(LA_DATASET_PATH, 'MIDIs', folder))])
    print(len(FILE_LIST))
    # Randomly sample 1000 files
    # np.random.seed(42)
    # FILE_LIST = np.random.choice(FILE_LIST, 1000, replace=False)
    results = Parallel(n_jobs=-1)(delayed(process_file)(file) for file in tqdm(FILE_LIST, desc='Processing files'))
    
    # has_tempo_changes = sum([result[0] for result in results])
    # has_key_changes = sum([result[1] for result in results])
    key_change_file_list = [os.path.basename(FILE_LIST[i]) for i, result in enumerate(results) if result[1] == 1]
    f = open('data/key_change_file_list.txt', 'w')
    f.write('\n'.join(key_change_file_list))
    f.close()
    # print(has_tempo_changes, has_key_changes)