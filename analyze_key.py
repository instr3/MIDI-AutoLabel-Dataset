import matplotlib.pyplot as plt

from pretty_midi_fix import UglyMIDI
from settings import LA_DATASET_PATH, RWC_DATASET_PATH
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from preprocess_large_midi_dataset import filter_la_quantization, preprocess_midi
import torch
import pretty_midi
from mir_eval.chord import pitch_class_to_semitone
import mir_eval.chord

FILE_LIST = []


def midi_to_pentatonic(midi: UglyMIDI, override_keys=None):
    key_changes = []
    for key_change in midi.key_signature_changes:
        key_changes.append([key_change.time, pretty_midi.key_number_to_key_name(key_change.key_number)])
    if override_keys is not None:
        key_changes = override_keys
    assert len(key_changes) > 0, "No key changes found in MIDI file."
    key_change_times = np.array([time for time, _ in key_changes])
    scale_maps = []
    base_scale_map = np.array([0, -1, 0, 1, 0, -1, 1, 0, -1, 0, -1, 1])
    for _, key_name in key_changes:
        scale, mode = key_name.split(' ')
        scale_semitones = pitch_class_to_semitone(scale)
        if mode == 'minor':
            scale_semitones = (scale_semitones + 3) % 12
        scale_maps.append(np.roll(base_scale_map, scale_semitones))
    def get_scale_map_by_time(time):
        index = np.searchsorted(key_change_times, time + 1e-3) - 1
        if index < 0:
            index = 0
        return scale_maps[index]
    for ins in midi.instruments:
        if ins.is_drum:
            continue
        else:
            new_notes = []
            for note in ins.notes:
                scale_map = get_scale_map_by_time(note.start)
                note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch + scale_map[note.pitch % 12],
                    start=note.start,
                    end=note.end
                )
                new_notes.append(note)
            ins.notes = new_notes
            ins.pitch_bends.clear()  # remove pitch bends
    return midi


def analyze_key_changes(midi_path):
    if isinstance(midi_path, UglyMIDI):
        midi = midi_path
    else:
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

def analyze_file(midi_file, max_polyphony=16, beat_div=4):
    try:
        midi = UglyMIDI(midi_file, constant_tempo=60.0 / beat_div)
    except:
        return None
    if not analyze_key_changes(midi):
        return None
    if not filter_la_quantization(midi):
        return None
    result1 = preprocess_midi(
        midi,
        max_polyphony=max_polyphony,
        beat_div=beat_div,
        filter=False # already filtered
    )
    if result1 is None:
        return None
    penta_midi = midi_to_pentatonic(midi)
    result2 = preprocess_midi(
        penta_midi,
        max_polyphony=max_polyphony,
        beat_div=beat_div,
        filter=False # already filtered
    )
    if result2 is None:
        return None
    (data1, pitch_shift_range1) = result1
    (data2, pitch_shift_range2) = result2
    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]
    pitch_shift_range = torch.tensor([max(pitch_shift_range1[0], pitch_shift_range2[0]), min(pitch_shift_range1[1], pitch_shift_range2[1])], dtype=pitch_shift_range1.dtype)
    return torch.cat((data1, data2), dim=1), pitch_shift_range

def create_key_dataset(max_polyphony, dataset_name, max_idx=None):
    f = open('data/key_change_file_list.txt', 'r')
    lines = [line.strip() for line in f.readlines() if line.strip()]
    f.close()
    midi_files = [os.path.join(LA_DATASET_PATH, 'MIDIs', line[0], line) for line in lines]
    if max_idx is not None:
        midi_files = midi_files[:max_idx]
    # Process files in parallel
    print(f'Processing {len(midi_files)} files')
    results = Parallel(n_jobs=-1)(delayed(analyze_file)(midi_file, max_polyphony=max_polyphony) for midi_file in tqdm(midi_files))
    # Filter out None results
    midi_files = [os.path.relpath(midi_files[i], os.path.join(LA_DATASET_PATH, 'MIDIs')) for i, result in enumerate(results) if result is not None]
    results = [result for result in results if result is not None]
    results_data = [result[0] for result in results]
    results_shift = [result[1] for result in results]
    # np.save(f'data/{dataset_name}.npy', np.concatenate(results, axis=0))
    torch.save(torch.cat(results_data, dim=0), f'data/{dataset_name}.pt')
    torch.save(torch.cat(results_shift, dim=0), f'data/{dataset_name}.pitch_shift_range.pt')
    lengths = [len(data) for data in results_data]
    # save midi file names
    f = open(f'data/{dataset_name}.txt', 'w')
    for i, midi_file in enumerate(midi_files):
        f.write(str(i) + '\t' + midi_file + '\n')
    # np.save(f'data/{dataset_name}.length.npy', np.array(lengths))
    torch.save(torch.tensor(lengths), f'data/{dataset_name}.length.pt')

def locate_tonal_scale(midi: UglyMIDI):
    SCALE_TEMPLATE = np.array([0, 2, 4, 5, 7, 9, 11])  # Major pentatonic scale
    evidences = []
    for i, ins in enumerate(midi.instruments):
        if ins.is_drum:
            continue
        for j, note in enumerate(ins.notes):
            if j + 7 <= len(ins.notes):
                # Inspect if 7 consecutive notes are like 1234567, in any order
                scales = np.zeros(12, dtype=int)
                for j2 in range(j, j + 7):
                    scales[ins.notes[j2].pitch % 12] += 1
                # Check if scales has 7 unique notes
                if np.sum(scales > 0) == 7:
                    # Check if the notes form a scale
                    for k in range(12):
                        if np.all(scales[(k + SCALE_TEMPLATE) % 12] > 0):
                            structure = [(ins.notes[j + x].pitch - k) % 12 for x in range(7)]
                            evidences.append([note.start, ins.notes[j + 6].end, k, structure, ins.name, i])
    evidences.sort(key=lambda x: x[0])
    return evidences


def midi_to_key_evidence(midi: UglyMIDI, context_window=192):
    end_time = int(np.ceil(midi.get_end_time()))
    labels = np.full(end_time, fill_value=-1, dtype=int)
    evidences = locate_tonal_scale(midi)
    if len(evidences) == 0:
        return None, -1, -1, -1
    ins_freq = np.zeros(len(midi.instruments), dtype=int)
    min_start_time = min([int(np.round(evidence[0])) for evidence in evidences])
    max_end_time = max([int(np.round(evidence[1])) for evidence in evidences])
    min_start_time = max(0, min_start_time - context_window)
    max_end_time = min(max_end_time, end_time + context_window)
    for start, end, root, _, _, ins_id in evidences:
        start_time = int(np.round(start))
        end_time = int(np.round(end))
        if end_time == start_time:
            end_time += 1
        labels[start_time:end_time] = root
        ins_freq[ins_id] += 1
    # Report the instruments that contain the most evidences (sorry, but we need to make it harder)
    max_freq_at = np.argmax(ins_freq)
    # Create a new ins that contains the labels
    ins = pretty_midi.Instrument(program=0, is_drum=False, name='Key Evidence')
    for i in range(len(labels)):
        if labels[i] != -1:
            ins.notes.append(pretty_midi.Note(velocity=100, pitch=labels[i], start=i, end=i + 1))
    return ins, max_freq_at, min_start_time, max_end_time

def analyze_file_evidence(midi_file, gt_label_file=None, max_polyphony=16, beat_div=4, visualize=False):
    try:
        midi = UglyMIDI(midi_file, constant_tempo=60.0 / beat_div)
    except:
        return None
    if not filter_la_quantization(midi):
        return None
    if gt_label_file is not None:
        f = open(gt_label_file, 'r')
        lines = [f.strip() for f in f.readlines() if f.strip()]
        f.close()
        perf_midi = UglyMIDI(midi_file)
        from pretty_midi_fix import get_time_mapping
        perf_to_score_mapping = get_time_mapping(perf_midi, midi)
        label_ins = pretty_midi.Instrument(program=0, is_drum=False, name='Key Evidence')
        for line in lines:
            start_time, end_time, label = line.split('\t')
            start_time = int(np.round(perf_to_score_mapping(float(start_time))))
            end_time = int(np.round(perf_to_score_mapping(float(end_time))))
            try:
                root, mode = label.strip().split(' ')
            except:
                print('Error parsing label:', label, 'in file', gt_label_file)
            root_id = mir_eval.chord.pitch_class_to_semitone(root)
            label_id = (root_id + {'major': 0, 'minor': 3}[mode]) % 12
            for i in range(start_time, end_time):
                label_ins.notes.append(pretty_midi.Note(velocity=100, pitch=label_id, start=i, end=i + 1))
        max_freq_at = None
        min_start_time = None
        max_end_time = None
    else:
        label_ins, max_freq_at, min_start_time, max_end_time = midi_to_key_evidence(midi)
    if label_ins is None:
        return None
    if max_freq_at is not None:
        midi.instruments[max_freq_at].notes.clear()  # bye the hint track
    result1 = preprocess_midi(
        midi,
        max_polyphony=max_polyphony,
        beat_div=beat_div,
        filter=False # already filtered
    )
    if result1 is None:
        return None
    if visualize:
        from copy import deepcopy
        midi_copy = deepcopy(midi)
        midi_copy.instruments.append(label_ins)
        file_path = os.path.join('temp', 'key_evidence', os.path.basename(midi_file))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        midi_copy.write(file_path)
    midi.instruments = [label_ins]
    result2 = preprocess_midi(
        midi,
        max_polyphony=1,
        beat_div=beat_div,
        filter=False # already filtered
    )
    if result2 is None:
        return None
    (data1, pitch_shift_range) = result1
    (data2, _) = result2  # we don't need pitch shift range for the label track
    min_length = min(len(data1), len(data2))
    data1 = data1[:min_length]
    data2 = data2[:min_length]
    # Cut the data to the range of the key evidence
    data = torch.cat((data1, data2), dim=1)
    if min_start_time is not None:
        data = data[min_start_time:max_end_time]
    return data, pitch_shift_range

def create_la_dataset(max_polyphony, dataset_name, max_idx=None, visualize=False, func_name='evidence'):
    func = {'evidence': analyze_file_evidence, 'cadence': analyze_file_cadence}[func_name]
    midi_files = []
    for folder in os.listdir(os.path.join(LA_DATASET_PATH, 'MIDIs')):
        midi_files.extend([os.path.join(LA_DATASET_PATH, 'MIDIs', folder, file)
            for file in os.listdir(os.path.join(LA_DATASET_PATH, 'MIDIs', folder))])
    if max_idx is not None:
        np.random.seed(42)
        midi_files = np.random.choice(midi_files, max_idx, replace=False)
    print(len(midi_files))
    # Process files in parallel
    print(f'Processing {len(midi_files)} files')
    results = Parallel(n_jobs=-1)(delayed(func)(midi_file, max_polyphony=max_polyphony, visualize=visualize) for midi_file in tqdm(midi_files))
    # Filter out None results
    midi_files = [os.path.relpath(midi_files[i], os.path.join(LA_DATASET_PATH, 'MIDIs')) for i, result in enumerate(results) if result is not None]
    results = [result for result in results if result is not None]
    results_data = [result[0] for result in results]
    results_shift = [result[1] for result in results]
    # np.save(f'data/{dataset_name}.npy', np.concatenate(results, axis=0))
    torch.save(torch.cat(results_data, dim=0), f'data/{dataset_name}.pt')
    torch.save(torch.cat(results_shift, dim=0), f'data/{dataset_name}.pitch_shift_range.pt')
    lengths = [len(data) for data in results_data]
    # save midi file names
    f = open(f'data/{dataset_name}.txt', 'w')
    for i, midi_file in enumerate(midi_files):
        f.write(str(i) + '\t' + midi_file + '\n')
    # np.save(f'data/{dataset_name}.length.npy', np.array(lengths))
    torch.save(torch.tensor(lengths), f'data/{dataset_name}.length.pt')

def create_rwc_dataset(max_polyphony, dataset_name, visualize=False, func_name='evidence'):
    func = {'evidence': analyze_file_evidence, 'cadence': analyze_file_cadence}[func_name]
    midi_files = []
    for id in range(100):
        midi_files.append([
            os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', f'RM-P{id + 1:03d}.SMF_SYNC.MID'),
            os.path.join(RWC_DATASET_PATH, 'keys_new', f'RM-P{id + 1:03d}.lab')
        ])
    print(len(midi_files))
    # Process files in parallel
    print(f'Processing {len(midi_files)} files')
    results = Parallel(n_jobs=1 if visualize else -1)(delayed(func)(midi_file, max_polyphony=max_polyphony, visualize=visualize, gt_label_file=label_file) for midi_file, label_file in tqdm(midi_files))
    # Filter out None results
    midi_files = [os.path.relpath(midi_files[i][0], os.path.join(LA_DATASET_PATH, 'MIDIs')) for i, result in enumerate(results) if result is not None]
    results = [result for result in results if result is not None]
    results_data = [result[0] for result in results]
    results_shift = [result[1] for result in results]
    # np.save(f'data/{dataset_name}.npy', np.concatenate(results, axis=0))
    torch.save(torch.cat(results_data, dim=0), f'data/{dataset_name}.pt')
    torch.save(torch.cat(results_shift, dim=0), f'data/{dataset_name}.pitch_shift_range.pt')
    lengths = [len(data) for data in results_data]
    # save midi file names
    f = open(f'data/{dataset_name}.txt', 'w')
    for i, midi_file in enumerate(midi_files):
        f.write(str(i) + '\t' + midi_file + '\n')
    # np.save(f'data/{dataset_name}.length.npy', np.array(lengths))
    torch.save(torch.tensor(lengths), f'data/{dataset_name}.length.pt')

def test_la_tonal_scales():
    FILE_LIST = []
    ROOT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    for folder in os.listdir(os.path.join(LA_DATASET_PATH, 'MIDIs')):
        FILE_LIST.extend([os.path.join(LA_DATASET_PATH, 'MIDIs', folder, file)
            for file in os.listdir(os.path.join(LA_DATASET_PATH, 'MIDIs', folder))])
    print(len(FILE_LIST))
    np.random.seed(42)
    FILE_LIST = np.random.choice(FILE_LIST, 100, replace=False)
    n_good_samples = 0
    for file in FILE_LIST:
        try:
            midi = UglyMIDI(file)
        except:
            continue
        if not any(ins.is_drum for ins in midi.instruments):
            continue # want pop-like songs
        evidences = locate_tonal_scale(midi)
        if len(evidences) > 0:
            print(f'File: {file}')
            for evidence in evidences:
                print(f'Time: {evidence[0]:.2f} - {evidence[1]:.2f}, Scale: {ROOT_NAMES[evidence[2]]}, Structure: {evidence[3]}, Instrument: {evidence[4]}, Instrument ID: {evidence[5]}')
            print('---')
            n_good_samples += 1
    print(f'Found {n_good_samples} files with tonal scales out of {len(FILE_LIST)} sampled files.')


def analyze_file_cadence(midi_file, gt_label_file=None, max_polyphony=16, beat_div=4, visualize=False):
    try:
        midi = UglyMIDI(midi_file, constant_tempo=60.0 / beat_div)
    except:
        return None
    if not filter_la_quantization(midi):
        return None
    if gt_label_file is None:
        cadence_chroma = get_cadence_chroma(midi)
        if cadence_chroma is None:
            return None
    result = preprocess_midi(
        midi,
        max_polyphony=max_polyphony,
        beat_div=beat_div,
        filter=False # already filtered
    )
    (data, pitch_shift_range) = result
    if result is None:
        return None
    if gt_label_file is None:
        cadence_chroma = torch.clip(torch.floor(torch.from_numpy(cadence_chroma) * 256), 0, 255).type(torch.uint8)
        chroma_expand = cadence_chroma[None].expand(data.shape[0], -1)
    else:
        cadence_chroma = np.zeros((len(data), 24))
        f = open(gt_label_file, 'r')
        lines = [f.strip() for f in f.readlines() if f.strip()]
        f.close()
        perf_midi = UglyMIDI(midi_file)
        from pretty_midi_fix import get_time_mapping
        perf_to_score_mapping = get_time_mapping(perf_midi, midi)
        for line in lines:
            start_time, end_time, label = line.split('\t')
            start_time = int(np.round(perf_to_score_mapping(float(start_time))))
            end_time = int(np.round(perf_to_score_mapping(float(end_time))))
            root, mode = label.strip().split(' ')
            root_id = mir_eval.chord.pitch_class_to_semitone(root)
            cadence_chroma[start_time:end_time, root_id] = 1
            cadence_chroma[start_time:end_time, (root_id + {'major': 4, 'minor': 3}[mode]) % 12] = 1
            cadence_chroma[start_time:end_time, (root_id + 7) % 12] = 1
            cadence_chroma[start_time:end_time, root_id + 12] = 1
        chroma_expand = torch.clip(torch.floor(torch.from_numpy(cadence_chroma) * 256), 0, 255).type(torch.uint8)
    if visualize:
        plt.imshow(chroma_expand.cpu().numpy().T, aspect='auto', origin='lower')
        plt.title('Cadence Chroma')
        plt.show()
    data = torch.cat((data, chroma_expand), dim=1)
    max_length = 512
    return (data[-max_length:], pitch_shift_range)

def get_cadence_chroma(midi: UglyMIDI):
    # First get end time of the last note
    end_time = 0
    pitched_notes = []
    cadence_chroma = np.zeros(12)
    for ins in midi.instruments:
        if ins.is_drum or ins.program >= 0x70:  # Ignore drums, percussive and sound effects
            continue
        for note in ins.notes:
            if note.end > end_time:
                end_time = note.end
    if end_time < 300:  # too short, might be without cadence
        return None

    for ins in midi.instruments:
        if ins.is_drum or ins.program >= 0x70:  # Ignore drums, percussive and sound effects
            continue
        for note in ins.notes:
            delta_end_time = end_time - note.end
            delta_length = note.end - note.start
            if delta_end_time > 18.0 or delta_length < 6.0 or note.velocity < 20:
                continue # Unreliable note
            if delta_length > delta_end_time:
                cadence_chroma[note.pitch % 12] += delta_length - delta_end_time
    if np.sum(cadence_chroma) < 0.5:
        return None # Cannot decide
    # Remove the effect of thirds and fifths
    cadence_tonal = cadence_chroma - np.roll(cadence_chroma, 3) - np.roll(cadence_chroma, 4) - np.roll(cadence_chroma, 7)
    cadence_tonal = np.clip(cadence_tonal, 0, None)  # Remove negative values
    if np.sum(cadence_tonal) < 1e-6:
        return None  # Cannot decide
    # Normalize the cadence chroma
    return np.concatenate([cadence_chroma / cadence_chroma.max(), cadence_tonal / cadence_tonal.max()])

if __name__ == '__main__':
    create_rwc_dataset(max_polyphony=16, dataset_name='rwc_key_cadence_16_v1', visualize=False, func_name='cadence')