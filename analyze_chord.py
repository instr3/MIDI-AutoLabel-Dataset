from types import NoneType
from pretty_midi_fix import UglyMIDI
import pretty_midi
import os
import numpy as np
from settings import LA_DATASET_PATH
from joblib import Parallel, delayed
from tqdm import tqdm
from preprocess_large_midi_dataset import filter_la_quantization, preprocess_midi
import torch

FILE_LIST = []


def analyze_file(midi_file, save_to_file=False, max_polyphony=16, beat_div=4, seq_length=96 * 4, seed=None, fixed_id=None):
    np.random.seed(seed)
    try:
        midi = UglyMIDI(midi_file, constant_tempo=60.0 / beat_div)
    except:
        return None
    end_time = midi.get_end_time()
    if end_time > 7200 * beat_div:
        # Too long, likely broken
        return None
    if not filter_la_quantization(midi):
        return None
    ins_ids = np.arange(len(midi.instruments))
    if fixed_id is None:
        np.random.shuffle(ins_ids)
    for i in ins_ids:
        ins = midi.instruments[ins_ids[i]]
        if fixed_id is not None:
            if i == fixed_id:
                midi_data = preprocess_midi(
                    midi_file,
                    max_polyphony=max_polyphony,
                    ins_ids='all',
                    beat_div=beat_div,
                    filter=False  # already filtered
                )[0]
                (start_frame, end_frame) = (0, midi_data.shape[0])
            else:
                continue
        else:
            (start_frame, end_frame) = get_chroma(ins, end_time, window_length=seq_length)
        if start_frame is not None:
            # print(midi_file, ins.name)
            midi_copy = pretty_midi.PrettyMIDI(initial_tempo=60.0 / beat_div)
            # midi_copy.instruments.append(scatter_midi_notes(ins, end_time))  # make this the first instrument
            for ins2 in midi.instruments:
                if ins2 != ins:
                    midi_copy.instruments.append(ins2)
            if save_to_file:
                output_midi_path = os.path.join('temp', 'chord_track_analysis', os.path.basename(midi_file))
                midi_copy.write(output_midi_path)
            result = preprocess_midi(
                midi_copy,
                max_polyphony=max_polyphony,
                fixed_length=end_frame,
                ins_ids='all',
                beat_div=beat_div,
                filter=False  # already filtered
            )
            if result is None:
                return None  # probably notrack is empty
            data, pitch_shift_range = result
            chroma = notes_to_chroma(ins, data.shape[0])
            clipped_data = data[start_frame:end_frame]
            chroma = chroma[start_frame:end_frame]
            return torch.cat([clipped_data, chroma], -1), pitch_shift_range
    return None


def notes_to_chroma(ins, length):
    chroma = torch.zeros(length, 12, dtype=torch.uint8)
    bass_note = torch.full((length, ), 128, dtype=torch.uint8)
    for note in ins.notes:
        start = max(0, int(np.round(note.start)))
        end = min(length, max(start + 1, int(np.round(note.end))))
        if end > start:
            chroma[start:end, note.pitch % 12] = 1
            bass_note[start:end] = torch.minimum(bass_note[start:end], torch.full_like(bass_note[start:end], note.pitch))
    bass_chroma = torch.zeros(length, 12, dtype=torch.uint8)
    for p in range(12):
        bass_chroma[:, p] = torch.logical_and(bass_note % 12 == p,
                                              bass_note < 128)
    return torch.cat((chroma, bass_chroma), dim=1)

def scatter_midi_notes(ins, end_time):
    n_frames = int(end_time)
    new_ins = pretty_midi.Instrument(27, is_drum=ins.is_drum, name='CHORD')  # electric guitar
    # Break all notes into single beats
    for note in ins.notes:
        start = max(0, int(np.round(note.start / 2) * 2))  # round to nearest eighth note
        end = min(n_frames, max(start + 2, int(np.round(note.end / 2) * 2)))  # round to nearest eighth note
        if end > start:
            for j in range(start, end):
                if j % 4 == 0 or j == start:
                    note_end = min((j // 4 + 1) * 4, end)
                    new_ins.notes.append(pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=j, end=note_end))
    return new_ins


def get_chroma(ins, end_time, window_length, accept_percentage=0.95):
    if ins.is_drum or ins.program >= 112: # percussive or fx
        return False
    n_frames = int(end_time)
    if n_frames < window_length:
        n_frames = window_length
    chroma = np.zeros((n_frames, 12), dtype=int)
    for note in ins.notes:
        start = max(0, int(np.round(note.start)))
        end = min(n_frames, int(np.round(note.end)))
        if end > start:
            chroma[start:end, note.pitch % 12] = 1
    chord_polyphony = chroma.sum(axis=1)
    score = (chord_polyphony >= 3).astype(float) + (chord_polyphony == 4).astype(float) * 0.1 - (
                chord_polyphony >= 6).astype(float) * 2.0 - (chord_polyphony == 2) * 0.5 - (chord_polyphony == 1) * 0.5
    running_score = np.cumsum(score)
    # Prepend a single 0 before running_score
    running_score = np.concatenate([[0], running_score])
    running_score = running_score[window_length:] - running_score[:-window_length]
    if running_score.max() >= window_length * accept_percentage:
        # Randomly select a start time where the running score is greater than the threshold
        start_frame = np.random.choice(np.where(running_score >= window_length * accept_percentage)[0])
        end_frame = start_frame + window_length
        return (start_frame, end_frame)
    else:
        return (None, None)


def create_chord_dataset(max_polyphony, dataset_name, max_idx=None, save_to_file=False):
    # Get all midi files in the folder, recursively
    midi_files = []
    for folder in os.listdir(os.path.join(LA_DATASET_PATH, 'MIDIs')):
        midi_files.extend([os.path.join(LA_DATASET_PATH, 'MIDIs', folder, file)
                           for file in os.listdir(os.path.join(LA_DATASET_PATH, 'MIDIs', folder))])
    print(len(midi_files))
    if max_idx is not None:
        midi_files = midi_files[:max_idx]
    # Process files in parallel
    print(f'Processing {len(midi_files)} files')
    results = Parallel(n_jobs=-1)(
        delayed(analyze_file)(midi_file, save_to_file=save_to_file, max_polyphony=max_polyphony, seed=seed) for
        seed, midi_file in tqdm(enumerate(midi_files), total=len(midi_files)))
    # Filter out None results
    midi_files = [os.path.relpath(midi_files[i], folder) for i, result in enumerate(results) if result is not None]
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


def create_rwc_chord_dataset(max_polyphony, dataset_name, save_to_file=False):
    # Get all midi files in the folder, recursively
    rwc_chord_folder = os.path.join('temp', 'rwc_chord')
    midi_files = os.listdir(rwc_chord_folder)
    print(len(midi_files))
    # Process files in parallel
    print(f'Processing {len(midi_files)} files')
    results = Parallel(n_jobs=-1)(
        delayed(analyze_file)(os.path.join(rwc_chord_folder, midi_file), save_to_file=save_to_file, max_polyphony=max_polyphony, seed=seed, fixed_id=0) for
        seed, midi_file in tqdm(enumerate(midi_files), total=len(midi_files)))
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

if __name__ == '__main__':
    # analyze_file(r'C:\Users\Admin\Documents\datasets\Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA\MIDIs\0\002b7f545e1f19ecb63acb183561662a.mid', save_to_file=False)
    create_chord_dataset(max_polyphony=16, dataset_name='la_chords_16_v9_chroma')
    create_rwc_chord_dataset(max_polyphony=16, dataset_name='rwc_chords_16_v9_chroma')
