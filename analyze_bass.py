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
from analyze_chord import scatter_midi_notes

FILE_LIST = []


def analyze_file(midi_file, save_to_file=False, max_polyphony=16, beat_div=4, seq_length=96 * 4, seed=None):
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
    min_pitch_ins_id = -1
    min_pitch_ins_pitch = np.inf
    for i, ins in enumerate(midi.instruments):
        if ins.is_drum or ins.program >= 112:
            continue
        if len(ins.notes) < 20:
            continue
        mean_pitch = np.mean([note.pitch for note in ins.notes])
        if mean_pitch < min_pitch_ins_pitch:
            min_pitch_ins_pitch = mean_pitch
            min_pitch_ins_id = i
    if min_pitch_ins_id == -1:
        return None
    (start_frame, end_frame) = get_bass(midi.instruments[min_pitch_ins_id], end_time, window_length=seq_length)
    if start_frame is not None:
        # print(midi_file, ins.name)
        midi_copy = pretty_midi.PrettyMIDI(initial_tempo=60.0 / beat_div)
        for ins2 in midi.instruments:
            if ins2 != midi.instruments[min_pitch_ins_id]:
                midi_copy.instruments.append(ins2)
        result = preprocess_midi(
            midi_copy,
            max_polyphony=max_polyphony,
            fixed_length=end_frame,
            ins_ids='all',
            beat_div=beat_div,
            filter=False  # already filtered
        )
        if save_to_file:
            midi_copy.instruments.insert(0, scatter_midi_notes(midi.instruments[min_pitch_ins_id], end_time))
            output_midi_path = os.path.join('temp', 'bass_track_analysis', os.path.basename(midi_file))
            midi_copy.write(output_midi_path)
        if result is None:
            return None  # probably notrack is empty
        data, pitch_shift_range = result
        bass_chroma = notes_to_bass_chroma(midi.instruments[min_pitch_ins_id], data.shape[0])
        clipped_data = data[start_frame:end_frame]
        bass_chroma = bass_chroma[start_frame:end_frame]
        return torch.cat([clipped_data, bass_chroma], -1), pitch_shift_range
    return None


def notes_to_bass_chroma(ins, length):
    bass_note = torch.full((length, ), 128, dtype=torch.uint8)
    for note in ins.notes:
        start = max(0, int(np.round(note.start)))
        end = min(length, max(start + 1, int(np.round(note.end))))
        if end > start:
            bass_note[start:end] = torch.minimum(bass_note[start:end], torch.full_like(bass_note[start:end], note.pitch))
    bass_chroma = torch.zeros(length, 12, dtype=torch.uint8)
    for p in range(12):
        bass_chroma[:, p] = torch.logical_and(bass_note % 12 == p,
                                              bass_note < 128)
    return bass_chroma

def get_bass(ins, end_time, window_length, accept_percentage=0.95):
    if ins.is_drum or ins.program >= 112: # percussive or fx
        return (None, None)
    if any([note.pitch > 60 for note in ins.notes]):
        # too high
        return (None, None)
    n_frames = int(end_time)
    if n_frames < window_length:
        n_frames = window_length
    bass = np.zeros((n_frames, 12), dtype=int)
    bass_cover = np.zeros(n_frames, dtype=int)
    for note in ins.notes:
        start = max(0, int(np.round(note.start)))
        end = min(n_frames, int(np.round(note.end)))
        start_cover = max(0, int(np.round(note.start)) - 4)
        end_cover = min(n_frames, int(np.round(note.end)) + 4)
        if end > start:
            bass[start:end, note.pitch % 12] = 1
            bass_cover[start_cover:end_cover] = 1
    bass_polyphony = bass.sum(axis=1)
    score = bass_cover.astype(float) - (bass_polyphony >= 2).astype(float) * 10.0
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


def create_bass_dataset(max_polyphony, dataset_name, max_idx=None, save_to_file=False):
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

if __name__ == '__main__':
    # analyze_file(r'C:\Users\Admin\Documents\datasets\Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA\MIDIs\0\002b7f545e1f19ecb63acb183561662a.mid', save_to_file=False)
    create_bass_dataset(max_polyphony=16, dataset_name='la_bass_16_v2')
    # create_rwc_chord_dataset(max_polyphony=16, dataset_name='rwc_chords_16_v8_chroma')
