import torch

from pretty_midi_fix import UglyMIDI
from settings import RWC_DATASET_PATH
import os
import numpy as np
import sys
from baselines.exported_midi_chord_recognition.main import rule_based_chord_recognition
from cp_transformer_probe_chord import RoformerProberChord
try:
    from baselines.chorder_chord_eval import extract_chord_chorder
except:
    extract_chord_chorder = None
from preprocess_large_midi_dataset import preprocess_midi
from tqdm import tqdm
import mir_eval.chord
from cp_transformer_fine_tune import decompress

def read_chord_file(chord_file):
    with open(chord_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    result = []
    for line in lines:
        tokens = line.split('\t')
        result.append([float(tokens[0]), float(tokens[1]), tokens[2]])
    return result

def get_split(dataset_name, split, split_ratio=10):
    if dataset_name == 'rwc':
        folder = os.path.join(RWC_DATASET_PATH, 'MidiAlignedChord')
        file_list = sorted(os.listdir(folder))
        midi_files = []
        for file in file_list:
            midi_file = os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file.split('.')[0] + '.SMF_SYNC.MID')
            midi_files.append([midi_file, os.path.join(folder, file)])
        song_indices = np.arange(len(midi_files))
        # Get training or validation split
        if split == 'all':
            pass
        elif split == 'train':
            song_indices = song_indices[song_indices % split_ratio > 1]
        elif split == 'val':
            song_indices = song_indices[song_indices % split_ratio == 1]
        elif split == 'test':
            song_indices = song_indices[song_indices % split_ratio == 0]
        elif split == 'sanity':
            song_indices = song_indices[song_indices % split_ratio == 1][:1]
        else:
            raise ValueError(f'Unknown split {split}')
        return [midi_files[i] for i in song_indices]
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

def chord_lab_to_labeled_intervals(chord_lab):
    return np.array([[chord[0], chord[1]] for chord in chord_lab]), [chord[2] for chord in chord_lab]

def perform_eval(estimate_function, dataset_name, split, model_name):
    test_dataset = get_split(dataset_name, split)
    eval_results = []
    for midi_path, ref_chord_path in tqdm(test_dataset):
        est_chord = estimate_function(midi_path)
        ref_chords = read_chord_file(ref_chord_path)
        ref_intervals, ref_labels = chord_lab_to_labeled_intervals(ref_chords)
        est_intervals, est_labels = chord_lab_to_labeled_intervals(est_chord)
        eval_result = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_intervals, est_labels)
        eval_results.append(eval_result)
        print(eval_result, file=sys.stderr, flush=True)
    print(f'Evaluation results for {model_name} on {dataset_name} {split}', file=sys.stderr)
    # Get the median of all evaluation results
    aggregated_results = {}
    for key in eval_results[0]:
        values = [result[key] for result in eval_results]
        aggregated_results[key] = np.median(values)
        aggregated_results[key + '-mean'] = f'{np.mean(values):.3f}+-{np.std(values, ddof=1):.3f}'
    print('root', aggregated_results['root'],  aggregated_results['root-mean'], file=sys.stderr)
    print('majmin', aggregated_results['majmin'], aggregated_results['majmin-mean'], file=sys.stderr)
    print('sevenths', aggregated_results['sevenths'], aggregated_results['sevenths-mean'], file=sys.stderr)

def aggregate_data(array_list):
    max_len = max([len(array) for array in array_list])
    feature_len = array_list[0].shape[1]
    result = np.zeros((max_len, feature_len))
    for i, array in enumerate(array_list):
        result[:len(array), :] += array
    return result / len(array_list)

def aggregate_data_binary(array_list):
    max_len = max([len(array) for array in array_list])
    feature_len = array_list[0].shape[1]
    result = np.zeros((max_len, len(array_list)), dtype=int)
    for i, array in enumerate(array_list):
        for j in range(feature_len):
            result[:len(array), i] += (array[:, j] > 0).astype(int) << j
    def bincount(arr):
        counts = np.bincount(arr)
        # Lower the weights of 0
        counts[0] -= len(array_list) * 2 / 3
        return counts.argmax()
    # Get mode along the last axis
    result = np.apply_along_axis(bincount, axis=1, arr=result)
    # Decode the binary result
    output = np.zeros((max_len, feature_len))
    for j in range(feature_len):
        output[:, j] = (result >> j) & 1
    return output

def model_chord_recognition(model, midi_path, n_samples, batch_size, chunk_length=384, temperature=1.0, replace_bass=False, write_to_file=True, return_raw_output=False):
    torch.random.manual_seed(42)
    np.random.seed(42)
    x1, _ = decompress(model, preprocess_midi(midi_path, 16, ins_ids=['all', 'all'], filter=False)[0])
    n_chunks = (x1.shape[1] - 1) // chunk_length + 1
    expected_length = n_chunks * chunk_length
    chunk_step = chunk_length // 2
    # TODO: this seems bad
    x1, _ = decompress(model, preprocess_midi(midi_path, 16, ins_ids=['all', 'all'], filter=False, fixed_length=expected_length)[0])
    print('length:', x1.shape[1])
    results = []
    prev_end_frame = 0
    batch_hops = batch_size // n_samples
    n_batch = ((n_chunks * 2 - 1) - 1) // batch_hops + 1
    with torch.no_grad():
        for batch_id in range(n_batch):
            x1p_collection = []
            for i in range(batch_hops * batch_id, min(batch_hops * (batch_id + 1), n_chunks * 2 - 1)):
                x1p = x1[:, i * chunk_step:i * chunk_step + chunk_length].repeat(n_samples, 1, 1)
                x1p_collection.append(x1p)
            x1p_collection = torch.cat(x1p_collection, dim=0)
            output = model.global_sampling(x1p_collection,
                                           torch.zeros(x1p_collection.shape[0], 0, x1.shape[-1], device=x1.device, dtype=x1.dtype),
                                           temperature=temperature)
            for i in range(batch_hops * batch_id, min(batch_hops * (batch_id + 1), n_chunks * 2 - 1)):
                sample_start = (i - batch_hops * batch_id) * n_samples
                batch_output = [f[sample_start:sample_start + n_samples] for f in output]
                target_length = chunk_length if i + 1 == n_chunks * 2 - 1 else chunk_length * 3 // 4
                batch_output = batch_output[prev_end_frame:target_length]
                prev_end_frame = target_length - chunk_length // 2
                results.extend(batch_output)
    bass_data, chroma_data, all_data = [], [], []
    if return_raw_output:
        return torch.stack(results, dim=0)
    for i in range(n_samples):
        output_i = [results[j][i:i + 1, :] for j in range(len(results))]
        midi = decode_output(output_i, fixed_program=0)
        if len(midi.instruments) == 0 or len(midi.instruments[0].notes) == 0:
            print(f'No notes in the output MIDI for {midi_path}, skipping', file=sys.stderr, flush=True)
            continue
        bass, chroma = rule_based_chord_recognition(midi, return_chroma=True)
        bass_data.append(bass)
        chroma_data.append(chroma)
    if len(bass_data) == 0:
        raise Exception('No valid outputs for {}'.format(midi_path))
    bass_data = aggregate_data(bass_data)
    chroma_data = aggregate_data(chroma_data)
    chords = rule_based_chord_recognition(midi_path, replace_chroma=(bass_data if replace_bass else None, chroma_data), use_transition=False)
    if write_to_file:
        output_file = os.path.join('cache_data', 'chord_eval', model.save_name, f'{os.path.basename(midi_path)}-temp{temperature}.txt')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        f = open(output_file, 'w')
        for (start, end, chord) in chords:
            f.write(f'{start:.3f}\t{end:.3f}\t{chord}\n')
        f.close()
    return chords

def get_voiced_time_range(midi):
    start_time = None
    end_time = None
    for ins in midi.instruments:
        if ins.is_drum or ins.program >= 0x70:  # Ignore drum and sound effect tracks
            continue
        for note in ins.notes:
            if start_time is None or note.start < start_time:
                start_time = note.start
            if end_time is None or note.end > end_time:
                end_time = note.end
    return int(np.round(start_time)), int(np.round(end_time))

def model_chord_probe(model, midi_path, n_samples, batch_size, chunk_length=384, bass_model=None, write_to_file=True, return_raw_output=False, return_chord_lab=True):
    assert n_samples == 1
    torch.random.manual_seed(42)
    np.random.seed(42)
    midi = UglyMIDI(midi_path, constant_tempo=60.0 / 4)
    x1, _ = decompress(model, preprocess_midi(midi, 16, ins_ids=['all', 'all'], filter=False)[0])
    n_chunks = (x1.shape[1] - 1) // chunk_length + 1
    expected_length = n_chunks * chunk_length
    chunk_step = chunk_length // 2
    # TODO: this seems bad
    x1, _ = decompress(model, preprocess_midi(midi, 16, ins_ids=['all', 'all'], filter=False, fixed_length=expected_length)[0])
    print('length:', x1.shape[1])
    results = []
    prev_end_frame = 0
    batch_hops = batch_size // n_samples
    n_batch = ((n_chunks * 2 - 1) - 1) // batch_hops + 1
    with torch.no_grad():
        for batch_id in range(n_batch):
            x1p_collection = []
            for i in range(batch_hops * batch_id, min(batch_hops * (batch_id + 1), n_chunks * 2 - 1)):
                x1p = x1[:, i * chunk_step:i * chunk_step + chunk_length].repeat(n_samples, 1, 1)
                x1p_collection.append(x1p)
            x1p_collection = torch.cat(x1p_collection, dim=0)
            output = model.inference(x1p_collection)
            if bass_model is not None:
                bass_output = bass_model.inference(x1p_collection)
                output[:, :, 12:] = bass_output # Replace the bass part with the output of the bass model
                # output[:, :, :12] = torch.maximum(output[:, :, :12], bass_output) # Combine the chord part with the bass part
            for i in range(batch_hops * batch_id, min(batch_hops * (batch_id + 1), n_chunks * 2 - 1)):
                sample_start = (i - batch_hops * batch_id) * n_samples
                batch_output = output[sample_start:sample_start + n_samples]
                target_length = chunk_length if i + 1 == n_chunks * 2 - 1 else chunk_length * 3 // 4
                batch_output = batch_output[:, prev_end_frame:target_length]
                prev_end_frame = target_length - chunk_length // 2
                results.append(batch_output)
    results = torch.cat(results, dim=1).squeeze(0)
    start_time, end_time = get_voiced_time_range(midi)
    if start_time is None:
        results = np.zeros_like(results)
    else:
        # Zero the frames outside the voiced range
        print('Voiced range:', start_time, end_time)
        results[:start_time] = 0
        results[end_time:] = 0
    # Pool 4 subbeats
    results = results[:results.shape[0] // 4 * 4]
    results = results.view(results.shape[0] // 4, 4, 24).mean(dim=1)
    if return_raw_output and not return_chord_lab:
        return results
    results = results.cpu().numpy()
    chroma_data = results[:, :12]
    bass_data = results[:, 12:]
    chords = rule_based_chord_recognition(midi_path, replace_chroma=(bass_data, chroma_data), use_transition=False)
    if write_to_file:
        output_file = os.path.join('cache_data', 'chord_eval', model.save_name, f'{os.path.basename(midi_path)}.txt')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        f = open(output_file, 'w')
        for (start, end, chord) in chords:
            f.write(f'{start:.3f}\t{end:.3f}\t{chord}\n')
        f.close()
    if return_raw_output and return_chord_lab:
        return results, chords
    return chords

def oracle_chord_recognition(midi_path):
    id = os.path.basename(midi_path).split('.')[0]
    oracle_file = os.path.join(f'temp/rwc_chord/{id}.mid')
    midi = UglyMIDI(oracle_file)
    midi.instruments = midi.instruments[:2]  # chord tracks
    bass_data, chroma_data = rule_based_chord_recognition(midi, return_chroma=True)
    return rule_based_chord_recognition(midi_path, replace_chroma=(bass_data, chroma_data))

def perform_eval_probe(model_name, bass_model_name, dataset_name, split, n_samples=1, batch_size=4):
    torch.random.manual_seed(42)
    np.random.seed(42)
    model_dir = model_name.split('.epoch')[0]
    if os.path.exists(f'ckpt/{model_name}'):
        model_path = f'ckpt/{model_name}'
    elif os.path.exists(f'ckpt/{model_dir}'):
        model_path = f'ckpt/{model_dir}/{model_name}'
    else:
        model_path = f'ckpt/{model_name}'
    model_type = RoformerProberChord
    model = model_type.load_from_checkpoint(model_path, strict=False)
    model.save_name = os.path.basename(model_path)
    model.cuda()
    model.eval()
    if bass_model_name is not None:
        bass_model = RoformerProberChord.load_from_checkpoint(bass_model_name, strict=False)
        bass_model.save_name = os.path.basename(bass_model_name)
        bass_model.cuda()
        bass_model.eval()
    else:
        bass_model = None
    return perform_eval(
        lambda midi_path: model_chord_probe(model, midi_path, n_samples, bass_model=bass_model, batch_size=batch_size),
        dataset_name, split, f'probe_vote{n_samples}:{model_name}')


def main():
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('model_name', type=str)
    args.add_argument('--bass_model_name', type=str)
    args.add_argument('--batch_size', type=int, default=4)
    args = args.parse_args()
    perform_eval_probe(args.model_name, args.bass_model_name, 'rwc', 'test', n_samples=1, batch_size=args.batch_size)

if __name__ == '__main__':
    main()