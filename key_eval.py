import numpy as np

from cp_transformer_probe import RoformerProber, RoformerProberPlain
from cp_transformer_probe_chord import RoformerProberChord
from cp_transformer_probe_cadence import RoformerProberCadence
from preprocess_large_midi_dataset import preprocess_midi, DURATION_TEMPLATES
from pretty_midi_fix import UglyMIDI
from settings import RWC_DATASET_PATH, LA_DATASET_PATH
import torch
import pretty_midi
import os
import sys
from settings import *
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
from hmm import hmm_decode
import mir_eval.key

def locate_rwc(rwc_id):
    return os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', f'RM-P{rwc_id + 1:03}.SMF_SYNC.MID')

cadence_model = None # Global variable to hold the cadence model
last_midi = None # Global variable to hold the last processed MIDI file
last_cadence_result = None # Global variable to hold the last cadence result


def eval_key(gt_label_path, est_labels):
    f = open(gt_label_path, 'r')
    ref_labels = [line.strip().split('\t') for line in f.readlines() if line.strip()]
    f.close()
    # Only eval one pair
    if len(ref_labels[0]) == 1:
        ref_key = ref_labels[0][0].replace('Major', 'major')
    else:
        ref_key = ref_labels[0][-1].replace('Major', 'major')
    est_key = est_labels[-1][1].replace('Major', 'major')
    ref_key = ref_key.replace('Cb', 'B')
    est_key = est_key.replace('Cb', 'B')
    scores = mir_eval.key.evaluate(ref_key, est_key)
    def get_scale_id(key_name):
        scale_id = mir_eval.chord.pitch_class_to_semitone(key_name.split(' ')[0])
        if 'minor' in key_name:
            scale_id = (scale_id + 3) % 12  # Convert minor to relative major
        elif 'major' in key_name:
            scale_id = scale_id % 12
        else:
            raise ValueError(f'Unknown key name: {key_name}')
        return scale_id
    scores['scale'] = 1.0 if get_scale_id(ref_key) == get_scale_id(est_key) else 0.0
    scores['acc'] = 1.0 if ref_key == est_key else 0.0  # why didn't mir_eval calculate this?
    return scores

def merge_score_median(scores_list):
    keys = scores_list[0].keys()
    merged_scores = {key: [] for key in keys}
    for scores in scores_list:
        for key in keys:
            merged_scores[key].append(scores[key])
    for key in keys:
        merged_scores[key] = np.median(merged_scores[key])
    return merged_scores

def merge_score_mean(scores_list):
    keys = scores_list[0].keys()
    merged_scores = {key: [] for key in keys}
    for scores in scores_list:
        for key in keys:
            try:
                merged_scores[key].append(scores[key])
            except:
                print(f'Error computing score for {key}')
                pass
    for key in keys:
        merged_scores[key] = np.mean(merged_scores[key])
    return merged_scores

def decompress(model, byte_arr):
    x = torch.tensor(byte_arr).unsqueeze(0)
    x = x.cuda()
    return model.base_model.preprocess(x, pitch_shift=torch.zeros(1, dtype=torch.int8).cuda())


TOKEN_ID_LABELS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
MODES = {0: 'Major', 2: 'Dorian', 4: 'Phrygian', 5: 'Lydian', 7: 'Mixolydian', 9: 'Minor', 11: 'Locrian'}
MODES_MAJMIN = {0: 'Major', 9: 'Minor'}
def rule_based_mode_estimation(midi, start_time, end_time, scale_id):
    histogram = np.zeros((12,), dtype=np.float32)
    for ins in midi.instruments:
        if ins.is_drum:
            continue
        for note in ins.notes:
            if note.start < start_time or note.start > end_time:
                continue
            pitch = note.pitch % 12
            histogram[pitch] += np.clip(end_time - start_time, 4.0, 24.0)
    best_mode_id = 0
    for mode_id in MODES:
        score = histogram[(mode_id + scale_id) % 12]
        if score > histogram[(best_mode_id + scale_id) % 12]:
            best_mode_id = mode_id
    return f'{TOKEN_ID_LABELS[(scale_id + best_mode_id) % 12]}:{MODES[best_mode_id]}'

def cadence_based_mode_estimation(midi, start_time, end_time, scale_id, use_majmin=False, batch_size=8):
    modes = MODES_MAJMIN if use_majmin else MODES
    global cadence_model, last_midi, last_cadence_result
    if cadence_model is None:
        cadence_model = RoformerProberCadence.load_from_checkpoint('ckpt/midi-unsupervised-models/cp_transformer_yinyang_probe_cadence_v0.1_batch_80_la_cadence_v4.epoch=00.val_loss=0.23473.ckpt', strict=False)
        cadence_model.cuda()
        cadence_model.eval()
        cadence_model.save_name = '$'
    if midi == last_midi:
        cadence = last_cadence_result
    else:
        cadence = probe(cadence_model, midi, ins_ids='all', visualize=False, decode=False, batch_size=batch_size)
        last_midi = midi
        last_cadence_result = cadence
    histogram = cadence[start_time:end_time, 12:].sum(axis=0)
    best_mode_id = 0
    for mode_id in modes:
        score = histogram[(mode_id + scale_id) % 12]
        if score > histogram[(best_mode_id + scale_id) % 12]:
            best_mode_id = mode_id
    return f'{TOKEN_ID_LABELS[(scale_id + best_mode_id) % 12]}:{modes[best_mode_id]}'


def probe(model, midi_path, ins_ids=None, generation_length=384, overlap_ratio=0.125, batch_size=8, beat_div=4, gt_key_path=None, visualize=True, decode=True):
    if isinstance(midi_path, UglyMIDI):
        score_midi = midi_path
    else:
        if ins_ids is None:
            raise ValueError('ins_ids must be provided')
        folder = f'probe_{model.save_name}'
        ins_ids_str = ','.join(ins_ids)
        file_name_pattern = f'temp/{folder}/{os.path.basename(midi_path)}_probe[{ins_ids_str}].mid'
        if os.path.exists(file_name_pattern):
            print(f'Already exists: {file_name_pattern}')
            return
        score_midi = UglyMIDI(midi_path, constant_tempo=60.0 / beat_div)
    x1 = decompress(model, preprocess_midi(score_midi, 16, ins_ids=ins_ids, filter=False, beat_div=beat_div)[0])
    hop_size = int(overlap_ratio * generation_length)
    n_chunks = max(1, (x1.shape[1] - generation_length - 1) // hop_size + 2)
    x1 = decompress(model, preprocess_midi(score_midi, 16, ins_ids=ins_ids, filter=False, beat_div=beat_div, fixed_length=(n_chunks - 1) * hop_size + generation_length)[0])
    n_batches = (n_chunks + batch_size - 1) // batch_size
    if '4096' in model.save_name:
        result = np.zeros((x1.shape[1], 4096))
    elif ('key' in model.save_name or 'bass' in model.save_name) and 'cadence' not in model.save_name and 'majmin' not in model.save_name:
        result = np.zeros((x1.shape[1], 12))
    else:
        result = np.zeros((x1.shape[1], 24))
    result_weights = np.zeros((x1.shape[1],))
    with torch.no_grad():
        for i_batch in range(n_batches):
            print('Processing batch', i_batch + 1, '/', n_batches, flush=True)
            input_collection = []
            for i_chunk in range(i_batch * batch_size, min((i_batch + 1) * batch_size, n_chunks)):
                start_pos = i_chunk * hop_size
                end_pos = start_pos + generation_length
                input_collection.append(x1[:, start_pos:end_pos, :])
            x1_batch = torch.cat(input_collection, dim=0)
            output = model.inference(x1_batch)
            output = output.cpu().numpy()
            for i_chunk in range(i_batch * batch_size, min((i_batch + 1) * batch_size, n_chunks)):
                start_pos = i_chunk * hop_size
                end_pos = start_pos + generation_length
                result[start_pos:end_pos] += output[i_chunk - i_batch * batch_size, :, :]
                result_weights[start_pos:end_pos] += 1.0
    result /= np.maximum(result_weights, 1e-6)[:, None]
    if decode:
        decoded_seq = hmm_decode(torch.from_numpy(np.log(result))[None], transition_penalty=200.0).squeeze(0)
        lab = []
        for i in range(decoded_seq.shape[0]):
            if i == 0 or decoded_seq[i] != decoded_seq[i - 1]:
                if i > 0:
                    lab[-1][1] = i
                lab.append([i, None, decoded_seq[i].item()])
        lab[-1][1] = decoded_seq.shape[0]
        for line in lab:
            line[2] = cadence_based_mode_estimation(score_midi, line[0], line[1], line[2], use_majmin = 'rwc' in model.save_name, batch_size=batch_size)  # RWC is supervise-finetuned on maj/min
        print(lab)
    if visualize:
        plt.imshow(result.T, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Probe on {os.path.basename(midi_path)}')
        plt.xlabel('Time Steps')
        plt.ylabel('Token ID')
        # Set y-ticks to note names
        if result.shape[1] == 12:
            plt.yticks(ticks=np.arange(len(TOKEN_ID_LABELS)), labels=TOKEN_ID_LABELS)
        elif result.shape[1] == 24:
            # Copy the labels for the second octave
            plt.yticks(ticks=np.arange(len(TOKEN_ID_LABELS) * 2), labels=TOKEN_ID_LABELS * 2)
        # Revert y-axis to have C at the bottom
        plt.gca().invert_yaxis()
        plt.show()
        if decode:
            output_path = os.path.join('temp', 'key_probe', model.save_name, os.path.basename(midi_path)[:-4] + '.lab')
            midi_perf = UglyMIDI(midi_path)
            beats = midi_perf.get_beats()

            score_beats = np.arange(len(beats)) * 4
            def score_to_perf_time(score_time):
                return np.interp(score_time, score_beats, beats)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            f = open(output_path, 'w')
            for start_time, end_time, label in lab:
                perf_time = score_to_perf_time(start_time)
                f.write(f'{perf_time:.3f}\t{label}\n')
            f.close()
            timing = np.array([score_to_perf_time(i) for i in range(result.shape[0])])
            spec_data = os.path.join('temp', 'key_probe', model.save_name, os.path.basename(midi_path)[:-4] + '.spec.pkl')
            # Write (timing, spec_data) to a file
            with open(spec_data, 'wb') as f:
                pickle.dump((timing, result), f)

    if decode and gt_key_path is not None:
        lab = [[lab[0][0], lab[0][-1].replace(':', ' ').replace('Minor', 'minor').replace('Dorian', 'minor').replace('Phrygian', 'minor').replace('Lydian', 'major').replace('Mixolydian', 'major').replace('Locrian', 'minor')]]
        return eval_key(gt_key_path, lab)
    else:
        return result

def get_model(model_name):
    model_dir = model_name.split('.epoch')[0]
    if os.path.exists(f'ckpt/{model_name}'):
        model_path = f'ckpt/{model_name}'
    elif os.path.exists(f'ckpt/{model_dir}'):
        model_path = f'ckpt/{model_dir}/{model_name}'
    else:
        raise FileNotFoundError(f'Cannot find model: {model_name}')
    model_type = RoformerProberCadence if 'cadence' in model_name \
        else RoformerProberChord if 'chroma' in model_name \
        else RoformerProberChord if 'bass' in model_name \
        else RoformerProber if 'encoder' in model_name \
        else RoformerProberPlain
    assert model_type is not None
    model = model_type.load_from_checkpoint(model_path, strict=False)
    model.save_name = os.path.basename(model_path)
    model.cuda()
    model.eval()
    return model

def eval_model_key_rwc(model_name, batch_size, overlap_ratio, decode=True):
    model = get_model(model_name)
    key_metrics = []
    ids_to_test = []
    for i in tqdm(range(100)):
        key_lab_path = os.path.join(RWC_DATASET_PATH, 'keys_new', f'RM-P{i + 1:03d}.lab')
        f = open(key_lab_path, 'r')
        lines = [line.strip() for line in f.readlines() if line.strip()]
        f.close()
        if len(lines) == 1:  # Only test files with single key
            ids_to_test.append(i)
    for i in tqdm(ids_to_test):
        key_lab_path = os.path.join(RWC_DATASET_PATH, 'keys_new', f'RM-P{i + 1:03d}.lab')
        key_metrics.append(probe(model, locate_rwc(i), ins_ids=['all'], gt_key_path=key_lab_path, batch_size=batch_size, overlap_ratio=overlap_ratio, visualize=False, decode=True))
    print('RWC Key Metrics:', merge_score_mean(key_metrics))

def eval_model_key_nottingham(model_name, batch_size, overlap_ratio, limit=-1, melody_only=False):
    model = get_model(model_name)
    key_metrics = []
    key_files = os.listdir(os.path.join(NOTTINGHAM_DATASET_PATH, 'processed', 'keys'))
    if limit > 0:
        key_files = key_files[:limit]
    for file in tqdm(key_files):
        key_lab_path = os.path.join(NOTTINGHAM_DATASET_PATH, 'processed', 'keys', file)
        if melody_only:
            midi_file = os.path.join(NOTTINGHAM_DATASET_PATH, 'MIDI', 'melody', file[:-4] + '.mid')
        else:
            midi_file = os.path.join(NOTTINGHAM_DATASET_PATH, 'MIDI', file[:-4] + '.mid')
        key_metrics.append(probe(model, midi_file, ins_ids=['all'], gt_key_path=key_lab_path, batch_size=batch_size, overlap_ratio=overlap_ratio, visualize=False, decode=True))
        print(file, key_metrics[-1])
    print('Nottingham Key Metrics:', merge_score_mean(key_metrics))

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('model_name', type=str)
    args.add_argument('--overlap_ratio', type=float, default=0.125, help='Overlap ratio for chunking the input sequence')
    args.add_argument('--visualize', action='store_true', default=False, help='Whether to visualize the probe results')
    args.add_argument('--no_visualize', action='store_false', dest='visualize')
    args.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
    args = args.parse_args()
    model_name_list = args.model_name.split(',')
    for model_name in model_name_list:
        eval_model_key_nottingham(model_name, batch_size=args.batch_size, overlap_ratio=args.overlap_ratio, limit=100, melody_only=True)
        # eval_model_key_rwc(model_name, batch_size=args.batch_size, overlap_ratio=args.overlap_ratio)