import time

from joblib import Parallel, delayed
import pretty_midi

from chord_eval import (model_chord_recognition, model_chord_probe, rule_based_chord_recognition, aggregate_data)
from cp_transformer_probe_chord import RoformerProberChord
from settings import LA_DATASET_PATH
from preprocess_large_midi_dataset import filter_la_quantization
import os
import sys
import argparse
from tqdm import tqdm
from pretty_midi_fix import UglyMIDI
import subprocess
import numpy as np
import torch
from correct_chord_spelling import correct_chord_spelling

def filter_and_create_midi_chord_dataset(model_name, n_gpus, gpu_id, n_samples=16, batch_size=32, bass_model_name=None):
    # Create output directory
    output_dir = os.path.join('temp', 'la_labels', 'v3-unsupervised', 'raw_chords')
    os.makedirs(output_dir, exist_ok=True)

    f = open('data/la_dataset_filtered.txt', 'r')
    file_list = [line.strip() for line in f.readlines() if line.strip()]

    print(f'Total MIDI files: {len(file_list)}')
    # Separate the file list into n_gpus parts
    file_list = file_list[gpu_id::n_gpus]
    print(f'Processing {len(file_list)} files on GPU {gpu_id}')
    model = RoformerProberChord.load_from_checkpoint(model_name, strict=False)
    model.cuda()
    model.eval()
    if bass_model_name is not None:
        bass_model = RoformerProberChord.load_from_checkpoint(bass_model_name, strict=False)
        bass_model.cuda()
        bass_model.eval()
    else:
        bass_model = None
    for file in tqdm(file_list, file=sys.stderr):
        output_file = os.path.join(output_dir, os.path.basename(file)[:-4] + '.pt')
        if os.path.exists(output_file):
            continue
        try:
            raw_outputs = model_chord_probe(model, os.path.join(LA_DATASET_PATH, 'MIDIs', file[0], file), batch_size=batch_size, n_samples=n_samples, return_raw_output=True, bass_model=bass_model)
            torch.save(raw_outputs, output_file)
        except Exception as e:
            print(f'Error processing {file}: {e}', file=sys.stderr)
            continue
    print(f'Processed {len(file_list)} files, results saved to {output_dir}')

def decode_chords():
    input_dir = os.path.join('temp', 'la_labels', 'v3-unsupervised', 'raw_chords')
    input_dir_key = os.path.join('temp', 'la_labels', 'v3-unsupervised', 'decoded_keys')
    # output_dir = os.path.join('temp', 'la_labels', 'v3-unsupervised', 'midi_chords')
    output_dir2 = os.path.join('temp', 'la_labels', 'v3-unsupervised', 'decoded_chords')
    # os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    file_list = os.listdir(input_dir)
    process_file_list = []
    for file in tqdm(file_list):
        output_path = os.path.join(output_dir2, file[:-3] + '.lab')
        midi_path = os.path.join(LA_DATASET_PATH, 'MIDIs', file[0], file[:-3] + '.mid')
        chord_path = os.path.join(input_dir, file)
        key_path = os.path.join(input_dir_key, file[:-3] + '.lab')
        process_file_list.append((midi_path, chord_path, key_path, output_path))
    Parallel(n_jobs=-1)(delayed(decode_chord)(midi_path, input_chord_path, input_key_path, output_path) for midi_path, input_chord_path, input_key_path, output_path in tqdm(process_file_list))

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
    if start_time is None or end_time is None:
        return None, None
    return int(np.round(start_time) * 4) // 4, int(np.round(end_time) * 4 - 1) // 4 + 1

def decode_chord(midi_path, input_chord_path, input_key_path, output_path):
    try:
        results = torch.load(input_chord_path, map_location='cpu')
    except:
        print(f'Could not load {input_chord_path}')
        return
    try:
        f = open(input_key_path, 'r')
        lines = [line.strip().split('\t') for line in f.readlines() if line.strip()]
        f.close()
        key_starts = np.array([float(line[0]) for line in lines], dtype=np.float32)
        key_labels = [line[2] for line in lines]
    except:
        print(f'Could not load {input_key_path}')
        return
    score_midi = UglyMIDI(midi_path, constant_tempo=60)
    has_pitched_instrument = any(not ins.is_drum and ins.program < 0x70 and len(ins.notes) > 0 for ins in score_midi.instruments)
    if not has_pitched_instrument:
        print(f'No pitched instrument in {midi_path}, skipping.')
        return
    results = results.cpu().numpy()
    start_time, end_time = get_voiced_time_range(score_midi)
    if start_time is not None: # clear silence at the beginning and end
        results[:start_time] = 0
        results[end_time:] = 0
    chroma_data = results[:, :12]
    bass_data = results[:, 12:]
    try:
        performance_midi = UglyMIDI(midi_path)
    except:
        print(f'Could not load performance MIDI {midi_path}')
        return
    perf_to_score_mapping = [(0, 0)]  # Initialize with the start time
    for ins_p, ins_s in zip(performance_midi.instruments, score_midi.instruments):
        for note_p, note_s in zip(ins_p.notes, ins_s.notes):
            perf_to_score_mapping.append((note_p.start, note_s.start))
            perf_to_score_mapping.append((note_p.end, note_s.end))
        for pitch_bend_p, pitch_bend_s in zip(ins_p.pitch_bends, ins_s.pitch_bends):
            perf_to_score_mapping.append((pitch_bend_p.time, pitch_bend_s.time))
    perf_to_score_mapping.sort(key=lambda x: x[0])
    perf_to_score_mapping = np.array(perf_to_score_mapping, dtype=np.float32)
    def score_time_to_perf(score_time):
        return np.interp(score_time, perf_to_score_mapping[:, 1], perf_to_score_mapping[:, 0])
    def get_key_at_time(perf_time):
        idx = np.searchsorted(key_starts, perf_time) - 1
        if idx < 0:
            return key_labels[0]
        return key_labels[idx]
    try:
        chords = rule_based_chord_recognition(score_midi, replace_chroma=(bass_data, chroma_data), use_transition=False)
    except Exception as e:
        print(f'Error processing {midi_path}: {e}')
        return
    f = open(output_path, 'w')

    for chord in chords:
        start_frame = int(chord[0])
        end_frame = int(chord[1])
        key = get_key_at_time(score_time_to_perf((start_frame + end_frame) / 2))
        chord_label = correct_chord_spelling(chord[2], key)
        f.write(f'{score_time_to_perf(chord[0])}\t{score_time_to_perf(chord[1])}\t{chord_label}\n')
    f.close()



def start_batch():
    args = argparse.ArgumentParser()
    args.add_argument('--model_name', type=str, default='ckpt/midi-unsupervised-models/cp_transformer_yinyang_probe_encoder_chroma_v0.4_batch_160_la_chords_16_v8_chroma.epoch=00.val_loss=0.15761.ckpt')
    args.add_argument('--bass_model_name', type=str, default='ckpt/midi-unsupervised-models/cp_transformer_yinyang_probe_encoder_chroma_v0.5_batch_160_la_bass_16_v2.epoch=00.val_loss=0.10736.ckpt')
    args.add_argument('--n_gpus', type=int, default=1)
    args.add_argument('--gpu_id', type=int, default=None, help='GPU ID to use for the process')
    args.add_argument('--n_samples', type=int, default=1)
    args.add_argument('--batch_size', type=int, default=16)
    args = args.parse_args()
    processes = []
    if args.gpu_id is None:
        for gpu_id in range(args.n_gpus):
            print(f'Starting process for GPU {gpu_id}')
            # Start a separate process for each GPU
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            python_path = sys.executable
            cmd = [
                python_path, 'bootstrap_chord.py',
                '--model_name', args.model_name,
                '--bass_model_name', args.bass_model_name,
                '--n_gpus', str(args.n_gpus),
                '--gpu_id', str(gpu_id),
                '--n_samples', str(args.n_samples),
                '--batch_size', str(args.batch_size),
            ]
            print(f"command: {subprocess.list2cmdline(cmd)}")
            # Do not store sys.output and throw it into the void
            result = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL)
            processes.append(result)
            print(f'Process started for GPU {gpu_id} with PID {result.pid}')
        while True:
            # Sleep for 10 seconds
            time.sleep(10)

            # Track if any process fails
            for process in processes:
                if process.returncode is not None and process.returncode != 0:
                    # Send signal to terminate all processes
                    print(f'Process {process.pid} failed with return code {process.returncode}, terminating all processes.')
                    for p in processes:
                        if p.poll() is None:
                            p.terminate()
                    for p in processes:
                        p.wait()
                    print(f'Process {process.pid} failed with return code {process.returncode}')
                    sys.exit(1)

            # Track if all processes are completed
            if all(process.returncode is not None for process in processes):
                print('All processes completed successfully.')
                sys.exit(0)

    else:
        print('Child process started for GPU ID:', args.gpu_id, file=sys.stderr)
        filter_and_create_midi_chord_dataset(args.model_name, args.n_gpus, args.gpu_id, n_samples=args.n_samples, batch_size=args.batch_size, bass_model_name=args.bass_model_name)

if __name__ == '__main__':
    decode_chords()
