from cp_transformer_probe import RoformerProber
import os
import sys
import torch
from tqdm import tqdm
from settings import LA_DATASET_PATH
from preprocess_large_midi_dataset import preprocess_midi
from cp_transformer_fine_tune import decompress
from pretty_midi_fix import UglyMIDI
import numpy as np
import argparse
import time
import subprocess
from joblib import Parallel, delayed
from key_names import KEY_MAP, MODE_LOOKUP
from hmm import hmm_decode

VISUALIZE = False

def cadence_based_mode_estimation(cadence, start_time, end_time, scale_id, majmin_only=False):
    histogram = cadence[start_time:end_time, 12:].sum(axis=0)
    best_mode_id = 0
    for mode_id in MODE_LOOKUP:
        if majmin_only and mode_id not in [0, 9]:
            continue
        score = histogram[(mode_id + scale_id) % 12]
        if score > histogram[(best_mode_id + scale_id) % 12]:
            best_mode_id = mode_id
    best_mode = MODE_LOOKUP[best_mode_id]
    key_name = KEY_MAP[best_mode][scale_id]
    # old_key_name = f'{TOKEN_ID_LABELS[(scale_id + best_mode_id) % 12]}:{MODES[best_mode_id]}'
    return key_name

def model_key_probe(model, midi_path, n_samples, batch_size, chunk_length=384):
    assert n_samples == 1
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
            output = model.inference(x1p_collection)
            for i in range(batch_hops * batch_id, min(batch_hops * (batch_id + 1), n_chunks * 2 - 1)):
                sample_start = (i - batch_hops * batch_id) * n_samples
                batch_output = output[sample_start:sample_start + n_samples]
                target_length = chunk_length
                batch_output = batch_output[:, prev_end_frame:target_length]
                prev_end_frame = target_length - chunk_length // 2
                results.append(batch_output)
    results = torch.cat(results, dim=1).squeeze(0)
    # Pool 4 subbeats
    results = results[:results.shape[0] // 4 * 4]
    results = results.view(results.shape[0] // 4, 4, 12).mean(dim=1)
    return results


def filter_and_create_midi_key_dataset(model_name, n_gpus, gpu_id, n_samples=1, batch_size=32, global_seed=42):
    # Create output directory
    output_dir = os.path.join('temp', 'la_labels', 'v3', 'raw_keys')
    os.makedirs(output_dir, exist_ok=True)

    f = open('data/la_dataset_filtered.txt', 'r')
    file_list = [line.strip() for line in f.readlines() if line.strip()]
    if global_seed is not None:
        np.random.seed(global_seed)
        np.random.shuffle(file_list)
    print(f'Total MIDI files: {len(file_list)}')
    # Separate the file list into n_gpus parts
    file_list = file_list[gpu_id::n_gpus]
    print(f'Processing {len(file_list)} files on GPU {gpu_id}')
    model = RoformerProber.load_from_checkpoint(model_name, strict=False)
    model.cuda()
    model.eval()
    for file in tqdm(file_list, file=sys.stderr):
        output_file = os.path.join(output_dir, os.path.basename(file)[:-4] + '.pt')
        if os.path.exists(output_file):
            continue
        try:
            raw_outputs = model_key_probe(model, os.path.join(LA_DATASET_PATH, 'MIDIs', file[0], file), batch_size=batch_size, n_samples=n_samples)
        except:
            print(f'Error processing {file}', file=sys.stderr)
            continue
        if VISUALIZE:
            import matplotlib.pyplot as plt
            plt.imshow(raw_outputs.cpu().numpy().T, aspect='auto', origin='lower', interpolation='none')
            plt.colorbar()
            plt.title(f'Raw Key Outputs for {file}')
            plt.xlabel('Time (subbeats)')
            plt.ylabel('Key')
            plt.show()
        torch.save(raw_outputs, output_file)
    print(f'Processed {len(file_list)} files, results saved to {output_dir}')

def decode_keys():
    majmin_only = True
    input_dir_scale = os.path.join('temp', 'la_labels', 'v3', 'raw_keys')
    input_dir_cadence = os.path.join('temp', 'la_labels', 'v2', 'raw_cadence')
    output_dir2 = os.path.join('temp', 'la_labels', 'v3', 'decoded_keys_majmin')
    os.makedirs(output_dir2, exist_ok=True)
    file_list = os.listdir(input_dir_scale)
    process_file_list = []
    for file in tqdm(file_list):
        output_path = os.path.join(output_dir2, file[:-3] + '.lab')
        midi_path = os.path.join(LA_DATASET_PATH, 'MIDIs', file[0], file[:-3] + '.mid')
        scale_path = os.path.join(input_dir_scale, file)
        cadence_path = os.path.join(input_dir_cadence, file)
        process_file_list.append((midi_path, scale_path, cadence_path, output_path))
    Parallel(n_jobs=-1)(delayed(decode_key)(midi_path, input_chord_path, input_key_path, output_path, majmin_only=majmin_only) for midi_path, input_chord_path, input_key_path, output_path in tqdm(process_file_list))

def decode_key(midi_path, input_scale_path, input_cadence_path, output_path, majmin_only=False):
    try:
        scale = torch.load(input_scale_path, map_location='cpu')
    except:
        print(f'Could not load {input_scale_path}')
        return
    try:
        cadences = torch.load(input_cadence_path, map_location='cpu')
    except:
        print(f'Could not load {input_cadence_path}')
        return
    score_midi = UglyMIDI(midi_path, constant_tempo=60)
    has_pitched_instrument = any(not ins.is_drum and len(ins.notes) > 0 for ins in score_midi.instruments)
    if not has_pitched_instrument:
        print(f'No pitched instrument in {midi_path}, skipping.')
        return
    end_frame = int((score_midi.get_end_time()))
    scale = scale[:end_frame]
    cadences = cadences[:end_frame]
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
    scale = scale.cpu().numpy()
    try:
        decoded_seq = hmm_decode(torch.from_numpy(np.log(scale))[None], transition_penalty=100.0).squeeze(0)
    except:
        print(f'Could not decode {input_scale_path}')
        return
    lab = []
    for i in range(decoded_seq.shape[0]):
        if i == 0 or decoded_seq[i] != decoded_seq[i - 1]:
            if i > 0:
                lab[-1][1] = i
            lab.append([i, None, decoded_seq[i].item()])
    lab[-1][1] = decoded_seq.shape[0]
    for line in lab:
        line[2] = cadence_based_mode_estimation(cadences, line[0], line[1], line[2], majmin_only=majmin_only)
    f = open(output_path, 'w')
    for key in lab:
        start_frame = int(key[0])
        end_frame = int(key[1])
        key_label = key[2]
        f.write(f'{score_time_to_perf(start_frame)}\t{score_time_to_perf(end_frame)}\t{key_label}\n')
    f.close()

def start_batch():
    args = argparse.ArgumentParser()
    args.add_argument('--model_name', type=str)
    args.add_argument('--n_gpus', type=int, default=1)
    args.add_argument('--gpu_id', type=int, default=None, help='GPU ID to use for the process')
    args.add_argument('--n_samples', type=int, default=1)
    args.add_argument('--batch_size', type=int, default=4)
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
                python_path, 'bootstrap_key.py',
                '--model_name', args.model_name,
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
        filter_and_create_midi_key_dataset(args.model_name, args.n_gpus, args.gpu_id, n_samples=args.n_samples, batch_size=args.batch_size)

if __name__ == '__main__':
    decode_keys()