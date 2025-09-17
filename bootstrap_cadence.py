from cp_transformer_probe_cadence import RoformerProberCadence
import os
import sys
import torch
from tqdm import tqdm
from settings import LA_DATASET_PATH
from preprocess_large_midi_dataset import preprocess_midi
from pretty_midi_fix import UglyMIDI
from cp_transformer_fine_tune import decompress
import numpy as np
import argparse
import time
import subprocess

VISUALIZE = False

def model_cadence_probe(model, midi_path, n_samples, batch_size, chunk_length=384, fixed_start=None, fixed_end=None):
    assert n_samples == 1
    torch.random.manual_seed(42)
    np.random.seed(42)
    x1, _ = decompress(model, preprocess_midi(midi_path, 16, ins_ids=['all', 'all'], filter=False)[0])
    raw_length = x1.shape[1]
    if fixed_start is not None:
        raw_length = fixed_end - fixed_start
    else:
        fixed_start = 0
    n_chunks = (raw_length - 1) // chunk_length + 1
    expected_length = n_chunks * chunk_length
    chunk_step = chunk_length // 2
    # TODO: this seems bad
    x1, _ = decompress(model, preprocess_midi(midi_path, 16, ins_ids=['all', 'all'], filter=False, fixed_length=expected_length + fixed_start)[0])
    x1 = x1[:, fixed_start:]
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
    results = results.view(results.shape[0] // 4, 4, 24).mean(dim=1)
    return results


def filter_and_create_midi_cadence_dataset(model_name, n_gpus, gpu_id, n_samples=1, batch_size=32, global_seed=42):
    # Create output directory
    output_dir = os.path.join('temp', 'la_labels', 'v2', 'raw_cadence')
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
    model = RoformerProberCadence.load_from_checkpoint(model_name, strict=False)
    model.cuda()
    model.eval()
    for file in tqdm(file_list, file=sys.stderr):
        output_file = os.path.join(output_dir, os.path.basename(file)[:-4] + '.pt')
        if os.path.exists(output_file):
            continue
        try:
            raw_outputs = model_cadence_probe(model, os.path.join(LA_DATASET_PATH, 'MIDIs', file[0], file), batch_size=batch_size, n_samples=n_samples)
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
                python_path, 'bootstrap_cadence.py',
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
        filter_and_create_midi_cadence_dataset(args.model_name, args.n_gpus, args.gpu_id, n_samples=args.n_samples, batch_size=args.batch_size)

if __name__ == '__main__':
    start_batch()