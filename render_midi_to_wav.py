import os
import subprocess
from tqdm import tqdm
from settings import LA_DATASET_PATH
import numpy as np
from settings import SOUNDFONT_PATH
from joblib import Parallel, delayed


def render_midi(midi_file, output_path, sr=44100):
    wav_file = os.path.join(output_path, os.path.basename(midi_file).replace('.mid', '.wav').replace('.MID', '.wav'))
    if os.path.isfile(wav_file):
        return

    command = [
        'fluidsynth',
        '-ni',
        '-F',
        wav_file,
        '-r',
        str(sr),
        SOUNDFONT_PATH,
        midi_file
    ]

    subprocess.run(command, check=True)

def render_all_la(is_full=False):
    output_path = os.path.join(LA_DATASET_PATH, 'wav')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_list = []
    for folder in os.listdir(os.path.join(LA_DATASET_PATH, 'MIDIs')):
        file_list.extend([os.path.join(LA_DATASET_PATH, 'MIDIs', folder, file)
                          for file in os.listdir(os.path.join(LA_DATASET_PATH, 'MIDIs', folder))])
    print(len(file_list))
    if not is_full:
        # Randomly sample 1000 files
        file_list = file_list[:1000]
    Parallel(n_jobs=-1)(delayed(render_midi)(midi_file, output_path) for midi_file in tqdm(file_list))

def render_la_pitch_shifted(pitch_shift_low=-5, pitch_shift_high=7, split_id=0, total_split=1, seed=42, n_cpus=-1):
    original_midi_path = os.path.join(LA_DATASET_PATH, 'midi_pitch_shifted')
    # Get all wave files
    file_list = [os.path.join(original_midi_path, file) for file in os.listdir(original_midi_path) if file.endswith('.mid')]
    if total_split > 1:
        # Split the MIDI files for parallel processing
        np.random.seed(seed)
        np.random.shuffle(file_list)
        file_list = file_list[split_id::total_split]
    print(f'Found {len(file_list)} MIDI files for pitch shifting.')
    # Create output directory for pitch-shifted files
    output_path = os.path.join(LA_DATASET_PATH, 'wav_pitch_shifted')
    os.makedirs(output_path, exist_ok=True)
    Parallel(n_jobs=n_cpus)(
        delayed(render_midi)(midi_file, output_path) for midi_file in tqdm(file_list)
    )

def main():

    import argparse
    parser = argparse.ArgumentParser(description='Render MIDI files to WAV format.')
    parser.add_argument('--split_id', type=int, default=0, help='ID of the split for parallel processing.')
    parser.add_argument('--total_split', type=int, default=1, help='Total number of splits for parallel processing.')
    parser.add_argument('--pitch_shift_low', type=int, default=-5, help='Lowest pitch shift value.')
    parser.add_argument('--pitch_shift_high', type=int, default=7, help='Highest pitch shift value.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--n_cpus', type=int, default=-1, help='Number of CPUs to use for parallel processing. -1 means using all available CPUs.')
    args = parser.parse_args()
    render_la_pitch_shifted(
        pitch_shift_low=args.pitch_shift_low,
        pitch_shift_high=args.pitch_shift_high,
        split_id=args.split_id,
        total_split=args.total_split,
        seed=args.seed,
        n_cpus=args.n_cpus
    )

if __name__ == '__main__':
    render_all_la(is_full=True)