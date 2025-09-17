import numpy as np
from pretty_midi_fix import UglyMIDI
import pretty_midi
from settings import RWC_DATASET_PATH, LA_DATASET_PATH
import os
from joblib import Parallel, delayed
import torch
import shutil

tokenize_dict = {'<sos>': 0, '<eos>': 1, '<pad>': 2}
tokenize_count = [-1, -1, -1]

DURATION_TEMPLATES = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096])

def filter_la_tempo_changes():
    la_folder = os.path.join(LA_DATASET_PATH, 'MIDIs')
    output_folder = os.path.join(LA_DATASET_PATH, 'with_tempo_changes')
    for folder in os.listdir(la_folder):
        for file in os.listdir(os.path.join(la_folder, folder)):
            midi_path = os.path.join(la_folder, folder, file)
            try:
                midi = pretty_midi.PrettyMIDI(midi_path)
            except:
                continue
            tempo_changes = midi.get_tempo_changes()
            if len(tempo_changes[0]) > 3:
                for ins in midi.instruments:
                    if ins.is_drum: # has a drum track
                        shutil.copy(midi_path, os.path.join(output_folder, file))

def analyze_la_quantization(midi_path, beat_div=4):
    try:
        # Read the midi as a score MIDI
        # E.g., a MIDI note with start_time = 3.0 means it starts at the 3rd subbeat
        midi = UglyMIDI(midi_path, constant_tempo=60.0 / beat_div)
    except:
        return None
    midi_end_time = int(midi.get_end_time())
    if midi_end_time <= 0:
        return None
    best_statistics = 1.0
    for i, ins in enumerate(midi.instruments):
        if len(ins.notes) > 20:
            statistics = np.zeros(beat_div, dtype=np.uint32)
            for note in ins.notes:
                start_time = int(round(note.start))
                statistics[start_time % beat_div] += 1
            statistics = statistics[1::2].sum() / len(ins.notes)
            best_statistics = min(best_statistics, statistics)
    return best_statistics

def filter_la_quantization(midi, beat_div=4):
    best_statistics = 1.0
    for i, ins in enumerate(midi.instruments):
        if len(ins.notes) > 20:
            statistics = np.zeros(beat_div, dtype=np.uint32)
            for note in ins.notes:
                start_time = int(round(note.start))
                statistics[start_time % beat_div] += 1
            statistics = statistics[1::2].sum() / len(ins.notes)
            best_statistics = min(best_statistics, statistics)
    # best_statistics < 4.5: likely well-quantized
    # best_statistics > 6.5: totally wrong beats, but tempo seems to be correct. Model can learn something from it.
    return best_statistics <= 0.45 or 0.65 < best_statistics < 1.0

def get_density(ins, sparse_beat_group=4):
    # Quantize using beat_div, then evaluate how many onsets are on sparse_beat_group
    return len([note for note in ins.notes if int(round(note.start)) % sparse_beat_group == 0]) / len(ins.notes)

def preprocess_midi(midi_path, max_polyphony, beat_div=4, ins_ids='all', filter=True, dedup=False, fixed_length=-1, return_midi_ins=False):
    '''
    Preprocess a MIDI file into a tensor representation
    :param midi_path: Path to the MIDI file
    :param max_polyphony: Maximum number of notes that can be played at the same time
    :param beat_div: Number of subbeats per beat
    :param ins_ids: List of instrument ids to include, or 'all' to include all instruments
    :return: A tensor representation of the MIDI file
    '''
    try:
        if isinstance(midi_path, pretty_midi.PrettyMIDI):
            midi = midi_path
        else:
            # Read the midi as a score MIDI
            # E.g., a MIDI note with start_time = 3.0 means it starts at the 3rd subbeat
            midi = UglyMIDI(midi_path, constant_tempo=60.0 / beat_div)
    except:
        return None
    midi_end_time = int(midi.get_end_time())
    if fixed_length >= 0:
        midi_end_time = fixed_length
    if midi_end_time <= 0:
        return None
    if filter and not filter_la_quantization(midi, beat_div):
        return None
    if not isinstance(ins_ids, list):
        ins_ids = [ins_ids]
    if ins_ids[0].startswith('dense'):
        if len(midi.instruments) > 1:
            densities = [get_density(ins) for i, ins in enumerate(midi.instruments)]
            if min(densities) < 0.7 and max(densities) > 0.8:
                # Good density distribution
                pass
            else:
                return None
        else:
            return None
    if ins_ids[0].startswith('random'):
        if len(midi.instruments) > 1:
            ins_random_split = np.zeros(len(midi.instruments), dtype=np.uint8)
            split_at = np.random.randint(1, len(midi.instruments))
            ins_random_split[np.random.permutation(len(midi.instruments))[:split_at]] = 1
        else:
            return None  # invalid midi file
    else:
        ins_random_split = None
    duration_boundaries = (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2
    # Trace max and min pitches to calculate the possible pitch shift range
    min_pitch = 127
    max_pitch = 0
    result_rolls = []
    result_midi_ins = []
    for ins_id in ins_ids:
        if return_midi_ins:
            result_midi_ins.append([])
        repeat_notes_dict = {}  # do dedup per instrument id
        has_any_note = False
        rolls = np.full((midi_end_time, max_polyphony, 4), dtype=np.uint8, fill_value=255)
        polyphony_counts = np.zeros(midi_end_time, dtype=np.uint8)
        for i, ins in enumerate(midi.instruments):
            output_midi_ins = pretty_midi.Instrument(program=ins.program, is_drum=ins.is_drum) if return_midi_ins else None
            program = ins.program
            if ins.is_drum:
                program = 127
            for note in ins.notes:
                start_time = int(round(note.start))
                end_time = int(round(note.end))
                if start_time >= 0 and start_time < midi_end_time and polyphony_counts[start_time] < max_polyphony:
                    duration = np.searchsorted(duration_boundaries, end_time - start_time)
                    if not ins.is_drum:
                        min_pitch = min(min_pitch, note.pitch)
                        max_pitch = max(max_pitch, note.pitch)
                    # If ins_id is not 'all', filter notes by ins_id
                    add_note = False
                    if ins_id == 'all':
                        add_note = True
                    elif isinstance(ins_id, int):
                        # Filter by program number, not implemented yet
                        raise NotImplementedError
                    elif isinstance(ins_id, str):
                        if '-' in ins_id:
                            # Filter by track ID, e.g., 'track-0', 'upto-1', 'from-2', 'notrack-3'
                            task, num = ins_id.split('-')
                            num = int(num)
                            if task == 'track':
                                add_note = i == num
                            elif task == 'upto':
                                add_note = i <= num
                            elif task == 'from':
                                add_note = i >= num
                            elif task == 'notrack':
                                add_note = i != num
                            elif task == 'random':
                                add_note = ins_random_split[i] == num
                            else:
                                raise NotImplementedError
                        elif ins_id == 'dense':
                            add_note = densities[i] <= 0.75
                        elif ins_id == 'sparse':
                            add_note = densities[i] > 0.75
                        elif ins_id == 'drum':
                            # Drum only
                            add_note = ins.is_drum
                        elif ins_id == 'nondrum':
                            # Non-drum only
                            add_note = not ins.is_drum
                        elif ins_id == 'empty':
                            # Empty track
                            add_note = False
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                    if add_note:
                        has_any_note = True
                        if return_midi_ins:
                            output_midi_ins.notes.append(pretty_midi.Note(start=note.start, end=note.end, pitch=note.pitch, velocity=note.velocity))
                        if dedup:
                            if (program, start_time, note.pitch) in repeat_notes_dict:
                                [prev_duration, prev_index] = repeat_notes_dict[(program, start_time, note.pitch)]
                                if prev_duration < duration:
                                    # Keep the longer note
                                    rolls[start_time, prev_index] = [program, note.pitch, duration, note.velocity]
                                    repeat_notes_dict[(program, start_time, note.pitch)] = [duration, prev_index]
                            else:
                                repeat_notes_dict[(program, start_time, note.pitch)] = [duration, polyphony_counts[start_time]]
                                # Add note to the roll
                                rolls[start_time, polyphony_counts[start_time]] = [program, note.pitch, duration, note.velocity]
                                polyphony_counts[start_time] += 1
                        else:
                            # Add note to the roll
                            rolls[start_time, polyphony_counts[start_time]] = [program, note.pitch, duration, note.velocity]
                            polyphony_counts[start_time] += 1
            if return_midi_ins and len(output_midi_ins.notes) > 0:
                result_midi_ins[-1].append(output_midi_ins)
        if not has_any_note and ins_id != 'empty':
            return None  # invalid midi file
        for i in range(midi_end_time):
            # Sort notes by ins first, then by pitch, then by duration
            rolls[i, :polyphony_counts[i]] = rolls[i, :polyphony_counts[i]][np.lexsort((rolls[i, :polyphony_counts[i], 2], rolls[i, :polyphony_counts[i], 1], rolls[i, :polyphony_counts[i], 0], rolls[i, :polyphony_counts[i], 3]))]
            if polyphony_counts[i] < max_polyphony:
                rolls[i, polyphony_counts[i], 0] = 254  # EOS token
        result_rolls.append(rolls)
    if return_midi_ins:
        return result_midi_ins
    result_rolls = np.concatenate(result_rolls, axis=1)
    # Get song-level pitch shift range
    pitch_shift_max = 127 - max_pitch
    pitch_shift_min = -min_pitch
    # Convert to tensor
    return torch.tensor(result_rolls.reshape(midi_end_time, -1)), torch.tensor([pitch_shift_min, pitch_shift_max], dtype=torch.int8)


def create_npy_dataset_from_midi(folder, max_polyphony, dataset_name, ins_ids='all', scan_subfolders=True, dedup=False, max_idx=None):
    # Get all midi files in the folder, recursively
    midi_files = []
    if scan_subfolders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.mid') or file.endswith('.MID'):
                    midi_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder):
            if file.endswith('.mid') or file.endswith('.MID'):
                midi_files.append(os.path.join(folder, file))
    if max_idx is not None:
        midi_files = midi_files[:max_idx]
    # Process files in parallel
    print(f'Processing {len(midi_files)} files')
    results = Parallel(n_jobs=-1, verbose=10)(delayed(preprocess_midi)(midi_file, max_polyphony, ins_ids=ins_ids, dedup=dedup) for midi_file in midi_files)
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

def create_rwc_cp(max_polyphony=16):
    rwc_folder = os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC')
    create_npy_dataset_from_midi(rwc_folder, max_polyphony, f'rwc_cp{max_polyphony}_v2')

def create_la_cp(max_polyphony=16):
    la_folder = os.path.join(LA_DATASET_PATH, 'MIDIs')
    create_npy_dataset_from_midi(la_folder, max_polyphony, f'la_cp{max_polyphony}_v2')

def create_nottingham_parts(max_polyphony=8):
    rwc_folder = os.path.join(NOTTINGHAM_DATASET_PATH, 'MIDI')
    create_npy_dataset_from_midi(rwc_folder, max_polyphony, f'nottingham_cp{max_polyphony}_v2_chord_mel', ins_ids=['track-1', 'track-0'], scan_subfolders=False)

def create_transposed_nottingham_test(max_polyphony=8):
    rwc_folder = os.path.join(R'E:\Programming\melodyt5\transposed_midi')
    create_npy_dataset_from_midi(rwc_folder, max_polyphony, f'transposed_nottingham_test_cp{max_polyphony}_v2_chord_mel', ins_ids=['track-1', 'track-0'], scan_subfolders=False)

def create_rwc_drums(max_polyphony=16):
    rwc_folder = os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC')
    create_npy_dataset_from_midi(rwc_folder, max_polyphony, f'rwc_cp{max_polyphony}_v2_drums_nondrum', ins_ids=['drum', 'nondrum'])

def create_rwc_drums_fix(max_polyphony=16):
    rwc_folder = os.path.join(RWC_DATASET_PATH, 'RemoveChannel10')
    create_npy_dataset_from_midi(rwc_folder, max_polyphony, f'rwc_cp{max_polyphony}_v2_drums_nondrum_fix', ins_ids=['drum', 'nondrum'])

def create_la_drums(max_polyphony=16):
    la_folder = os.path.join(LA_DATASET_PATH, 'MIDIs')
    create_npy_dataset_from_midi(la_folder, max_polyphony, f'la_cp{max_polyphony}_v2_drums_nondrum_dedup', ins_ids=['drum', 'nondrum'], max_idx=50000, dedup=True)

def create_rwc_chords(max_polyphony=16):
    rwc_folder = os.path.join('temp', 'rwc_chord')
    create_npy_dataset_from_midi(rwc_folder, max_polyphony, f'rwc_cp{max_polyphony}_v3_chordlab', ins_ids=['from-1', 'upto-0'], scan_subfolders=False)

def create_rwc_chords_v1(max_polyphony=16):
    rwc_folder = os.path.join('temp', 'rwc_chord_v1')
    create_npy_dataset_from_midi(rwc_folder, max_polyphony, f'rwc_cp{max_polyphony}_v1_chordlab', ins_ids=['from-1', 'upto-0'], scan_subfolders=False)

def create_rwc_majmin_chords(max_polyphony=16):
    rwc_folder = os.path.join('temp', 'rwc_chord_majmin')
    create_npy_dataset_from_midi(rwc_folder, max_polyphony, f'rwc_cp{max_polyphony}_v2_majmin', ins_ids=['from-1', 'upto-0'], scan_subfolders=False)

def create_la_random_split(max_polyphony=16):
    la_folder = os.path.join(LA_DATASET_PATH, 'MIDIs')
    create_npy_dataset_from_midi(la_folder, max_polyphony, f'la_cp{max_polyphony}_v2_random_dedup', ins_ids=['random-0', 'random-1'], max_idx=50000, dedup=True)

def create_la_density(max_polyphony=16):
    la_folder = os.path.join(LA_DATASET_PATH, 'MIDIs')
    create_npy_dataset_from_midi(la_folder, max_polyphony, f'la_cp{max_polyphony}_v2_density', ins_ids=['dense', 'sparse'], max_idx=50000)

def create_la_med(max_polyphony=16):
    la_folder = os.path.join(LA_DATASET_PATH, 'MIDIs')
    create_npy_dataset_from_midi(la_folder, max_polyphony, f'la_cp{max_polyphony}_v2_med_dedup', max_idx=50000, dedup=True)

def create_la_chords(max_polyphony=16):
    rwc_folder = os.path.join('temp', 'la_chord')
    create_npy_dataset_from_midi(rwc_folder, max_polyphony, f'la_cp{max_polyphony}_v1_chordlab', ins_ids=['from-1', 'upto-0'], scan_subfolders=False)

if __name__ == '__main__':
    create_nottingham_parts()