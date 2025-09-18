from settings import RWC_DATASET_PATH, LA_DATASET_PATH
import os
from pretty_midi_fix import UglyMIDI
import pretty_midi
import numpy as np
np.int = int
from mir_eval.chord import encode, rotate_bitmap_to_root, QUALITIES
from joblib import Parallel, delayed
from tqdm import tqdm

def chord_to_midi(chord_str, bass_starting_pitch=36):
    root_number, semitone_bitmap, bass_number = encode(chord_str)
    if root_number < 0:
        return []
    bass_number = (bass_number + root_number) % 12
    semitone_bitmap = rotate_bitmap_to_root(semitone_bitmap, root_number)
    pitches = [bass_number + bass_starting_pitch]
    _, extended_semitone_bitmap, _ = encode(chord_str, reduce_extended_chords=True)
    extended_semitone_bitmap = rotate_bitmap_to_root(extended_semitone_bitmap, root_number)
    for i in range(12):
        if semitone_bitmap[i]:
            pitches.append(i + bass_starting_pitch + 12)
        elif extended_semitone_bitmap[i]:
            pitches.append(i + bass_starting_pitch + 24)
    return pitches

def chord_to_majmin(chord_str):
    # Input: chord quality
    # Output: 0 - 11 for maj, 12 - 23 for min, 24 for no chord, 25 for others
    root_number, semitone_bitmap, bass_number = encode(chord_str)
    if root_number < 0:
        return 24
    major_quality = QUALITIES['maj']
    minor_quality = QUALITIES['min']
    contains_major = np.all(np.logical_and(semitone_bitmap, major_quality) == major_quality)
    contains_minor = np.all(np.logical_and(semitone_bitmap, minor_quality) == minor_quality)
    if contains_major and not contains_minor:
        return root_number
    elif contains_minor and not contains_major:
        return root_number + 12
    return -1

def create_drum_track(beat_time, downbeat_time, subbeat_time_boundaries, unit_time):
    drum_track = pretty_midi.Instrument(program=0, is_drum=True)
    def quantize_time(time):
        return np.searchsorted(subbeat_time_boundaries, time)
    quantized_downbeats = quantize_time(downbeat_time)
    start_id = quantized_downbeats[0]
    if quantized_downbeats[2] - quantized_downbeats[1] != quantized_downbeats[1] - quantized_downbeats[0]:
        start_id = quantized_downbeats[1]
    quantized_beats = np.arange(len(subbeat_time_boundaries) + 1)
    quantized_stronger_beats = quantize_time(np.interp(np.arange(len(downbeat_time) - 1) + 0.5, np.arange(len(downbeat_time)), downbeat_time))
    downbeat_ins = pretty_midi.DRUM_MAP.index('Acoustic Bass Drum') + 35
    stronger_beat_ins = pretty_midi.DRUM_MAP.index('Acoustic Snare') + 35
    beat_ins = pretty_midi.DRUM_MAP.index('Closed Hi Hat') + 35
    for t in quantized_downbeats:
        if t >= start_id:
            drum_track.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=downbeat_ins,
                start=t * unit_time,
                end=(t + 1) * unit_time
            ))
    for t in quantized_stronger_beats:
        if t >= start_id:
            drum_track.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=stronger_beat_ins,
                start=t * unit_time,
                end=(t + 1) * unit_time
            ))
    for t in quantized_beats:
        if t >= start_id:
            drum_track.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=beat_ins,
                start=t * unit_time,
                end=(t + 1) * unit_time
            ))
    return drum_track

def create_half_speed_midi(midi_path, output_path):
    try:
        score_midi = UglyMIDI(midi_path, constant_tempo=120.0)
    except:
        print('Failed to load', midi_path)
        return
    for ins in score_midi.instruments:
        ins.pitch_bends.clear()  # remove pitch bends
        for note in ins.notes:
            note.start *= 2
            note.end *= 2
    score_midi.write(output_path)

def add_chord_track(midi_path, chord_lab_path, output_path, subbeat_div=2, shift=0, quality_only=False):
    performance_midi = UglyMIDI(midi_path)
    score_midi = UglyMIDI(midi_path, constant_tempo=120.0)
    beat_time = performance_midi.get_beats()
    downbeat_time = performance_midi.get_downbeats()
    # interpolate to get subbeat time
    n_beats = len(beat_time)
    subbeat_indices = np.arange((n_beats - 1) * subbeat_div + 1) / subbeat_div
    subbeat_time = np.interp(subbeat_indices, np.arange(n_beats), beat_time)
    subbeat_time_boundaries = (subbeat_time[:-1] + subbeat_time[1:]) / 2
    chord_track = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('String Ensemble 1'))
    if chord_lab_path is not None:
        f = open(chord_lab_path, 'r')
        lines = [line.strip() for line in f.readlines() if line.strip()]
        f.close()
    else:
        lines = []  # no chord labels
    def quantize_time(time):
        return np.searchsorted(subbeat_time_boundaries, time)
    quantized_downbeats = quantize_time(downbeat_time)
    quantized_downbeats = np.concatenate([[-np.inf], quantized_downbeats, [np.inf]])
    for line in lines:
        start_time, end_time, chord = line.split('\t')
        start_time = float(start_time)
        end_time = float(end_time)
        start_time_quantized = quantize_time(start_time)
        end_time_quantized = quantize_time(end_time)
        if quality_only:
            quality_index = chord_to_majmin(chord)
            # Encode as fake midi notes
            if quality_index >= 0:
                pitches = [quality_index + 12]
            else:
                pitches = []
        else:
            pitches = chord_to_midi(chord)
        if quality_only:
            for t in range(start_time_quantized, end_time_quantized):
                for pitch in pitches:
                    chord_track.notes.append(pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=t / (subbeat_div * 2),
                        end=(t + 1) / (subbeat_div * 2),
                    ))
        else:
            # separate chords at downbeats
            start_downbeat_id = np.searchsorted(quantized_downbeats, start_time_quantized)
            end_downbeat_id = np.searchsorted(quantized_downbeats, end_time_quantized)
            for downbeat_id in range(start_downbeat_id, end_downbeat_id + 1):
                s = max(start_time_quantized, quantized_downbeats[downbeat_id - 1])
                e = min(end_time_quantized, quantized_downbeats[downbeat_id])
                if s >= e:
                    continue
                for pitch in pitches:
                    chord_track.notes.append(pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=s / (subbeat_div * 2),
                        end=e / (subbeat_div * 2),
                    ))
    if shift != 0:
        for ins in score_midi.instruments:
            for note in ins.notes:
                note.start += shift / 2
                note.end += shift / 2
    for ins in score_midi.instruments:
        # Clear all pitch bends
        ins.pitch_bends.clear()
    score_midi.instruments.insert(0, chord_track)
    score_midi.write(output_path)

def create_rwc_chord_dataset():
    chord_lab_path = os.path.join(RWC_DATASET_PATH, 'MidiAlignedChord')
    for file in os.listdir(chord_lab_path):
        if file.endswith('.TXT'):
            midi_path = os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file[:-15] + '.SMF_SYNC.MID')
            add_chord_track(midi_path, os.path.join(chord_lab_path, file), os.path.join('temp', 'rwc_chord', file[:-15] + '.mid'), subbeat_div=4)
            add_chord_track(midi_path, os.path.join(chord_lab_path, file), os.path.join('temp', 'rwc_chord_majmin', file[:-15] + '.mid'), quality_only=True, subbeat_div=4)

def create_rwc_half_speed_dataset():
    for file in os.listdir(os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC')):
        if file.endswith('.SMF_SYNC.MID'):
            create_half_speed_midi(os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file), os.path.join('temp', 'rwc_half_speed', file[:7] + '.half.mid'))


def create_la_chord_dataset():
    chord_lab_path = os.path.join('temp', 'la_labels', 'v2', 'decoded_chords')
    file_list = []
    output_path = os.path.join('temp', 'la_chord')
    os.makedirs(output_path, exist_ok=True)
    for file in os.listdir(chord_lab_path):
        if file.endswith('.lab'):
            midi_path = os.path.join(LA_DATASET_PATH, 'MIDIs', file[0], file[:-4] + '.mid')
            output_file_path = os.path.join(output_path, os.path.basename(midi_path))
            file_list.append((midi_path, os.path.join(chord_lab_path, file), output_file_path))
    print(f'{len(file_list)} midi files to process')
    Parallel(n_jobs=-1)(delayed(add_chord_track)(midi_path, chord_lab_path, output_file_path, subbeat_div=4) for midi_path, chord_lab_path, output_file_path in tqdm(file_list))

if __name__ == '__main__':
    create_la_chord_dataset()
    exit(0)
    test_songs = [
        ['394045b83f247bb862d7b09b1aacd78f.mid', 0],
        ['4261342f0970488e1381cb39867c48e1.mid', 0],
        ['f947e58c78aa7c8055ef8dfc424ca22e.mid', 0],
        ['d9520bbf2bccd6424aa09f5694aa68f7.mid', 4.0],
        ['cf5f3bc804e474f4d0baf0c74656b042.mid', 1.0],
    ]
    for test_song, shift in test_songs:
        file_path = os.path.join(LA_DATASET_PATH, 'MIDIs', test_song[0], test_song)
        output_path = os.path.join('temp', 'la_beat', test_song.replace('.mid', '_beat.mid'))
        add_chord_track(file_path, None, output_path, shift=shift)