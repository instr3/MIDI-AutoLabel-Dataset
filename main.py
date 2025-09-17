from bootstrap_key import model_key_probe, RoformerProber, cadence_based_mode_estimation
from bootstrap_cadence import model_cadence_probe, RoformerProberCadence
from cp_transformer_probe_chord import RoformerProberChord
from chord_eval import model_chord_probe
from io_new.chordlab_io import ChordLabIO
from pretty_midi_fix import UglyMIDI, get_time_mapping
import os
import numpy as np
import torch
from settings import RWC_DATASET_PATH
from hmm import hmm_decode
from key_names import NUM_TO_SCALE_SIMPLE
from correct_chord_spelling import correct_chord_spelling


def analyze(midi_path, use_supervised_model, visualize=True):
    sr = 24000
    perf_midi = UglyMIDI(midi_path)
    score_midi = UglyMIDI(midi_path, constant_tempo=60 / 4)
    score_to_performance_mapping = get_time_mapping(score_midi, perf_midi)
    beats = perf_midi.get_beats()
    if use_supervised_model:
        key_model = RoformerProber.load_from_checkpoint('ckpt/midi-supervised-models/cp_transformer_yinyang_probe_encoder_v0.2_batch_20_rwc_key_evidence_16_v1.epoch=00.val_loss=0.16162.ckpt', strict=False)
    else:
        key_model = RoformerProber.load_from_checkpoint('ckpt/midi-unsupervised-models/cp_transformer_yinyang_probe_encoder_v0.2_batch_20_la_key_evidence_16_v2.epoch=00.val_loss=0.43888.ckpt', strict=False)
    cadence_model = RoformerProberCadence.load_from_checkpoint('ckpt/midi-unsupervised-models/cp_transformer_yinyang_probe_cadence_v0.1_batch_80_la_cadence_v4.epoch=00.val_loss=0.23473.ckpt', strict=False)
    key = model_key_probe(key_model, score_midi, n_samples=1, batch_size=4)
    key_seq = hmm_decode(key[None].cpu(), transition_penalty=5.0).squeeze(0)
    prev_key_change = 0
    cadence = []
    key_lab = []
    for i in range(1, len(key_seq)):
        if key_seq[i] != key_seq[i - 1] or i + 1 == len(key_seq):
            seg_cadence = model_cadence_probe(cadence_model, score_midi, n_samples=1, batch_size=4, fixed_start=prev_key_change * 4, fixed_end=i * 4)
            seg_cadence[0] = 1.0
            best_seg_key = cadence_based_mode_estimation(seg_cadence, start_time=0, end_time=(i - prev_key_change) * 4, majmin_only=True if use_supervised_model else False, scale_id=key_seq[prev_key_change])
            key_lab.append([score_to_performance_mapping(prev_key_change * 4), score_to_performance_mapping(i * 4), best_seg_key])
            cadence.append(seg_cadence[:i - prev_key_change])
            prev_key_change = i
    cadence = torch.cat(cadence, dim=0)
    del key_model
    del cadence_model
    if use_supervised_model:
        bass_model = None  # bass is fine-tuned in the chroma model
        chord_model = RoformerProberChord.load_from_checkpoint('ckpt/midi-supervised-models/cp_transformer_yinyang_probe_encoder_chroma_v0.4_batch_8_rwc_chords_16_v8_chroma.epoch=00.val_loss=0.22750.ckpt', strict=False)
    else:
        bass_model = RoformerProberChord.load_from_checkpoint('ckpt/midi-unsupervised-models/cp_transformer_yinyang_probe_encoder_chroma_v0.5_batch_160_la_bass_16_v2.epoch=00.val_loss=0.10736.ckpt', strict=False)
        chord_model = RoformerProberChord.load_from_checkpoint('ckpt/midi-unsupervised-models/cp_transformer_yinyang_probe_encoder_chroma_v0.4_batch_160_la_chords_16_v8_chroma.epoch=00.val_loss=0.15761.ckpt', strict=False)
    chord, raw_chord_lab = model_chord_probe(chord_model, midi_path, n_samples=1, batch_size=4, write_to_file=False, return_raw_output=True, bass_model=bass_model, return_chord_lab=True)
    # Correct chord spelling under the estimated key (e.g., C#:maj -> Db:maj for some keys)

    key_starts = np.array([float(line[0]) for line in key_lab], dtype=np.float32)
    key_labels = [line[2] for line in key_lab]
    def get_key_at_time(perf_time):
        idx = np.searchsorted(key_starts, perf_time) - 1
        if idx < 0:
            return key_labels[0]
        return key_labels[idx]
    chord_lab = []
    for chord_item in raw_chord_lab:
        start_time = chord_item[0]
        end_time = chord_item[1]
        chord_label = correct_chord_spelling(chord_item[2], get_key_at_time((start_time + end_time) / 2))
        chord_lab.append([start_time, end_time, chord_label])
    if visualize:
        chord = chord[:len(beats)]
        key = key[:len(beats)].cpu().numpy()
        cadence = cadence[:len(beats)].cpu().numpy()
        cadence_tags = ['note ' + s for s in NUM_TO_SCALE_SIMPLE] + ['tonal ' + s for s in NUM_TO_SCALE_SIMPLE]
        key_tags = ['scale ' + s for s in NUM_TO_SCALE_SIMPLE]
        chord_tags = ['note ' + s for s in NUM_TO_SCALE_SIMPLE] + ['bass ' + s for s in NUM_TO_SCALE_SIMPLE]
        from render_midi_to_wav import render_midi
        from mir import DataEntry, io
        file_name = os.path.basename(midi_path)
        entry = DataEntry()
        entry.prop.set('sr', sr)
        entry.prop.set('hop_length', 960)
        audio_path = os.path.join('temp', 'midi_visualize', file_name[:-4] + '.wav')
        if not os.path.exists(audio_path):
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            render_midi(midi_path, os.path.dirname(audio_path), sr=24000)
        entry.append_file(audio_path, io.MusicIO, 'music')
        entry.append_data((key_tags, beats, key), io.RegionalSpectrogramIO, 'key')
        entry.append_data((cadence_tags, beats, cadence), io.RegionalSpectrogramIO, 'cadence')
        entry.append_data((chord_tags, beats, chord), io.RegionalSpectrogramIO, 'chord')
        entry.append_data(chord_lab, ChordLabIO, 'chord_lab')
        entry.append_data(key_lab, ChordLabIO, 'key_lab')
        entry.visualize(['key', 'cadence', 'chord', 'chord_lab', 'key_lab'])
    else:
        chord_text = ', '.join([f'{chord_lab[i][0]:.3f}-{chord_lab[i][1]:.3f} {chord_lab[i][2]}' for i in range(len(chord_lab))])
        key_text = ', '.join([f'{key_lab[i][0]:.3f}-{key_lab[i][1]:.3f} {key_lab[i][2]}' for i in range(len(key_lab))])
        print('Chord labels:', chord_text)
        print('Key segments:', key_text)

if __name__ == '__main__':
    analyze('data/example_midis/000a2a3abdeb6e47294800a45015153d-4-4.mid', use_supervised_model=True, visualize=False)
    analyze('data/example_midis/000c176cd4886471b8298f53b6d903bb.mid', use_supervised_model=True, visualize=False)
    analyze('data/example_midis/032f5bbb8a7ad4dfa25bc925c0c25ea5_fix.mid', use_supervised_model=True, visualize=False)