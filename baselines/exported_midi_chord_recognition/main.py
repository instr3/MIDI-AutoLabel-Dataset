from mir import DataEntry
from mir import io
from .extractors.midi_utilities import get_valid_channel_count,is_percussive_channel,MidiBeatExtractor
from .extractors.rule_based_channel_reweight import midi_to_thickness_and_bass_weights
from .midi_chord import ChordRecognition
from .chord_class import ChordClass
import numpy as np
from .io_new.chordlab_io import ChordLabIO

def process_chord(entry, extra_division, single_beat_switch=True, verbose=False, return_chroma=False, replace_chroma=None, use_transition=True):
    '''

    Parameters
    ----------
    entry: the song to be processed. Properties required:
        entry.midi: the pretry midi object
        entry.beat: extracted beat and downbeat
    extra_division: extra divisions to each beat.
        For chord recognition on beat-level, use extra_division=1
        For chord recognition on half-beat-level, use extra_division=2

    Returns
    -------
    Extracted chord sequence
    '''

    midi=entry.midi
    beats=midi.get_beats()
    if(extra_division>1):
        beat_interp=np.linspace(beats[:-1],beats[1:],extra_division+1).T
        last_beat=beat_interp[-1,-1]
        beats=np.append(beat_interp[:,:-1].reshape((-1)),last_beat)
    downbeats=midi.get_downbeats()
    j=0
    beat_pos=-2
    beat=[]
    for i in range(len(beats)):
        if(j<len(downbeats) and beats[i]==downbeats[j]):
            beat_pos=1
            j+=1
        else:
            beat_pos=beat_pos+1
        assert(beat_pos>0)
        beat.append([beats[i],beat_pos])
    rec=ChordRecognition(entry,ChordClass(),single_beat_switch=single_beat_switch)
    weights=midi_to_thickness_and_bass_weights(entry.midi)
    channel_names=[ins.name for ins in midi.instruments if not is_percussive_channel(ins)]
    if verbose:
        print('The name and weight of each channel is:')
        for i in range(len(channel_names)):
            print('%d | %.6f | %s'%(i, weights[i], channel_names[i]))
    rec.process_feature(weights)
    if return_chroma:
        return rec.beat_bass, rec.beat_chroma
    if replace_chroma is not None:
        if replace_chroma[0] is not None:
            rec.beat_bass = replace_chroma[0]
        if replace_chroma[1] is not None:
            rec.beat_chroma = replace_chroma[1]
    chord=rec.decode(use_transition=use_transition)
    return chord

def transcribe_cb1000_midi(midi_path,output_path):
    '''
    Perform chord recognition on a midi
    :param midi_path: the path to the midi file
    :param output_path: the path to the output file
    '''
    entry=DataEntry()
    entry.append_file(midi_path,io.MidiIO,'midi')
    entry.append_extractor(MidiBeatExtractor,'beat')
    result=process_chord(entry,extra_division=2)
    entry.append_data(result,ChordLabIO,'pred')
    entry.save('pred',output_path)

def rule_based_chord_recognition(midi_path, return_chroma=False, replace_chroma=None, use_transition=True):
    '''
    Perform chord recognition on a midi
    :param midi_path: the path to the midi file
    :param output_path: the path to the output file
    '''
    entry=DataEntry()
    if not isinstance(midi_path, str):
        entry.append_data(midi_path, io.MidiIO, 'midi')
    else:
        entry.append_file(midi_path, io.MidiIO, 'midi')
    entry.append_extractor(MidiBeatExtractor,'beat')
    result = process_chord(entry, extra_division=2, return_chroma=return_chroma, replace_chroma=replace_chroma, use_transition=use_transition)
    return result

if __name__ == '__main__':
    import sys
    if(len(sys.argv)!=3):
        print('Usage: main.py midi_path output_path')
        exit(0)
    transcribe_cb1000_midi(sys.argv[1],sys.argv[2])
