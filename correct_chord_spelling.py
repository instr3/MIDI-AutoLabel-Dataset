from baselines.exported_midi_chord_recognition.chord_class import QUALITIES
import numpy as np
np.int = int  # For compatibility with older versions of numpy in mir_eval
import mir_eval.chord
from key_names import KEY_MAP, MODE_NAMES, MODE_STARTS
DEFAULT_SPELLING = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', '#5', '6', 'b7', '7']
KEY_SPELLINGS = np.array([
    [0, 0],  # C
    [1, -1],  # Db
    [1, 0],  # D
    [2, -1],  # Eb
    [2, 0],  # E
    [3, 0],  # F
    [-1, -1],  # Does not care
    [4, 0],  # G
    [5, -1],  # Ab
    [5, 0],  # A
    [6, -1],  # Bb
    [6, 0],  # B
], dtype=int)

CHORD_SPELLING_TABLE = None
CIRCLE_OF_FIFTH = np.array([3, 0, 5, 1, 6, 2, 7])
CIRCLE_OF_FIFTH_INV = np.array([1, 3, 5, 0, 2, 4, 6])
NOTE_NAMES = 'CDEFGAB'

def scale_degree_to_tuple(scale_degree):
    offset = 0
    if scale_degree.startswith("#"):
        offset = scale_degree.count("#")
        scale_degree = scale_degree.strip("#")
    elif scale_degree.startswith("b"):
        offset = -1 * scale_degree.count("b")
        scale_degree = scale_degree.strip("b")
    return np.array([int(scale_degree) - 1, offset], dtype=int)

def note_name_to_tuple(note_name):
    scale_degree = NOTE_NAMES.index(note_name[0])
    offset = 0
    if note_name.endswith("#"):
        offset = note_name.count("#")
    elif note_name.endswith("b"):
        offset = -1 * note_name.count("b")
    return np.array([scale_degree, offset], dtype=int)

def scale_degrees_to_tuple(scale_degrees):
    return np.stack([scale_degree_to_tuple(sd) for sd in scale_degrees], axis=0)


def get_chord_spelling_table():
    global CHORD_SPELLING_TABLE
    if CHORD_SPELLING_TABLE is not None:
        return CHORD_SPELLING_TABLE
    chord_spelling_table = {}
    for quality in QUALITIES:
        quality_spelling = []
        chroma = QUALITIES[quality]
        for i in range(12):
            if chroma[i] > 0:
                if '#9' in quality and i == DEFAULT_SPELLING.index('b3'):
                    quality_spelling.append('#2')
                elif 'dim7' in quality and i == DEFAULT_SPELLING.index('6'):
                    quality_spelling.append('bb7')
                else:
                    quality_spelling.append(DEFAULT_SPELLING[i])
        chord_spelling_table[quality] = scale_degrees_to_tuple(quality_spelling)
    CHORD_SPELLING_TABLE = chord_spelling_table
    return chord_spelling_table

def score_spelling_under_key(spelling, key):
    '''
    Score the chord spelling under a given key.
    :param spelling: Numpy array of shape (N, 2)
    :param key: Numpy array of shape (2,) representing the degree of 1 (DO)
    :return: Score of the chord spelling under the key.
    '''
    key_pos = CIRCLE_OF_FIFTH_INV[key[0]] + key[1] * 7
    spelling_pos = CIRCLE_OF_FIFTH_INV[spelling[:, 0] % 7] + spelling[:, 1] * 7
    relative_pos = np.clip(np.abs((spelling_pos - (key_pos + 2))) - 3, 0, None)
    return np.sum(relative_pos)

def correct_chord_spelling_by_strength(label, key_strength):
    '''
    Correct the chord spelling based on the key strength. Explicitly avoids F#/Gb ambiguity.
    :param label: Chord label in the format 'Root:Quality' or 'Root:Quality/Inversion'
    :param key_strength: Key strength as a string (e.g., 'C', 'Db') or a numpy array of shape (12,) representing the strength of each pitch class.
    :return: Corrected chord label in the format 'Root:Quality/Inversion'.
    '''
    if label == 'N' or label == 'X':
        return label
    if isinstance(key_strength, str):
        key_semitone = mir_eval.chord.pitch_class_to_semitone(key_strength)
        key_strength = np.zeros(12, dtype=float)
        key_strength[key_semitone] = 1.0
    # Ignore all inversions
    inversion = ''
    if '/' in label:
        label, inversion = label.split('/')
    # Separate quality and root
    root, quality = label.split(':')
    for i in range(12):
        pass
    root_semitone = mir_eval.chord.pitch_class_to_semitone(root)
    scores = []
    for possible_root_id in range(7):
        possible_root = NOTE_NAMES[possible_root_id]
        root_spelling = np.array([
            possible_root_id,
            (root_semitone - mir_eval.chord.pitch_class_to_semitone(possible_root) + 6) % 12 - 6
        ])
        chord_spelling = get_chord_spelling_table()[quality] + root_spelling[None, :]
        result = 0
        # Score the spelling under each possible key (except for F#/Gb, which is ambiguous given chroma)
        for i, key_spelling in enumerate(KEY_SPELLINGS):
            if key_spelling[0] == -1:
                continue
            else:
                result += key_strength[i] * score_spelling_under_key(chord_spelling, key_spelling)
        scores.append(result)
    best_root_id = np.argmin(scores)
    best_root = NOTE_NAMES[best_root_id]
    best_root_spelling = np.array([
        best_root_id,
        (root_semitone - mir_eval.chord.pitch_class_to_semitone(best_root) + 6) % 12 - 6
    ])
    if best_root_spelling[1] < 0:
        best_root = best_root + 'b' * abs(best_root_spelling[1])
    else:
        best_root = best_root + '#' * best_root_spelling[1]
    if inversion:
        return f"{best_root}:{quality}/{inversion}"
    else:
        return f"{best_root}:{quality}"

def correct_chord_spelling(label, key_name):
    if label == 'N' or label == 'X':
        return label
    key_tonal, key_mode = key_name.split(':')
    key_mode = MODE_NAMES.index(key_mode)
    scale_semitone = (mir_eval.chord.pitch_class_to_semitone(key_tonal) - MODE_STARTS[key_mode]) % 12
    assert KEY_MAP[key_mode][scale_semitone] == key_name, f'Unsupported key name: {key_name}'
    key_spelling = note_name_to_tuple(KEY_MAP[0][scale_semitone].split(':')[0])
    # Ignore all inversions
    inversion = ''
    if '/' in label:
        label, inversion = label.split('/')
    # Separate quality and root
    root, quality = label.split(':')
    for i in range(12):
        pass
    root_semitone = mir_eval.chord.pitch_class_to_semitone(root)
    scores = []
    for possible_root_id in range(7):
        possible_root = NOTE_NAMES[possible_root_id]
        root_spelling = np.array([
            possible_root_id,
            (root_semitone - mir_eval.chord.pitch_class_to_semitone(possible_root) + 6) % 12 - 6
        ])
        chord_spelling = get_chord_spelling_table()[quality] + root_spelling[None, :]
        scores.append(score_spelling_under_key(chord_spelling, key_spelling))
    best_root_id = np.argmin(scores)
    best_root = NOTE_NAMES[best_root_id]
    best_root_spelling = np.array([
        best_root_id,
        (root_semitone - mir_eval.chord.pitch_class_to_semitone(best_root) + 6) % 12 - 6
    ])
    if best_root_spelling[1] < 0:
        best_root = best_root + 'b' * abs(best_root_spelling[1])
    else:
        best_root = best_root + '#' * best_root_spelling[1]
    if inversion:
        return f"{best_root}:{quality}/{inversion}"
    else:
        return f"{best_root}:{quality}"

if __name__ == '__main__':
    print(correct_chord_spelling('F#:7', 'Db:major'))
    print(correct_chord_spelling('F:7', 'B:major'))
    print(correct_chord_spelling('Gb:hdim7', 'A:minor'))
    print(correct_chord_spelling('F#:aug', 'B:locrian'))
    print(correct_chord_spelling('F#:aug', 'B:major'))