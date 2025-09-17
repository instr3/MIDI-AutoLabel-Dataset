KEY_MAP = [
    ['C:major', 'Db:major', 'D:major', 'Eb:major', 'E:major', 'F:major', 'F#:major', 'G:major', 'Ab:major', 'A:major', 'Bb:major', 'B:major'],
    ['D:dorian', 'Eb:dorian', 'E:dorian', 'F:dorian', 'F#:dorian', 'G:dorian', 'G#:dorian', 'A:dorian', 'Bb:dorian', 'B:dorian', 'C:dorian', 'C#:dorian'],
    ['E:phrygian', 'F:phrygian', 'F#:phrygian', 'G:phrygian', 'G#:phrygian', 'A:phrygian', 'A#:phrygian', 'B:phrygian', 'C:phrygian', 'C#:phrygian', 'D:phrygian', 'D#:phrygian'],
    ['F:lydian', 'Gb:lydian', 'G:lydian', 'Ab:lydian', 'A:lydian', 'Bb:lydian', 'B:lydian', 'C:lydian', 'Db:lydian', 'D:lydian', 'Eb:lydian', 'E:lydian'],
    ['G:mixolydian', 'Ab:mixolydian', 'A:mixolydian', 'Bb:mixolydian', 'B:mixolydian', 'C:mixolydian', 'C#:mixolydian', 'D:mixolydian', 'Eb:mixolydian', 'E:mixolydian', 'F:mixolydian', 'F#:mixolydian'],
    ['A:minor', 'Bb:minor', 'B:minor', 'C:minor', 'C#:minor', 'D:minor', 'D#:minor', 'E:minor', 'F:minor', 'F#:minor', 'G:minor', 'G#:minor'],
    ['B:locrian', 'C:locrian', 'C#:locrian', 'D:locrian', 'D#:locrian', 'E:locrian', 'E#:locrian', 'F#:locrian', 'G:locrian', 'G#:locrian', 'A:locrian', 'A#:locrian']
]

MODE_NAMES = ['major', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'minor', 'locrian']
MODE_STARTS = [0, 2, 4, 5, 7, 9, 11]

MODE_LOOKUP = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}

NUM_TO_SCALE_SIMPLE = ['C', 'Cb', 'D', 'Eb', 'E', 'F', 'Fb', 'G', 'Ab', 'A', 'Bb', 'B']

if __name__ == '__main__':
    for j in range(-5, 7):
        for i in range(7):
            print(KEY_MAP[i][j * 5 % 12], end=' ')
        print()
