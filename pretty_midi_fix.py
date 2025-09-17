import pretty_midi
import mido
import six
import warnings
import numpy as np
import collections
from typing import Union
import copy
# from ctypes import c_int32
from pretty_midi.utilities import (key_name_to_key_number, qpm_to_bpm)
from pretty_midi import TimeSignature
from pretty_midi import KeySignature, Lyric, Text
from heapq import merge

def get_time_mapping(performance_midi: pretty_midi.PrettyMIDI, score_midi: pretty_midi.PrettyMIDI):
    perf_to_score_mapping = [(0, 0)]  # Initialize with the start time
    for ins_p, ins_s in zip(performance_midi.instruments, score_midi.instruments):
        for note_p, note_s in zip(ins_p.notes, ins_s.notes):
            perf_to_score_mapping.append((note_p.start, note_s.start))
            perf_to_score_mapping.append((note_p.end, note_s.end))
        for pitch_bend_p, pitch_bend_s in zip(ins_p.pitch_bends, ins_s.pitch_bends):
            perf_to_score_mapping.append((pitch_bend_p.time, pitch_bend_s.time))
    perf_to_score_mapping.sort(key=lambda x: x[0])
    perf_to_score_mapping = np.array(perf_to_score_mapping, dtype=np.float32)

    def perf_time_to_score(perf_time):
        return np.interp(perf_time, perf_to_score_mapping[:, 0], perf_to_score_mapping[:, 1])
    return perf_time_to_score



class UglyMIDI(pretty_midi.PrettyMIDI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def __init__(self, midi_file: str, constant_tempo: Union[None, float] = None, verbose: bool = False, fix_track0: bool = True):
        """Initialize either by populating it with MIDI data from a file or
        from scratch with no data.

        """
        # Load in the MIDI data using the midi module
        if isinstance(midi_file, six.string_types):
            # If a string was given, pass it as the string filename
            midi_data = mido.MidiFile(filename=midi_file, clip=True)
        else:
            # Otherwise, try passing it in as a file pointer
            midi_data = mido.MidiFile(file=midi_file, clip=True)

        # Convert tick values in midi_data to absolute, a useful thing.
        for track in midi_data.tracks:
            tick = 0
            # has_negative = False
            for event in track:
                if event.time > 0x7FFFFFFF:
                    # Nothing is written on the standard MIDI file format! How ugly. How should we do this?
                    # has_negative = True
                    event.time = 0 # c_int32(event.time).value
                event.time += tick
                tick = event.time
            # if has_negative:
            #     track.sort(key=lambda e: e.time)

        # Move all tempo, key, time signature events to track 0
        if fix_track0:
            for track_id in range(1, len(midi_data.tracks)):
                track = midi_data.tracks[track_id]
                for event in track:
                    if event.type in ('set_tempo', 'key_signature', 'time_signature'):
                        midi_data.tracks[0].append(event)
                # Clear all moved events at once, to make it faster
                midi_data.tracks[track_id] = [event for event in track if event.type not in ('set_tempo', 'key_signature', 'time_signature')]
            
            # Sort track 0 again. list.sort should be stable, which is important.
            midi_data.tracks[0].sort(key=lambda e: e.time)

        # Store the resolution for later use
        self.resolution = midi_data.ticks_per_beat

        # Populate the list of tempo changes (tick scales)
        if constant_tempo is not None:
            self._tick_scales = [(0, 60.0/(float(constant_tempo)*self.resolution))]
        else:
            self._load_tempo_changes(midi_data)

        # Update the array which maps ticks to time
        max_tick = max([max([e.time for e in t])
                        for t in midi_data.tracks]) + 1
        # If max_tick is huge, the MIDI file is probably corrupt
        # and creating the __tick_to_time array will thrash memory
        if max_tick > pretty_midi.MAX_TICK:
            raise ValueError(('MIDI file has a largest tick of {},'
                              ' it is likely corrupt'.format(max_tick)))

        # Create list that maps ticks to time in seconds
        self._update_tick_to_time(max_tick)

        # Populate the list of key and time signature changes
        self._load_metadata(midi_data)

        # Populate the list of instruments
        self._load_instruments(midi_data)

    def _load_metadata(self, midi_data):
        """Populates ``self.time_signature_changes`` with ``TimeSignature``
        objects, ``self.key_signature_changes`` with ``KeySignature`` objects,
        ``self.lyrics`` with ``Lyric`` objects and ``self.text_events`` with
        ``Text`` objects.

        Parameters
        ----------
        midi_data : midi.FileReader
            MIDI object from which data will be read.
        """

        # Initialize empty lists for storing key signature changes, time
        # signature changes, and lyrics
        self.key_signature_changes = []
        self.time_signature_changes = []
        self.lyrics = []
        self.text_events = []
        self.markers = []

        for event in midi_data.tracks[0]:
            if event.type == 'key_signature':
                key_obj = KeySignature(
                    key_name_to_key_number(event.key),
                    self._PrettyMIDI__tick_to_time[event.time])
                self.key_signature_changes.append(key_obj)

            elif event.type == 'time_signature':
                ts_obj = TimeSignature(event.numerator,
                                       event.denominator,
                                       self._PrettyMIDI__tick_to_time[event.time])
                self.time_signature_changes.append(ts_obj)

        # We search for lyrics and text events on all tracks
        # Lists of lyrics and text events lists, for every track
        tracks_with_lyrics = []
        tracks_with_text_events = []
        for track in midi_data.tracks:
            # Track specific lists that get appended if not empty
            lyrics = []
            text_events = []
            for event in track:
                if event.type == 'lyrics':
                    lyrics.append(Lyric(
                        event.text, self._PrettyMIDI__tick_to_time[event.time]))
                elif event.type == 'text':
                    text_events.append(Text(
                        event.text, self._PrettyMIDI__tick_to_time[event.time]))
                elif event.type == 'marker':
                    self.markers.append(Text(
                        event.text, self._PrettyMIDI__tick_to_time[event.time]))

            if lyrics:
                tracks_with_lyrics.append(lyrics)
            if text_events:
                tracks_with_text_events.append(text_events)

        # We merge the already sorted lists for every track, based on time
        self.lyrics = list(merge(*tracks_with_lyrics, key=lambda x: x.time))
        self.text_events = list(merge(*tracks_with_text_events, key=lambda x: x.time))

    def get_beats_extra(self, start_time=0.):
        """Returns a list of beat locations, according to MIDI tempo changes.
        For compound meters (any whose numerator is a multiple of 3 greater
        than 3), this method returns every third denominator note (for 6/8
        or 6/16 time, for example, it will return every third 8th note or
        16th note, respectively). For all other meters, this method returns
        every denominator note (every quarter note for 3/4 or 4/4 time, for
        example).

        Parameters
        ----------
        start_time : float
            Location of the first beat, in seconds.

        Returns
        -------
        beats : np.ndarray
            Beat locations, in seconds.

        """
        # Get tempo changes and tempos
        tempo_change_times, tempi = self.get_tempo_changes()
        # Create beat list; first beat is at first onset
        beats = [[start_time, -1, -1, -1]]
        # Index of the tempo we're using
        tempo_idx = 0
        # Move past all the tempo changes up to the supplied start time
        while (tempo_idx < tempo_change_times.shape[0] - 1 and
                beats[-1][0] > tempo_change_times[tempo_idx + 1]):
            tempo_idx += 1
        # Logic requires that time signature changes are sorted by time
        self.time_signature_changes.sort(key=lambda ts: ts.time)
        # Index of the time signature change we're using
        ts_idx = 0
        # Move past all time signature changes up to the supplied start time
        while (ts_idx < len(self.time_signature_changes) - 1 and
                beats[-1][0] >= self.time_signature_changes[ts_idx + 1].time):
            ts_idx += 1

        def get_current_bpm():
            ''' Convenience function which computs the current BPM based on the
            current tempo change and time signature events '''
            # When there are time signature changes, use them to compute BPM
            if self.time_signature_changes:
                return qpm_to_bpm(
                    tempi[tempo_idx],
                    self.time_signature_changes[ts_idx].numerator,
                    self.time_signature_changes[ts_idx].denominator)
            # Otherwise, just use the raw tempo change event tempo
            else:
                return tempi[tempo_idx]

        def gt_or_close(a, b):
            ''' Returns True if a > b or a is close to b '''
            return a > b or np.isclose(a, b)

        # Get track end time
        end_time = self.get_end_time()
        # Add beats in
        while beats[-1][0] < end_time:
            # Update the current bpm
            bpm = get_current_bpm()
            # Compute expected beat location, one period later
            next_beat = beats[-1][0] + 60.0/bpm
            # If the beat location passes a tempo change boundary...
            if (tempo_idx < tempo_change_times.shape[0] - 1 and
                    next_beat > tempo_change_times[tempo_idx + 1]):
                # Start by setting the beat location to the current beat...
                next_beat = beats[-1][0]
                # with the entire beat remaining
                beat_remaining = 1.0
                # While a beat with the current tempo would pass a tempo
                # change boundary...
                while (tempo_idx < tempo_change_times.shape[0] - 1 and
                        next_beat + beat_remaining*60.0/bpm >=
                        tempo_change_times[tempo_idx + 1]):
                    # Compute the amount the beat location overshoots
                    overshot_ratio = (tempo_change_times[tempo_idx + 1] -
                                      next_beat)/(60.0/bpm)
                    # Add in the amount of the beat during this tempo
                    next_beat += overshot_ratio*60.0/bpm
                    # Less of the beat remains now
                    beat_remaining -= overshot_ratio
                    # Increment the tempo index
                    tempo_idx = tempo_idx + 1
                    # Update the current bpm
                    bpm = get_current_bpm()
                # Add in the remainder of the beat at the current tempo
                next_beat += beat_remaining*60./bpm
            # Check if we have just passed the first time signature change
            if self.time_signature_changes and ts_idx == 0:
                current_ts_time = self.time_signature_changes[ts_idx].time
                if (current_ts_time > beats[-1][0] and
                        gt_or_close(next_beat, current_ts_time)):
                    # Set the next beat to the time signature change time
                    next_beat = current_ts_time
            # If the next beat location passes the next time signature change
            # boundary
            if ts_idx < len(self.time_signature_changes) - 1:
                # Time of the next time signature change
                next_ts_time = self.time_signature_changes[ts_idx + 1].time
                if gt_or_close(next_beat, next_ts_time):
                    # Set the next beat to the time signature change time
                    next_beat = next_ts_time
                    # Update the time signature index
                    ts_idx += 1
                    # Update the current bpm
                    bpm = get_current_bpm()
            if self.time_signature_changes:
                beats.append([next_beat, get_current_bpm(), self.time_signature_changes[ts_idx].numerator, self.time_signature_changes[ts_idx].denominator])
            else:
                beats.append([next_beat, bpm, 4, 4])  # default values. Ugly. Be careful!

        # The last beat will pass the end_time barrier, so don't include it
        beats = np.array(beats[:-1])
        return beats

    def get_downbeats_extra(self, start_time=0.):
        """Return a list of downbeat locations, according to MIDI tempo changes
        and time signature change events.

        Parameters
        ----------
        start_time : float
            Location of the first downbeat, in seconds.

        Returns
        -------
        downbeats : np.ndarray
            Downbeat locations, in seconds.

        """
        # Get beat locations
        beats_extra = self.get_beats_extra(start_time)
        # Make a copy of time signatures as we will be manipulating it
        time_signatures = copy.deepcopy(self.time_signature_changes)

        # If there are no time signatures or they start after 0s, add a 4/4
        # signature at time 0
        if not time_signatures or time_signatures[0].time > start_time:
            time_signatures.insert(0, TimeSignature(4, 4, start_time))

        def index(array, value, default):
            """ Returns the first index of a value in an array, or `default` if
            the value doesn't appear in the array."""
            idx = np.flatnonzero(np.isclose(array, value))
            if idx.size > 0:
                return idx[0]
            else:
                return default

        downbeats = []
        end_beat_idx = 0
        # Iterate over spans of time signatures
        for start_ts, end_ts in zip(time_signatures[:-1], time_signatures[1:]):
            # Get index of first beat at start_ts.time, or else use first beat
            start_beat_idx = index(beats_extra[:, 0], start_ts.time, 0)
            # Get index of first beat at end_ts.time, or else use last beat
            end_beat_idx = index(beats_extra[:, 0], end_ts.time, start_beat_idx)

            # Add beats within this time signature range, skipping beats
            # according to the current time signature
            if start_ts.numerator % 3 == 0 and start_ts.numerator != 3:
                downbeats.extend(beats_extra[
                    start_beat_idx:end_beat_idx:(start_ts.numerator // 3)])
            else:
                downbeats.extend(beats_extra[
                    start_beat_idx:end_beat_idx:start_ts.numerator])
        # Add in beats from the second-to-last to last time signature
        final_ts = time_signatures[-1]
        start_beat_idx = index(beats_extra[:, 0], final_ts.time, end_beat_idx)

        if final_ts.numerator % 3 == 0 and final_ts.numerator != 3:
            downbeats.extend(beats_extra[start_beat_idx::(final_ts.numerator // 3)])
        else:
            downbeats.extend(beats_extra[start_beat_idx::final_ts.numerator])
        # Convert from list to array
        downbeats = np.stack(downbeats, axis=0)
        # Return all downbeats after start_time
        return downbeats[downbeats[:, 0] >= start_time]

    def _load_instruments(self, midi_data):
        """Populates ``self.instruments`` using ``midi_data``.

        Parameters
        ----------
        midi_data : midi.FileReader
            MIDI object from which data will be read.
        """
        # MIDI files can contain a collection of tracks; each track can have
        # events occuring on one of sixteen channels, and events can correspond
        # to different instruments according to the most recently occurring
        # program number.  So, we need a way to keep track of which instrument
        # is playing on each track on each channel.  This dict will map from
        # program number, drum/not drum, channel, and track index to instrument
        # indices, which we will retrieve/populate using the __get_instrument
        # function below.
        instrument_map = collections.OrderedDict()
        # Store a similar mapping to instruments storing "straggler events",
        # e.g. events which appear before we want to initialize an Instrument
        stragglers = {}
        # This dict will map track indices to any track names encountered
        track_name_map = collections.defaultdict(str)

        def __get_instrument(program, channel, track, create_new):
            """Gets the Instrument corresponding to the given program number,
            drum/non-drum type, channel, and track index.  If no such
            instrument exists, one is created.

            """
            # If we have already created an instrument for this program
            # number/track/channel, return it
            if (program, channel, track) in instrument_map:
                return instrument_map[(program, channel, track)]
            # If there's a straggler instrument for this instrument and we
            # aren't being requested to create a new instrument
            if not create_new and (channel, track) in stragglers:
                return stragglers[(channel, track)]
            # If we are told to, create a new instrument and store it
            if create_new:
                is_drum = (channel == 9)
                instrument = pretty_midi.Instrument(
                    program, is_drum, track_name_map[track_idx])
                instrument.channel = channel
                # If any events appeared for this instrument before now,
                # include them in the new instrument
                if (channel, track) in stragglers:
                    straggler = stragglers[(channel, track)]
                    instrument.control_changes = straggler.control_changes
                    instrument.pitch_bends = straggler.pitch_bends
                # Add the instrument to the instrument map
                instrument_map[(program, channel, track)] = instrument
            # Otherwise, create a "straggler" instrument which holds events
            # which appear before we actually want to create a proper new
            # instrument
            else:
                # Create a "straggler" instrument
                instrument = pretty_midi.Instrument(program, track_name_map[track_idx])
                # Note that stragglers ignores program number, because we want
                # to store all events on a track which appear before the first
                # note-on, regardless of program
                stragglers[(channel, track)] = instrument
            return instrument

        for track_idx, track in enumerate(midi_data.tracks):
            # Keep track of last note on location:
            # key = (instrument, note),
            # value = (note-on tick, velocity)
            last_note_on = collections.defaultdict(list)
            # Keep track of which instrument is playing in each channel
            # initialize to program 0 for all channels
            current_instrument = np.zeros(16, dtype=np.int32)
            for event in track:
                # Look for track name events
                if event.type == 'track_name':
                    # Set the track name for the current track
                    track_name_map[track_idx] = event.name
                # Look for program change events
                if event.type == 'program_change':
                    # Update the instrument for this channel
                    current_instrument[event.channel] = event.program
                # Note ons are note on events with velocity > 0
                elif event.type == 'note_on' and event.velocity > 0:
                    # Store this as the last note-on location
                    note_on_index = (event.channel, event.note)
                    last_note_on[note_on_index].append((
                        event.time, event.velocity))
                # Note offs can also be note on events with 0 velocity
                elif event.type == 'note_off' or (event.type == 'note_on' and
                                                  event.velocity == 0):
                    # Check that a note-on exists (ignore spurious note-offs)
                    key = (event.channel, event.note)
                    if key in last_note_on:
                        # Get the start/stop times and velocity of every note
                        # which was turned on with this instrument/drum/pitch.
                        # One note-off may close multiple note-on events from
                        # previous ticks. In case there's a note-off and then
                        # note-on at the same tick we keep the open note from
                        # this tick.
                        end_tick = event.time
                        open_notes = last_note_on[key]

                        notes_to_close = [
                            (start_tick, velocity)
                            for start_tick, velocity in open_notes
                            if start_tick != end_tick]
                        notes_to_keep = [
                            (start_tick, velocity)
                            for start_tick, velocity in open_notes
                            if start_tick == end_tick]

                        for start_tick, velocity in notes_to_close:
                            start_time = self._PrettyMIDI__tick_to_time[start_tick]
                            end_time = self._PrettyMIDI__tick_to_time[end_tick]
                            # Create the note event
                            note = pretty_midi.Note(velocity, event.note, start_time,
                                        end_time)
                            # Get the program and drum type for the current
                            # instrument
                            program = current_instrument[event.channel]
                            # Retrieve the Instrument instance for the current
                            # instrument
                            # Create a new instrument if none exists
                            instrument = __get_instrument(
                                program, event.channel, track_idx, 1)
                            # Add the note event
                            instrument.notes.append(note)

                        if len(notes_to_close) > 0 and len(notes_to_keep) > 0:
                            # Note-on on the same tick but we already closed
                            # some previous notes -> it will continue, keep it.
                            last_note_on[key] = notes_to_keep
                        else:
                            # Remove the last note on for this instrument
                            del last_note_on[key]
                # Store pitch bends
                elif event.type == 'pitchwheel':
                    # Create pitch bend class instance
                    bend = pretty_midi.PitchBend(event.pitch,
                                     self._PrettyMIDI__tick_to_time[event.time])
                    # Get the program for the current inst
                    program = current_instrument[event.channel]
                    # Retrieve the Instrument instance for the current inst
                    # Don't create a new instrument if none exists
                    instrument = __get_instrument(
                        program, event.channel, track_idx, 0)
                    # Add the pitch bend event
                    instrument.pitch_bends.append(bend)
                # Store control changes
                elif event.type == 'control_change':
                    control_change = pretty_midi.ControlChange(
                        event.control, event.value,
                        self._PrettyMIDI__tick_to_time[event.time])
                    # Get the program for the current inst
                    program = current_instrument[event.channel]
                    # Retrieve the Instrument instance for the current inst
                    # Don't create a new instrument if none exists
                    instrument = __get_instrument(
                        program, event.channel, track_idx, 0)
                    # Add the control change event
                    instrument.control_changes.append(control_change)
        # Initialize list of instruments from instrument_map
        self.instruments = [i for i in instrument_map.values()]


def __sanity_sort_stability_check():
    # Original data
    data = [(1, 'a'), (2, 'b'), (1, 'c'), (3, 'd'), (2, 'e'), (2, 'f'), (4, 'g'), (1, 'h'), (0, 'i'), (1, 'j'), (2, 'l'), (4, 'k'), (2, 'm')]

    # Create a copy of the data to sort
    data_to_sort = data.copy()

    # Sort the copied list based on the first element of each tuple
    data_to_sort.sort(key=lambda x: x[0])

    # Print the sorted list
    print('Sorted list:', data_to_sort)

    # Check if the relative order of elements with equal keys is preserved
    original_order = {key: [] for key, _ in data}
    for key, value in data:
        original_order[key].append(value)

    sorted_order = {key: [] for key, _ in data_to_sort}
    for key, value in data_to_sort:
        sorted_order[key].append(value)

    is_stable = all(sorted_order[key] == original_order[key] for key in original_order)
    print('Is the sort stable?', is_stable)

if __name__ == '__main__':
    __sanity_sort_stability_check()