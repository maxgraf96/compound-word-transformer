import numpy as np
import os
import pickle
import sys
import tempfile
import time
from enum import Enum
from multiprocessing import Process, Value
from pathlib import Path
from threading import Thread

import mido
import onnx
import torch
from audiolazy import lazy_midi  # For converting midi strings to midi numbers
from mido import MidiFile, MidiTrack, Message, MetaMessage

import main_cp
import onnx_converter
from dataset.representations.uncond.cp.constants import BEAT_RESOLUTION, ABLETON_MIDI_TICK_RESOLUTION

ppqn = ABLETON_MIDI_TICK_RESOLUTION
ppbar = ppqn * 4  # in 4/4 metre
last_event_delta = 0
last_event_tick = 0
generate_every_n_bars = 0


def tick_to_beat(ableton_tick):
    ret = int(ableton_tick * (BEAT_RESOLUTION / ABLETON_MIDI_TICK_RESOLUTION))
    return ret


shared_bool = False


def write_midi(notes):
    filename = 'mido_created.mid'

    mid = MidiFile()
    mid.ticks_per_beat = BEAT_RESOLUTION
    track = MidiTrack()
    mid.tracks.append(track)

    # Set instrument to acoustic grand piano
    # See https://www.noterepeat.com/articles/how-to/213-midi-basics-common-terms-explained
    # for all default MIDI instruments
    instrument = 0
    track.append(Message('program_change', program=instrument, time=0))
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
    for note_msg in notes:
        track.append(note_msg)

    mid.save(filename)
    return filename


def sanity_check(notes):
    if len(notes) == 0:
        notes.clear()
        return notes

    # 1) All note ons have note offs
    ons = sum(1 for note in notes if note.type == 'note_on')
    offs = sum(1 for note in notes if note.type == 'note_off')
    clear = ons == offs
    if not clear:
        print("Found some notes without note offs, setting note off to bar end.")
        # Create note offs a quarter note later
        # Get rogue last few note ons (without note off events)
        rogue_notes = []
        i = -1
        last_note = notes[i]
        while last_note is not None and last_note.type == 'note_on':
            rogue_notes.append(last_note)
            i -= 1
            try:
                # Take previous note
                last_note = notes[i]
            except IndexError:
                # All notes are note ons
                last_note = None
        # Create note off events for rogue notes
        for i in range(len(rogue_notes)):
            note = rogue_notes[i]
            if i == 0:
                new_note = Message('note_off', note=note.note, velocity=note.velocity,
                                   time=tick_to_beat(generate_every_n_bars * 4 * ABLETON_MIDI_TICK_RESOLUTION - last_event_tick))
            else:
                new_note = Message('note_off', note=note.note, velocity=note.velocity,
                                   time=0)
            notes.append(new_note)

        # Check if all notes are note offs
        all_note_offs = True
        for note in notes:
            if note.type != 'note_off':
                all_note_offs = False
                break
        if all_note_offs:
            notes.clear()

    return notes


class State(Enum):
    PAUSED = 0
    PLAYING = 1


def midi_loop(data, name=''):
    global last_event_delta, last_event_tick, generate_every_n_bars
    print(mido.get_input_names())
    port = mido.open_input('loopMIDI Port 2')

    notes = []
    bar_counter = 0
    generate_every_n_bars = 2
    # Current tick
    tick = 1
    last_event_tick = 0
    last_event_delta = 0
    state = State.PAUSED

    def clear():
        notes.clear()
        tick = 1
        last_event_tick = 0
        bar_counter = 0
        last_event_delta = 0
        return tick, last_event_tick, bar_counter, last_event_delta

    while True:
        if state == State.PAUSED:
            time.sleep(0.1)
        for msg in port.iter_pending():
            # print(msg)
            # Get message type
            if tick > 1 and msg.type == 'note_on':
                # Get time difference in ticks to last event
                last_event_delta = tick_to_beat(tick - last_event_tick)
                msg.velocity = 80
                # Create note_on for note
                note_on = Message('note_on', note=msg.note, velocity=msg.velocity, time=last_event_delta)
                # Append note on
                notes.append(note_on)
                last_event_tick = tick
            if tick > 1 and msg.type == 'note_off':
                # Discard hanging/dead notes
                if last_event_delta == 0:
                    continue
                # Get time difference in ticks to last event
                last_event_delta = tick_to_beat(tick - last_event_tick)
                msg.velocity = 80
                note_off = Message('note_off', note=msg.note, velocity=msg.velocity, time=last_event_delta)
                notes.append(note_off)
                last_event_tick = tick

            # MIDI clock runs at 24ppqn (pulses per quarter note)
            # => 96 pulses =^= 1 bar
            if msg.type == 'clock':
                tick += 1
                if tick % ppbar == 0:
                    bar_counter += 1
                    print(bar_counter)
                if bar_counter > 0 and tick % (generate_every_n_bars * ppbar) == 0:
                    # Sanity checks
                    notes = sanity_check(notes)

                    if len(notes) > 0:
                        print("Writing MIDI for ", generate_every_n_bars, " bars.")

                        print("Notes: ")
                        print(notes)

                        # Create MIDI file
                        write_midi(notes)
                        data.value = True

                    # Clear the note list and reset tick
                    tick, last_event_tick, bar_counter, last_event_delta = clear()

            if msg.type == 'control_change':
                # Stop button
                if msg.control == 123:
                    state = State.PAUSED
                    tick, last_event_tick, bar_counter, last_event_delta = clear()
            else:
                state = State.PLAYING


if __name__ == '__main__':
    shared_bool = Value('b', False)

    # Start MIDI listening loop
    process = Process(target=midi_loop, args=(shared_bool, 'shared_bool'))
    process.start()

    # Load net
    # path
    path_ckpt = main_cp.info_load_model[0]  # path to ckpt dir
    loss = main_cp.info_load_model[1]  # loss
    name = 'loss_' + str(loss)
    path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')

    # load
    dictionary = pickle.load(open(main_cp.path_dictionary, 'rb'))
    event2word, word2event = dictionary

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # init model
    net = main_cp.TransformerModel(n_class, is_training=False)
    net.cuda()
    net.eval()

    # load model
    print('[*] load model from:', path_saved_ckpt)
    net.load_state_dict(torch.load(path_saved_ckpt))

    while True:
        try:
            if shared_bool.value == 1:
                main_cp.generate_bars_from_midi_prompt(net, event2word, word2event, dictionary, 'mido_created.mid', 2)
                shared_bool.value = 0
                print("Done writing MIDI")
            time.sleep(0.01)
        except Exception as e:
            sys.exit()

