## source of original loadPieces:               https://github.com/danieldjohnson/biaxial-rnn-music-composition/blob/master/multi_training.py
## source of original midiToNoteStateMatrix:    https://github.com/danieldjohnson/biaxial-rnn-music-composition/blob/master/midi_to_statematrix.py
## source of midi package                       https://github.com/louisabraham/python3-midi

import midi
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


# full range of a piano keyboard sub-contra A to five-lined C
lowerBound = 21 # inclusiv, sub-contra A is  21
upperBound = 109 # exlusiv, five-lined C is 108, the highest note to be included
span = upperBound-lowerBound
num_rest = 0
num_play_hold = 1
num_play_artic = 2

def midiToNoteStateMatrix(midifile, verbose= False, verbose_ts=True, time_signature = 3):

    pattern = midi.read_midifile(midifile)
    timeleft = [track[0].tick for track in pattern]
    posns = [0 for track in pattern]
    state = [[0, 0, 0] for x in range(span)]
    statematrix = []
    statematrix.append(state)
    condition = True
    time = 0

    if verbose_ts:
        print("Resolution", pattern.resolution)
    
    while condition:
        if verbose:
            print("time:", time)
        if (time % (pattern.resolution / 12)) == 0 and time != 0: 
            # == (pattern.resolution / 8):#, if I am not mistaken this introduces a 32th note shift, don't know what this was intended for?
            # (time % (pattern.resolution / 12)) == 0 -> a state has the length of a 48th note
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            if verbose:
                if (time % pattern.resolution) == 0 and time != 0: 
                    beat = (time//pattern.resolution % time_signature)
                    print(beat)
                    print((beat%2,beat//2%2))
            state = [[oldstate[x][0],0, oldstate[x][2]] for x in range(span)]
            statematrix.append(state)
            if verbose:
                print("crossed note boundary, current state:", state)
        for i in range(len(timeleft)): #For each track
            if not condition:
                break
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if verbose:
                    print(evt)
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        if verbose_ts:
                            print("Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time))
                        pass
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerBound] = [0, 0, 0]
                            if verbose:                                
                                print("velocity:", evt.velocity)
                                print("current_state:", state)
                        else:
                            state[evt.pitch-lowerBound] = [1, 1, evt.velocity]
                            if verbose:
                                print("velocity:", evt.velocity)
                                print("pitch:", evt.pitch)
                                print("current_state:", state)
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator in (5,7,11,13):
                        if verbose_ts:
                            print("Bailing - strange time signature: " +  str(evt.numerator) + " at time:" + str(time))
                        out =  statematrix
                        condition = False
                        break
                try:
                    if verbose:
                        print("timeleft_" + str(i) + ": " + str(timeleft[i]))
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    if verbose:
                        print("timeleft_" + str(i) + ": None")
                    timeleft[i] = None

            if timeleft[i] is not None:
                if verbose:
                    print("timeleft_" + str(i) +"_new:" + str(timeleft[i]))
                timeleft[i] -= 1


        if all(t is None for t in timeleft):
            break

        time += 1

    return statematrix

def loadPieces(dirpath, min_time_steps, verbose=False):
    pieces = {}

    arr = sorted(os.scandir(dirpath), key=lambda x: x.name)

    for f in arr:

        if f.name[-4:] not in ('.mid','.MID'):
            continue

        name = f.name[:-4]

        try:
            outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, f.name),False,verbose)
        except:
            if verbose:
                print('Skip bad file = ', name)
            continue
            
        if len(outMatrix) < min_time_steps:
            if verbose:
                print('Skip too short, but valid file = ', name)
            continue

        pieces[name] = outMatrix
        if verbose:
            print("Loaded {}".format(name))
    return pieces

def noteStateMatrixToMidi(statematrix, name="example", tickscale=48):

    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upperBound-lowerBound
    
    lastcmdtime = 0
    prevstate = [[0,0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        velocity = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
                    velocity.append(n[2])
            elif n[0] == 1:
                onNotes.append(i)
                velocity.append(n[2])
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
            lastcmdtime = time
        for note in enumerate(onNotes):
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=velocity[note[0]], pitch=note[1]+lowerBound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)

def generate_audio(batch_predictions, save_path_dir, midi_name, sample=False, tickscale=48, verbose=False):

    try:
        os.mkdir(save_path_dir)    
        if verbose:
            print('creating new destination folder')
        
    except:
        if verbose:
            print('destination folder exists')
    save_path = save_path_dir + midi_name
    
    if sample:
        midi_matrix =tfp.distributions.Bernoulli(logits=tf.transpose(batch_predictions[:,:,:,0:2], perm=[0,2,1,3]))[0,:,:,:].sample()
        # if note is not played, mask out articulation
        midi_matrix_p = midi_matrix[:,:,0]
        midi_matrix_a = midi_matrix[:,:,1] * midi_matrix[:,:,0] 
        midi_matrix_v = tf.cast(tf.transpose(batch_predictions, perm=[0,2,1,3])[0,:,:,2], dtype=tf.int32)
        midi_matrix = tf.stack([midi_matrix_p, midi_matrix_a, midi_matrix_v], axis=-1)
    else:
        midi_matrix = tf.transpose(batch_predictions, perm=[0,2,1,3])[0,:,:,:]
    midi_matrix = tf.cast(midi_matrix, tf.int32).numpy()
    noteStateMatrixToMidi(midi_matrix, save_path, tickscale)
    if verbose:
        print('Saved midi to', save_path)

def get_piece_summary_df(genre, df, Op_loaded_list):
    genre_works = df.loc[df['Work name'].str.contains(genre)].copy()
    MIDI_file_op_numbers = [i for i in Op_loaded_list if i in genre_works['Op'].tolist()]
    n_MIDI_files_per_op_number = np.unique(MIDI_file_op_numbers, return_counts=True)[1].tolist()
    genre_works['MIDI files'] = n_MIDI_files_per_op_number
    genre_works['genre'] = genre
    return genre_works

def get_subset_training_pieces_for_genre(genre, training_pieces, genre_summary):
    return{k:training_pieces[k] for k in training_pieces.keys() \
     if int(str(k)[4:6]) in genre_summary.loc[genre, 'Op']}