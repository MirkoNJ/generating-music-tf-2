import tensorflow as tf
import numpy as np
import os
import modules.batch as batch

num_notes_octave = 12
Midi_low = 21
Midi_high = 108

def inputKernel(input_data, Midi_low=21, Midi_high=108, time_init=0):
    """
    Arguments:
        input_data: size = [batch_size x num_notes x num_timesteps x 2] 
            (the input data represents that at the previous timestep of what we are trying to predict)
        Midi_low: integer
        Midi_high: integer
    Returns:
        Note_State_Expand: size = [batch_size x num_notes x num_timesteps x 80]
    """ 

    # capture input_data dimensions (batch_size and num_timesteps are variable length)
    batch_size = input_data.shape[0]    #  16
    num_notes = input_data.shape[1]     #  88
    num_timesteps = input_data.shape[2] # 128

    # MIDI note number (only a function of the note index)
    Midi_indices = tf.squeeze(tf.range(start=Midi_low, limit = Midi_high+1, delta=1))
    Midi_indices = (Midi_indices - Midi_low) / (Midi_high+1- Midi_low) # min-max-scale to [0,1]
    x_Midi = tf.ones((batch_size, num_timesteps, 1, num_notes))*tf.cast(Midi_indices, dtype=tf.float32)
    x_Midi = tf.transpose(x_Midi, perm=[0,3,1,2]) # shape (16, 88, 128, 1) -> [batch_size, num_notes, num_timesteps, 1]

    # part_pitchclass (only a function of the note index)
    Midi_pitchclasses = tf.squeeze(tf.cast(x_Midi * (Midi_high+1- Midi_low) + Midi_low, dtype=tf.int32) % num_notes_octave, axis=3)
    x_pitch_class = tf.one_hot(tf.cast(Midi_pitchclasses, dtype=tf.uint8), depth=num_notes_octave) # shape (16, 88, 128, 12)

    # part_prev_vicinity
    input_flatten = tf.cast(tf.transpose(input_data, perm=[0,2,1,3]), dtype=tf.float32) # remove velocity if defined
    input_flatten = tf.reshape(input_flatten, [batch_size*num_timesteps, num_notes, 3]) # shape (128*16=2048, 88, 3)
    input_flatten_p = tf.slice(input_flatten, [0,0,0],size=[-1, -1, 1])                 # shape (2048, 88, 1) -> [batch size, width, in channels]
    input_flatten_a = tf.slice(input_flatten, [0,0,1],size=[-1, -1, 1])                 # shape (2048, 88, 1) -> [batch size, width, in channels]
    input_flatten_vel = tf.slice(input_flatten, [0,0,2],size=[-1, -1, 1])                 # shape (2048, 88, 1) -> [batch size, width, in channels]


    # reverse identity kernel
    filt_vicinity = tf.cast(tf.expand_dims(tf.eye(num_notes_octave * 2 + 1), axis=1), dtype=tf.float32) 
    # shape (25,1,25) = [kernel_1d_size, in_channels, "number_of_kernels_1d" = out_channels]
    # the relative values one octave up and one octave down

    # 1D convolutional filter for each play and articulate arrays 
    vicinity_p = tf.nn.conv1d(input_flatten_p, filt_vicinity, stride=1, padding='SAME') # shape (2048, 88, 25)
    vicinity_a = tf.nn.conv1d(input_flatten_a, filt_vicinity, stride=1, padding='SAME') # shape (2048, 88, 25)
    vicinity_vel = tf.nn.conv1d(input_flatten_vel, filt_vicinity, stride=1, padding='SAME') # shape (2048, 88, 25)
    vicinity_vel = (vicinity_vel - 0) / (127- 0) # min-max-scale to [0,1]
    
    # concatenate back together and restack such that play-articulate-velocity numbers alternate
    vicinity = tf.stack([vicinity_p, vicinity_a, vicinity_vel], axis=3) # 1 array shape (2048, 88, 25, 3)
    vicinity = tf.unstack(vicinity, axis=2)               # 25 arrays of shape (2048, 88, 3)
    vicinity = tf.concat(vicinity, axis=2)                # 1 array shape (2048, 88, 75) 

    # reshape by major dimensions, THEN swap axes
    x_vicinity = tf.reshape(vicinity, shape = [batch_size, num_timesteps, num_notes, (num_notes_octave * 2 + 1) * 3]) # shape (16, 128, 88, 75)
    x_vicinity = tf.transpose(x_vicinity, perm=[0,2,1,3]) # shape (16, 88, 128, 75)

    # kernel
    filt_context = tf.expand_dims(tf.tile(tf.eye(num_notes_octave), multiples=[(num_notes // num_notes_octave) * 2, 1]), axis=1) 
    # shape (168, 1, 12) = [kernel_1d_size, in_channels, "number_of_kernels_1d" = out_channels]
    # n = num_notes // num_notes_octave * 2 stacked identy matrices where n is the rounded number of octaves, times two, 
    # for both directions (eg. n = 14 -> consider the seven higher and the seven lower octaves)

    # part_prev_context
    context = tf.nn.conv1d(input_flatten_p, filt_context, stride=1, padding='SAME')
    x_context = tf.reshape(context, shape=[batch_size, num_timesteps, num_notes, num_notes_octave])
    x_context = tf.transpose(x_context, perm=[0,2,1,3])

    # beat (only a function of the time axis index plus the time_init value)
    Time_indices = tf.range(time_init, num_timesteps + time_init)
    x_Time = tf.reshape(tf.tile(Time_indices, multiples=[batch_size*num_notes]), shape=[batch_size, num_notes, num_timesteps,1])
    x_beat = tf.cast(tf.concat([x_Time%2,  x_Time//2%2, x_Time//4%2, x_Time//8%2, (x_Time%3)%2, (x_Time%3)//2%2], axis=-1), dtype=tf.float32) #(x_Time%3)%2, (x_Time%3)//2%2 -> 48th notes per bar instead of 16 -> two more bits necessary

    # add the mean velocity of one octave up and one octave down
    velocity_count = tf.cast(tf.math.count_nonzero(tf.math.round(vicinity_p), axis=2, keepdims=True), dtype=tf.float32)
    velocity_sum = tf.math.reduce_sum(vicinity_vel, axis=2, keepdims=True)
    x_velocity = velocity_sum / tf.math.maximum(velocity_count,1) # shape (2048, 88, 1), avoid division by zero
    x_velocity = tf.reshape(x_velocity, shape = [batch_size, num_timesteps, num_notes, 1]) 
    x_velocity = tf.transpose(x_velocity, perm=[0,2,1,3]) # shape (16, 88, 128, 1)

    # zero
    x_zero = tf.zeros([batch_size, num_notes, num_timesteps,1])

    # final array (input vectors per batch, note and timestep)
    Note_State_Expand = tf.concat([x_Midi, x_pitch_class, x_vicinity, x_context, x_beat, x_velocity, x_zero], axis=-1)
    
    return Note_State_Expand

def noteRNNInputSummary(x):
    x = tf.cast(x, dtype=tf.float32).numpy()
    x[0] = x[0] * (Midi_high+1 - Midi_low) + Midi_low
    for i in range(13,88):
        if i%3 == 0:
            x[i]=x[i]*127
    x[106] = x[106] * 127
    x= np.round(x)
    x = x.astype(int)
    res = {}
    res = {'note' : x[0]}
    res['pitch_class'] = x[1:13]
    res['part_prev_vicinity'] = x[13:88]
    res['part_prev_vicinity_lower'] = x[13:49]
    res['part_prev_vicinity_same'] = x[49:52]
    res['part_prev_vicinity_higher'] = x[52:88]
    res['part_context'] = x[88:100]
    res['beat'] = x[100:106]
    res['velocity'] = x[106]
    res['zero'] = x[107]
    return res

def alignXy(X, y):

    y_new_shape = tf.TensorShape(np.array(y.shape.as_list())+ np.array([0,0,-1,0]))
    X_new_shape = tf.TensorShape(np.array(X.shape.as_list())+ np.array([0,0,-1,0]))
    y = tf.slice(y, [0,0,1,0], y_new_shape)
    X = tf.slice(X, [0,0,0,0], X_new_shape)
    return X, y

def createDataSet(pieces, sample_size, num_timesteps, batch_size = 2):

    num_timesteps += 1
    y = batch.getPieceBatch(pieces, sample_size, num_timesteps)
    X = inputKernel(y, Midi_low, Midi_high)
    X, y = alignXy(X, y)
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

    return dataset
    
