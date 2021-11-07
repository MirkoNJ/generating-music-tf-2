import random
import tensorflow as tf
import pdb

division_len = 48 # interval between possible start locations -> implies only new bars, since 48th notes


def getPieceSegment(pieces, num_time_steps):
    piece_output = random.choice(list(pieces.values()))
    start = random.randrange(0, (len(piece_output) - num_time_steps), division_len)
    seg_out = piece_output[start : (start + num_time_steps)]

    return seg_out

def getPieceBatch(pieces, batch_size, num_time_steps):
    out = [getPieceSegment(pieces, num_time_steps) for _ in range(batch_size)]
    out = tf.convert_to_tensor(out, dtype=tf.float32)
    out = tf.transpose(out, perm=[0,2,1,3])
    return out

def getPieceSegment2(piece, num_time_steps, start):
    if len(piece) < (start + num_time_steps):
        start = len(piece) - num_time_steps
    # print("Start: {}".format(start))
    # print("End: {}".format(start + num_time_steps))
    return piece[start : (start + num_time_steps)]
    
def getPieceBatch2(piece, batch_size, num_time_steps, start_old):
    starts = [int(start_old + x * num_time_steps * 0.5) for x in range(batch_size)]
    out = [getPieceSegment2(piece, num_time_steps, start) for start in starts]
    out = tf.convert_to_tensor(out, dtype=tf.float32)
    out = tf.transpose(out, perm=[0,2,1,3])
    start_out = int(max(starts)+num_time_steps*0.5)
    return out, start_out