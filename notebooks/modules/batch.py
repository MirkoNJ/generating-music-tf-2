import random
import tensorflow as tf

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