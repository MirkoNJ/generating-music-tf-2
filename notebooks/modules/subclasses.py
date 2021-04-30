import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.keras import backend as K

from tensorflow.keras.layers import Dense, Layer, LSTMCell

from tensorflow.python.keras.utils import losses_utils
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from tensorflow_addons.metrics import MeanMetricWrapper


class BatchReshape(Layer):
    
    def __init__(self, target_shape, back_transform, **kwargs):
        """Creates a `tf.keras.layers.Reshape`  layer instance.
        Args:
          target_shape: Target shape. Tuple of integers, including the
            samples dimension (batch size).
          **kwargs: Any additional layer keyword arguments.
        """
        super(BatchReshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)
        self.back_transform = back_transform
        
    def call(self, inputs, shape=None):
        if self.back_transform: 
            if shape is not None:
                result = array_ops.reshape(inputs,(array_ops.shape(inputs)[0]/array_ops.shape(shape)[0],) +  self.target_shape)
            else:
                result = array_ops.reshape(inputs,(array_ops.shape(inputs)[0]/self.target_shape[0],) +  self.target_shape)

        else:
            result = array_ops.reshape(inputs,(array_ops.shape(inputs)[0]*array_ops.shape(inputs)[1],) +  self.target_shape)
        return result
    
    def get_config(self):
        config = {
            'target_shape': self.target_shape, 
            'back_transform': self.back_transform
            }
        base_config = super(BatchReshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class GetShape(Layer):
    def init(self):
        super(GetShape,self).__init__()

    def build(self,input_shape):
        super(GetShape,self).build(input_shape)

    def call(self,inputs):
        output = inputs[0,0,:,0]
        return output

class SampleNote(Layer):

    def __init__(self, **kwargs):
        super(SampleNote, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SampleNote, self).build(input_shape)  

    def call(self, y_in):
        
        note = tfp.distributions.Bernoulli(logits=y_in[:,0:2], dtype=y_in.dtype).sample()
        p = array_ops.slice(note, [0,0], [-1,1])
        a = array_ops.slice(note, [0,1], [-1,1])          
        a = p*a
        v = y_in[:,2:3]
        note = array_ops.concat([p,a,v], axis=-1)
        return note

class SliceNotesTensor(Layer):
    def init(self):
        super(SliceNotesTensor,self).__init__()

    def build(self,input_shape):
        super(SliceNotesTensor,self).build(input_shape)

    def call(self,inputs):
        notes = inputs.shape[1]
        output = [inputs[:,n,:] for n in range(notes)]
        return output

class SliceNotesVelocityTensor(Layer):
    def init(self):
        super(SliceNotesVelocityTensor,self).__init__()

    def build(self,input_shape):
        super(SliceNotesVelocityTensor,self).build(input_shape)

    def call(self,inputs):
        
        output = [inputs[:,:,:,0:2], inputs[:,:,:,2:3]]
        return output

class SliceNoteVelocityTensor(Layer):
    def init(self):
        super(SliceNoteVelocityTensor,self).__init__()

    def build(self,input_shape):
        super(SliceNoteVelocityTensor,self).build(input_shape)

    def call(self,inputs):
        output = inputs[:,0:2]
        return output


class ExpandDims(Layer):
    def init(self):
        super(ExpandDims,self).__init__()

    def build(self,input_shape):
        super(ExpandDims,self).build(input_shape)

    def call(self, inputs, axis=1):
        output = [K.expand_dims(x, axis=axis) for x in inputs]
        return output

class BackTransformVelocity(Layer):
    def init(self):
        super(BackTransformVelocity,self).__init__()

    def build(self,input_shape):
        super(BackTransformVelocity,self).build(input_shape)

    def call(self,inputs):
        output = tf.math.sigmoid(inputs)
        output = tf.cast(output * 127, dtype=tf.float32)
        return output



class CustomSigmoidFocalCrossEntropy(LossFunctionWrapper):
    def __init__(
        self,
        from_logits: bool = False,
        alpha: FloatTensorLike = 0.25,
        gamma: FloatTensorLike = 2.0,
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: str = "sigmoid_focal_crossentropy",
    ):
        super().__init__(
            custom_sigmoid_focal_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            alpha=alpha,
            gamma=gamma,
        )

@tf.function
def custom_sigmoid_focal_crossentropy(
    y_true: TensorLike,
    y_pred: TensorLike,
    alpha: FloatTensorLike = 0.25,
    gamma: FloatTensorLike = 2.0,
    from_logits: bool = False,
) -> tf.Tensor:
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero")

    y_pred = tf.convert_to_tensor(y_pred)[:,:,:,0:2]
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)[:,:,:,0:2]

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # if note is not played, mask out loss term for articulation   
    ce_p = ce[:,:,:,0]
    ce_a = ce[:,:,:,1] * y_true[:,:,:,0] 
    ce = tf.stack([ce_p, ce_a], axis=-1)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    # if note is not played, mask out probability for articulation   
    pred_prob_p = pred_prob[:,:,:,0]
    pred_prob_a = pred_prob[:,:,:,1] * y_true[:,:,:,0] 
    pred_prob = tf.stack([pred_prob_p, pred_prob_a], axis=-1)

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)

class CustomBinaryAccuracy(MeanMetricWrapper):

    def __init__(self, name='binary_accuracy', dtype=None, threshold=0.5):
        super(CustomBinaryAccuracy, self).__init__(
            custom_binary_accuracy, name, dtype=dtype, threshold=threshold)

@tf.function
def custom_binary_accuracy(y_true, y_pred, threshold=0.5):
    # from logits
    y_true = y_true[:,:,:,0:2]
    y_pred = tf.sigmoid(y_pred)

    # if note is not played, mask out missclassification from  accuracy
    y_pred_p = y_pred[:,:,:,0]
    y_pred_a = y_pred[:,:,:,1] * y_true[:,:,:,0] 
    y_pred = tf.stack([y_pred_p, y_pred_a], axis=-1)

    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    return K.mean(math_ops.equal(y_true, y_pred), axis=-1)

class MeanSquaredErrorVelocity(LossFunctionWrapper):

    def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_squared_error'):
        super().__init__(
            mean_squared_error_velocity, name=name, reduction=reduction)


@tf.function
def mean_squared_error_velocity(y_true, y_pred):

    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    # mask error in prediction if note is not played
    y_pred = y_pred[:,:,:,0:1] * y_true[:,:,:,0:1]

    return K.sum(math_ops.squared_difference(y_pred, y_true[:,:,:,2:3]), axis=-1)

@tf.function
def root_mean_squared_error_velocity_metric(y_true, y_pred):

    error = mean_squared_error_velocity(y_true, y_pred)
    error_sum = tf.math.reduce_sum(error)
    error_count =  math_ops.cast(tf.math.count_nonzero(error), error_sum.dtype)
    error_count = tf.math.maximum(error_count,1) # avoid division by zero

    return tf.math.sqrt(error_sum/error_count)


