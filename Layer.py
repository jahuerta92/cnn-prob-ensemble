from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec

from math import floor
import numpy as np


class RandomCropping2D(Layer):
    '''Cropping layer for 2D input (e.g. picture).
    It crops along spatial dimensions at random, i.e. width and height.
    # Arguments
        window: tuple of int (length 2)
            Height and width of the future output. Everything else is trimmed.
        seed: int
            Random seed for cropping each window. For reproducibility.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    # Input shape
        4D tensor with shape:
        (samples, depth, first_axis_to_crop, second_axis_to_crop)
    # Output shape
        4D tensor with shape:
        (samples, depth, first_cropped_axis, second_cropped_axis)
    # Examples
    ```python
        # crop the input image and feature meps
        model = Sequential()
        model.add(RandomCropping2D(cropping=(8,16), input_shape=(3, 28, 28)))
        # now model.output_shape == (None, 3, 20, 12)
        model.add(Convolution2D(64, 3, 3, border_mode='same))
        model.add(RandomCropping2D(cropping=(2,2)))
        # now model.output_shape == (None, 64, 18, 10)
    ```
    '''

    def __init__(self, window=None, dim_ordering='default', **kwargs):
        super(RandomCropping2D, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = 'tf'
        assert window is not None, 'window must be a tuple'
        self.window = tuple(window)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            return (input_shape[0],
                    input_shape[1],
                    self.window[0],
                    self.window[1])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    self.window[0],
                    self.window[1],
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, training=None):
        assert x.shape is not None, 'Input layer must have a batch size defined'
        input_shape = x.shape
        window_w, window_h = self.window

        def randrange(v):
            return tf.random.uniform((1,), 0, v, dtype='int32')[0]

        def random_slice_on(_x, w, h):
            img_w, img_h = input_shape[w], input_shape[h]
            rand_w, rand_h = randrange(img_w - window_w), randrange(img_h - window_h)
            if w is 1:
                return _x[:, rand_w:(rand_w + window_w), rand_h:(rand_h + window_h), :]
            elif w is 2:
                return _x[:, :, rand_w:(rand_w + window_w), rand_h:(rand_h + window_h)]

        # tf.map_fn(lambda img: tf.image.random_crop(img, [input_shape[0], window_w, window_h], x)
        if self.dim_ordering == 'th':
            return random_slice_on(x, 2, 3)
        elif self.dim_ordering == 'tf':
            return random_slice_on(x, 1, 2)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def get_config(self):
        config = {'window': self.window}
        base_config = super(RandomCropping2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
