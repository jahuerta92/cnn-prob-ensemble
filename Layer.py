from __future__ import absolute_import

from keras.engine import Layer, InputSpec
import keras.backend as K
from math import floor
import random as rnd

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

    def __init__(self, window=None, random_seed=1, dim_ordering='default', **kwargs):
        super(RandomCropping2D, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        assert window is not None, 'window must be a tuple'
        self.window = tuple(window)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]
        self.random_seed = random_seed


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

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        window_w, window_h = self.window
        rnd.seed(self.random_seed)

        def randrange(v):
            tensor = K.random_uniform((2,), 0, v, dtype='int32')
            value = K.gather(tensor, 0)
            return value

        # tf.map_fn(lambda img: tf.image.random_crop(img, [input_shape[0], window_w, window_h], x)
        if self.dim_ordering == 'th':
            img_w, img_h = input_shape[2], input_shape[3]
            rand_w, rand_h = randrange(img_w - window_w), randrange(img_h - window_h)
            return x[:, :, rand_w:rand_w+window_w, rand_h:rand_h+rand_h]
        elif self.dim_ordering == 'tf':
            img_w, img_h = input_shape[1], input_shape[2]
            rand_w, rand_h = randrange(img_w - window_w), randrange(img_h - window_h)
            return x[:, rand_w:rand_w+window_w, rand_h:rand_h+rand_h, :]

    def get_config(self):
        config = {'window': self.window, 'random_seed': self.random_seed}
        base_config = super(RandomCropping2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
