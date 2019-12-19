from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras.models import Model
from math import floor


def make_vgg19(img_shape, ceil_shape, labels):
    base = VGG19(include_top=False, weights='imagenet', input_shape=img_shape)
    ceil_input = Input(shape=(ceil_shape,))

    x = Dense(16, activation="relu")(ceil_input)
    x = Dropout(0.5)(x)

    y = GlobalAveragePooling2D()(base.output)
    x = Concatenate()([x, y])
    x = Dense(2048)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)

    predictions = Dense(labels, activation="softmax")(x)

    disabled = floor(len(base.layers) * 0.25)
    for layer in base.layers[:disabled]:
        if 'batch_normalization' not in layer.name:
            layer.trainable = False

    return Model([base.input, ceil_input], predictions)

def make_cropnet(img_shape, ceil_shape, labels, window_size=128, overlap=32):
    #base = VGG19(include_top=False, weights='imagenet', input_shape=img_shape)
    h, w, ch = img_shape
    ceil_input = Input(shape=(ceil_shape,))
    img_input = Input(shape=img_shape)
    offset_high = h - window_size + overlap
    offset_low = window_size - overlap

    subnet_1 = Cropping2D(cropping=((0, offset_high), (0, offset_high)))(img_input)
    subnet_2 = Cropping2D(cropping=((0, offset_high), (offset_low, h)))(img_input)
    subnet_3 = Cropping2D(cropping=((offset_low, h), (0, offset_high)))(img_input)
    subnet_4 = Cropping2D(cropping=((offset_low, h), (offset_low, h)))(img_input)

    subnets = [subnet_1, subnet_2, subnet_3, subnet_4]
    subnets = [VGG19(include_top=False, weights=None)(layer) for layer in subnets]
    subnets = [GlobalAveragePooling2D()(layer) for layer in subnets]
    subnets = [Dense(2048, activation='relu')(layer) for layer in subnets]
    subnets = [Dropout(0.5)(layer) for layer in subnets]
    subnets = [Dense(labels, activation="softmax")(layer) for layer in subnets]

    mainnet = VGG19(include_top=False, weights=None)(img_input)
    mainnet = GlobalAveragePooling2D()(mainnet)
    mainnet = Dense(2048, activation='relu')(mainnet)
    mainnet = Dense(labels, activation="softmax")(mainnet)

    ceilnet = Dense(16, activation="relu")(ceil_input)
    ceilnet = Dropout(0.5)(ceilnet)

    voting = Concatenate()(subnets + [mainnet, ceilnet])
    voting = Dense(32, activation="relu")(voting)
    voting = Dropout(0.5)(voting)

    predictions = Dense(labels, activation="softmax")(voting)

    return Model([img_input, ceil_input], predictions)
