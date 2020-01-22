from math import floor
from Layer import RandomCropping2D

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# Modelo para pruebas sin gpu
def make_dummy(img_shape, ceil_shape, labels):
    img_input = Input(shape=img_shape)
    ceil_input = Input(shape=(ceil_shape,))
    x = Conv2D(1, (3, 3))(img_input)
    x = RandomCropping2D(window=(4, 4), batch_keep=.75)(x)
    x = GlobalAveragePooling2D()(x)
    x = Concatenate()([x, ceil_input])
    x = Dense(1)(x)
    predictions = Dense(labels, activation='softmax', name='out')(x)

    return Model([img_input, ceil_input], predictions)


# Los metodos make_<model> hacen y reciben lo mismo.
# img_shape: tupla con las dimensiones de la entrada (siempre channels last)
# ceil_shape: numero de informacion adicional, 7 habitualmente en este caso
# labels: numero de clases para sacar, 12 habitualmente en este caso
# Pasando una red, devuelve una funcion que usara una una red prehecha de Keras (VGG, inception...)
def make_prebuilt(prebuilt, freeze_prop=.25, wgh='imagenet'):
    def _prebuilt(img_shape, ceil_shape, labels):
        base = prebuilt(include_top=False, weights=wgh, input_shape=img_shape)
        ceil_input = Input(shape=(ceil_shape,))

        x = Dense(16, activation="relu")(ceil_input)
        x = Dropout(0.5)(x)

        y = GlobalAveragePooling2D()(base.output)
        x = Concatenate()([x, y])
        x = Dense(2048)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        predictions = Dense(labels, activation="softmax", name='out')(x)

        disabled = floor(len(base.layers) * freeze_prop)
        for layer in base.layers[:disabled]:
            if 'batch_normalization' not in layer.name:
                layer.trainable = False

        return Model([base.input, ceil_input], predictions)
    return _prebuilt


def make_prebuilt_extended(prebuilt, freeze_prop=.25):
    def _prebuilt(img_shape, ceil_shape, labels):
        base = prebuilt(include_top=False, weights='imagenet', input_shape=img_shape)
        ceil_input = Input(shape=(ceil_shape,))

        x = Dense(16, activation="relu")(ceil_input)
        x = Dropout(0.5)(x)

        y = GlobalAveragePooling2D()(base.output)
        x = Concatenate()([x, y])
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        predictions = Dense(labels, activation="softmax", name='out')(x)

        disabled = floor(len(base.layers) * freeze_prop)
        for layer in base.layers[:disabled]:
            if 'batch_normalization' not in layer.name:
                layer.trainable = False

        return Model([base.input, ceil_input], predictions)
    return _prebuilt


# Crear un bloque de vgg19, _layer es el tensor de entrada
def make_vgg19_block(_layer, filters=64, convolutions=2):
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(_layer)
    for _ in range(convolutions-1):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    return MaxPooling2D((2, 2), strides=(2, 2))(x)


# Crear un bloque
def make_residual_block(_layer, filters=64, convolutions=2):
    res = Conv2D(filters, (1, 1), padding='same')(_layer)
    x = BatchNormalization()(res)
    for i in range(convolutions):
        if i > 0:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, res])
    x = Activation('relu')(x)
    return MaxPooling2D((3, 3), strides=(2, 2))(x)


# Crea una VGG19 manualmente, sin nombres. layer es el tensor de entrada
def make_vgg19_manual(layer):
    vgg = make_vgg19_block(layer)
    vgg = make_vgg19_block(vgg, 128)
    vgg = make_vgg19_block(vgg, 256, 4)
    vgg = make_vgg19_block(vgg, 512, 4)
    vgg = make_vgg19_block(vgg, 512, 4)

    return vgg

# Crea una alternativa a VGG19 más ligera.
def make_shallow_manual(layer):
    vgg = make_vgg19_block(layer, 32, 2)
    vgg = make_vgg19_block(vgg, 64, 2)
    vgg = make_vgg19_block(vgg, 128, 2)
    vgg = make_vgg19_block(vgg, 256, 3)
    vgg = make_vgg19_block(vgg, 256, 3)

    return vgg

def make_residual_manual(layer):
    vgg = make_residual_block(layer, 32, 2)
    vgg = make_residual_block(vgg, 32, 2)
    vgg = make_residual_block(vgg, 64, 2)
    vgg = make_residual_block(vgg, 64, 2)
    vgg = make_residual_block(vgg, 128, 2)
    vgg = make_residual_block(vgg, 128, 2)

    return vgg

# Primera version de la arquitectura cropnet.
def make_cropnetv1(img_shape, ceil_shape, labels, window_size=128, overlap=32):
    size = img_shape[0]
    ceil_input = Input(shape=(ceil_shape,))
    img_input = Input(shape=img_shape)
    offset = size - window_size - overlap

    # Recortar cuatro secciones
    subnet_1 = Cropping2D(cropping=((0, offset), (0, offset)))(img_input)
    subnet_2 = Cropping2D(cropping=((0, offset), (offset, 0)))(img_input)
    subnet_3 = Cropping2D(cropping=((offset, 0), (0, offset)))(img_input)
    subnet_4 = Cropping2D(cropping=((offset, 0), (offset, 0)))(img_input)

    # Declarar las vgg19 de cada seccion
    subnets = [subnet_1, subnet_2, subnet_3, subnet_4]
    subnets = [make_shallow_manual(layer) for layer in subnets]
    subnets = [GlobalAveragePooling2D()(layer) for layer in subnets]
    subnets = [Dense(2048, activation='relu')(layer) for layer in subnets]
    subnets = [Dropout(0.5)(layer) for layer in subnets]
    subnets_predictions = [Dense(labels, activation="softmax", name='out_{}'.format(i+1))(layer)
                           for layer, i in zip(subnets, range(len(subnets)))]

    # Declarar la vgg19 de la imagen completa
    mainnet = make_shallow_manual(img_input)
    mainnet = GlobalAveragePooling2D()(mainnet)
    mainnet = Dense(2048, activation='relu')(mainnet)
    mainnet_predictions = Dense(labels, activation="softmax", name='out_0')(mainnet)

    # Declarar el mlp de la informacion de ceilometro
    ceilnet = Dense(16, activation="relu")(ceil_input)
    ceilnet = Dropout(0.5)(ceilnet)

    # Concatenar votaciones + info de ceilometro
    voting = Concatenate()(subnets_predictions + [mainnet_predictions, ceilnet])
    voting = Dense(32, activation="relu")(voting)
    voting = Dropout(0.5)(voting)

    # Devolver una prediccion final
    predictions = Dense(labels, activation="softmax", name='out')(voting)

    return Model([img_input, ceil_input], predictions)

# Segunda version de la arquitectura cropnet. Incluye una salida auxiliar por cada modelo del ensemble
def make_cropnetv2(img_shape, ceil_shape, labels, window_size=128, overlap=32):
    size = img_shape[0]
    ceil_input = Input(shape=(ceil_shape,))
    img_input = Input(shape=img_shape)
    offset = size - window_size - overlap

    # Recortar cuatro secciones
    subnet_1 = Cropping2D(cropping=((0, offset), (0, offset)))(img_input)
    subnet_2 = Cropping2D(cropping=((0, offset), (offset, 0)))(img_input)
    subnet_3 = Cropping2D(cropping=((offset, 0), (0, offset)))(img_input)
    subnet_4 = Cropping2D(cropping=((offset, 0), (offset, 0)))(img_input)

    # Declarar las resnets de cada seccion
    subnets = [subnet_1, subnet_2, subnet_3, subnet_4]
    subnets = [make_shallow_manual(layer) for layer in subnets]
    subnets = [GlobalAveragePooling2D()(layer) for layer in subnets]
    subnets = [Dense(2048, activation='relu')(layer) for layer in subnets]
    subnets = [Dropout(0.5)(layer) for layer in subnets]
    subnets_predictions = [Dense(labels, activation="softmax", name='out_{}'.format(i+1))(layer)
                           for layer, i in zip(subnets, range(len(subnets)))]

    # Declarar la vgg19 de la imagen completa
    mainnet = make_shallow_manual(img_input)
    mainnet = GlobalAveragePooling2D()(mainnet)
    mainnet = Dense(2048, activation='relu')(mainnet)
    mainnet_predictions = Dense(labels, activation="softmax", name='out_0')(mainnet)

    # Declarar el mlp de la informacion de ceilometro
    ceilnet = Dense(16, activation="relu")(ceil_input)
    ceilnet = Dropout(0.5)(ceilnet)

    # Concatenar votaciones + info de ceilometro
    voting = Concatenate()(subnets_predictions + [mainnet_predictions, ceilnet])
    voting = Dense(128, activation="relu")(voting)
    voting = Dropout(0.5)(voting)

    # Devolver una prediccion final
    predictions = Dense(labels, activation="softmax", name='out')(voting)

    return Model([img_input, ceil_input], [predictions, mainnet_predictions] + subnets_predictions)

# Tercera version de la arquitectura cropnet. Cambio las subredes convolucionales tipo vgg por resnet
def make_cropnetv3(img_shape, ceil_shape, labels, window_size=128, overlap=32):
    size = img_shape[0]
    ceil_input = Input(shape=(ceil_shape,))
    img_input = Input(shape=img_shape)
    offset = size - window_size - overlap

    # Recortar cuatro secciones
    subnet_1 = Cropping2D(cropping=((0, offset), (0, offset)))(img_input)
    subnet_2 = Cropping2D(cropping=((0, offset), (offset, 0)))(img_input)
    subnet_3 = Cropping2D(cropping=((offset, 0), (0, offset)))(img_input)
    subnet_4 = Cropping2D(cropping=((offset, 0), (offset, 0)))(img_input)

    # Declarar las vgg19 de cada seccion
    subnets = [subnet_1, subnet_2, subnet_3, subnet_4]

    def make_net(_layer, out_name):
        x = make_residual_manual(_layer)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        return Dense(labels, activation="softmax", name=out_name)(x)

    # Declarar las subredes y
    subnets_predictions = [make_net(layer, 'out_{}'.format(i+1)) for i, layer in enumerate(subnets)]
    mainnet_predictions = make_net(img_input, 'out_0')

    # Declarar el mlp de la informacion de ceilometro
    ceilnet = Dense(16, activation="relu")(ceil_input)
    ceilnet = Dropout(0.5)(ceilnet)

    # Concatenar votaciones + info de ceilometro
    voting = Concatenate()(subnets_predictions + [mainnet_predictions, ceilnet])
    voting = Dense(128, activation="relu")(voting)
    voting = Dropout(0.5)(voting)

    # Devolver una prediccion final
    predictions = Dense(labels, activation="softmax", name='out')(voting)

    return Model([img_input, ceil_input], [predictions, mainnet_predictions] + subnets_predictions)


def make_very_small_net(_layer, start_filters=4):
    def make_very_small_layer(__layer, filters):
        x = Conv2D(filters, (1, 3), padding='same')(__layer)
        x = Conv2D(filters, (3, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D((3, 3), (2, 2))(x)
        return x

    x = make_very_small_layer(_layer, start_filters)
    for i in range(3):
        x = make_very_small_layer(x, start_filters * (2 ** (i+1)))
    return x


def make_rcropnetv1(n_crops=4):
    def _make_rcropnetv1(img_shape, ceil_shape, labels, window_size=128):
        size = img_shape[0]
        ceil_input = Input(shape=(ceil_shape,))
        img_input = Input(shape=img_shape)

        # Declarar las vgg19 de cada seccion
        wsize = window_size

        def make_net(_layer, net_maker, out_name, cropping=True):
            if cropping:
                x = RandomCropping2D(window=(wsize, wsize))(_layer)
                x = net_maker(x)
            else:
                x = net_maker(_layer)
            x = GlobalAveragePooling2D()(x)
            dense = 16
            if not cropping:
                dense = 256
            x = Dense(dense, activation='relu')(x)
            x = Dropout(0.5)(x)
            return Dense(labels, activation="softmax", name=out_name)(x)

        # Declarar las subredes y
        subnets_predictions = [make_net(img_input, make_very_small_net, 'out_{}'.format(i + 1)) for i in
                               range(n_crops)]
        mainnet_predictions = make_net(img_input, make_shallow_manual, 'out_0', cropping=False)

        # Declarar el mlp de la informacion de ceilometro
        ceilnet = Dense(16, activation="relu")(ceil_input)
        ceilnet = Dropout(0.5)(ceilnet)

        # Concatenar votaciones + info de ceilometro
        voting = Concatenate()(subnets_predictions + [mainnet_predictions, ceilnet])
        voting = Dense(128, activation="relu")(voting)
        voting = Dropout(0.5)(voting)

        # Devolver una prediccion final
        predictions = Dense(labels, activation="softmax", name='out')(voting)

        return Model([img_input, ceil_input], [predictions, mainnet_predictions] + subnets_predictions)
    return _make_rcropnetv1


def make_rcropnetv2(n_crops=4, window_size=128):
    def _make_rcropnetv2(img_shape, ceil_shape, labels):
        size = img_shape[0]
        ceil_input = Input(shape=(ceil_shape,))
        img_input = Input(shape=img_shape)

        # Declarar las vgg19 de cada seccion
        wsize = window_size

        # Declarar el mlp de la informacion de ceilometro
        ceilnet = Dense(16, activation="relu")(ceil_input)
        ceilnet = Dropout(0.5)(ceilnet)

        def make_net(_layer, net_maker):
            x = net_maker(_layer)
            x = GlobalAveragePooling2D()(x)
            dense = 256
            x = Concatenate()([ceilnet, x])
            x = Dense(dense, activation='relu')(x)
            return Dropout(0.5)(x)

        crop = RandomCropping2D(window=(wsize, wsize))(img_input)
        subnet = make_net(crop, make_shallow_manual)

        subnet_blueprint = Model(inputs=[img_input, ceil_input], outputs=subnet)

        # Declarar las subredes
        subnets = [subnet_blueprint([img_input, ceil_input]) for _ in range(n_crops)]
        subnets_predictions = [Dense(labels, activation="softmax", name='out_{}'.format(i + 1))(net)
                               for i, net in enumerate(subnets)]
        mainnet = make_net(img_input, make_shallow_manual)
        mainnet_predictions = Dense(labels, activation="softmax", name='out_0')(mainnet)

        # Concatenar votaciones + info de ceilometro
        predictions = Average(name='out')(subnets_predictions + [mainnet_predictions])

        return Model([img_input, ceil_input], [predictions, mainnet_predictions] + subnets_predictions)
    return _make_rcropnetv2
