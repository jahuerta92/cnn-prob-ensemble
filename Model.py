from keras.layers import *
from keras.models import Model
from math import floor


# Modelo para pruebas sin gpu
def make_dummy(img_shape, ceil_shape, labels):
    img_input = Input(shape=img_shape)
    ceil_input = Input(shape=(ceil_shape,))
    x = Conv2D(1, (3, 3))(img_input)
    x = Cropping2D((124, 124))(x)
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
def make_prebuilt(prebuilt, freeze_prop=.25):
    def _prebuilt(img_shape, ceil_shape, labels):
        base = prebuilt(include_top=False, weights='imagenet', input_shape=img_shape)
        ceil_input = Input(shape=(ceil_shape,))

        x = Dense(16, activation="relu")(ceil_input)
        x = Dropout(0.5)(x)

        y = GlobalAveragePooling2D()(base.output)
        x = Concatenate()([x, y])
        x = Dense(2048)(x)
        x = Dropout(0.5)(x)
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
    vgg = make_vgg19_block(layer, 32, 1)
    vgg = make_vgg19_block(vgg, 64, 1)
    vgg = make_vgg19_block(vgg, 128, 1)
    vgg = make_vgg19_block(vgg, 256, 2)
    vgg = make_vgg19_block(vgg, 256, 2)

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
    voting = Concatenate()(subnets + [mainnet, ceilnet])
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
    voting = Concatenate()(subnets + [mainnet, ceilnet])
    voting = Dense(32, activation="relu")(voting)
    voting = Dropout(0.5)(voting)

    # Devolver una prediccion final
    predictions = Dense(labels, activation="softmax", name='out')(voting)

    return Model([img_input, ceil_input], [predictions, mainnet_predictions] + subnets_predictions)
