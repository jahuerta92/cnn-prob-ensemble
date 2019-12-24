import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import load_model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from random import seed

# file_dir: Directorio en el que se encuentran los archivos feat_file e img_file
# _prop: Proporcion entrenamiento y validacion (test por omision)
# _file: Nombre del fichero
# Devuelve un diccionario con 'train' 'valid' 'test' con tuplas de tipo (imagen,ceilometro,etiqueta)
# 'label_encoder' es el LabelBinarizer que transforma la clave en columnas
def load_data(file_dir, train_prop=.7, valid_prop=.1, feat_file='cloud_features.csv', img_file='images.npz'):
    features = pd.read_csv('%s/%s' % (file_dir, feat_file), sep=';', decimal=',')
    cloud_type = np.array(features['cloud.type'])
    ceil_info = np.array(features[["ceil.height0", "ceil.height1",
                                   "ceil.height2", "ceil.depth0",
                                   "ceil.depth1", "ceil.depth2",
                                   "ceil.layers"]])

    # De string a una matriz de clases para la red
    encoder = LabelBinarizer()
    cloud_encoded = encoder.fit_transform(cloud_type)

    # Escalado de las variables de entrada
    ceil_info = scale(ceil_info, copy=False)

    seed(1)
    in_train, in_test = train_test_split(np.array(range(0,len(cloud_type))),
                                         train_size=train_prop + valid_prop,
                                         stratify=cloud_type, random_state=1)
    in_train, in_valid = train_test_split(in_train, train_size=train_prop / (train_prop + valid_prop),
                                          stratify=cloud_type[in_train],
                                          random_state=1)
    y_train, y_test, y_valid = cloud_encoded[in_train], cloud_encoded[in_test], cloud_encoded[in_valid]
    ceil_train, ceil_test, ceil_valid = ceil_info[in_train], ceil_info[in_test], ceil_info[in_valid]

    del ceil_info
    del cloud_encoded
    del cloud_type

    print('Label train set shape: %d, %d' % y_train.shape)
    print('Label valid set shape: %d, %d' % y_valid.shape)
    print('Label test set shape: %d, %d' % y_test.shape)

    images = np.load('%s/%s' % (file_dir, img_file), 'r', True)['arr_0']

    # Particion de las imagenes
    img_train, img_test, img_valid = images[in_train], images[in_test], images[in_valid]
    img_test = img_test.astype('float32', copy=False)

    del images

    print('')
    print('Image train set shape: %d, %d, %d, %d' % img_train.shape)
    print('Image valid set shape: %d, %d, %d, %d' % img_valid.shape)
    print('Image test set shape: %d, %d, %d, %d' % img_test.shape)

    data = {'train': (img_train, ceil_train, y_train , ),
            'valid': (img_valid, ceil_valid, y_valid),
            'test': (img_test, ceil_test, y_test),
            'label_encoder': encoder}

    return data

# Crea un ImageDataGenerator generico a partir de los datos de entrenamiento.
def make_data_generator(train_data):
    img_train, ceil_train, y_train = train_data
    dgen = ImageDataGenerator(featurewise_center=True, samplewise_center=True,
                             rotation_range=180, width_shift_range=.3,
                             height_shift_range=.3, brightness_range=[.5, 1.0],
                             zoom_range=[.5, 1.0], shear_range=45,
                             fill_mode='nearest', horizontal_flip=True,
                             vertical_flip=True)
    dgen.fit(img_train)
    return dgen


# train_data: Datos de entrenamiento (img, ceil, y)
# valid_data: Datos de validacion
# data_generator: ImageDataGenerator
# model_builder: Constructor de modelos
# model_name: nombre del modelo a guardar
# model_dir: directorio del modelo a guardar
# Entrena un modelo construido a partir de un constructor de Model
def fit_model(train_data, valid_data, data_generator, model_builder, model_name, model_dir='./results',
              max_epochs=1000, batch_size=64, lr=1e-4, n_outputs=1):
    img_train, ceil_train, y_train = train_data
    img_valid, ceil_valid, y_valid = valid_data

    label_num = y_train.shape[1]
    ceil_features = ceil_train.shape[1]
    img_shape = img_train.shape[1:]

    model = model_builder(img_shape, ceil_features, label_num)  # Model([base.input, ceil_input], predictions)

    # Declarar losses y accuracies en caso de que haya mas de una salida
    losses = {'out': 'categorical_crossentropy'}
    accuracies = {'out': 'accuracy'}

    if n_outputs > 1:
        losses.update({'out_%d' % i: 'categorical_crossentropy' for i in range(n_outputs-1)})
        accuracies.update({'out_%d' % i: 'accuracy' for i in range(n_outputs-1)})

    # Compilar el modelo
    model.compile(loss=losses,
                  optimizer=Adam(lr=lr, clipnorm=1.),
                  metrics=accuracies)
    print('Layers: %d' % len(model.layers))
    model.summary()

    # Entrenar el modelo
    seed(1)

    monitored_metric = 'val_acc'
    if n_outputs > 1:
        monitored_metric = 'val_out_acc'

    callback_list = [ModelCheckpoint('%s/%s' % (model_dir, '%s_model.h5' % model_name),
                                     monitor=monitored_metric, save_best_only=True),
                     EarlyStopping(monitor=monitored_metric, min_delta=0.0001, patience=25)]

    # Generar el flow de salidas (No soporta multisalida por defecto DataGenGenerator)
    def multi_output_generator(gen, X, y, bs=batch_size):
        gen_x = gen.flow(X, y, seed=1, batch_size=bs)
        while True:
            x_next, y_next = gen_x.next()
            yield x_next, [y_next[:] for _ in range(n_outputs)]

    model.fit_generator(multi_output_generator(data_generator, (img_train, ceil_train), y_train),
                        steps_per_epoch=len(img_train) / batch_size,
                        epochs=max_epochs,
                        verbose=2,
                        validation_data=multi_output_generator(data_generator, (img_valid, ceil_valid), y_valid),
                        validation_steps=len(img_valid) / batch_size,
                        callbacks=callback_list)

    return model


# file_dir: Directorio en el que se encuentran los archivos a cargar y guardar
# model_name: Nombre del modelo usado
# encoder: Objeto usado para transformar la clase a columnas
# normalizer: DataGenerator para normalizar las entradas
# test_data: Datos de evaluacion
# 'encoder' es el LabelBinarizer que transforma la clave en columnas
def save_results(file_dir, model_name, encoder, data_generator, test_data, n_outputs=1):
    # Cargar y evaluar el mejor modelo
    model = load_model('%s/%s' % (file_dir, '%s_model.h5' % model_name))
    img_test, ceil_test, y_test = test_data

    # Extraer las predicciones del modelo
    standard_img_test = data_generator.standardize(img_test)
    model_test_predictions = model.predict([standard_img_test, ceil_test])

    if len(model_test_predictions) == n_outputs and len(model_test_predictions) > 1:
        model_test_predictions = model_test_predictions[0]

    decoded_predictions = encoder.inverse_transform(model_test_predictions)
    decoded_observations = encoder.inverse_transform(y_test)
    pred_obs = pd.DataFrame(data={'pred': decoded_predictions, 'obs': decoded_observations})

    # Evaluar el Modelo

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    matrix = confusion_matrix(pred_obs['obs'], pred_obs['pred'])
    names = pred_obs['obs'].unique()

    sns.heatmap(matrix, annot=True, cbar=False, xticklabels=names, yticklabels=names)
    print(classification_report(pred_obs['obs'], pred_obs['pred']))

    # Almacenar las predicciones del modelo entrenado
    pred_obs.to_csv('%s/%s' % (file_dir, '%s_preds.csv' % model_name))
