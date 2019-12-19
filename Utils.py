# Cargar los datos con pandas y numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from random import seed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import load_model


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

    print('Image train set shape: %d, %d, %d, %d' % img_train.shape)
    print('Image train set shape: %d, %d, %d, %d' % img_valid.shape)
    print('Image test set shape: %d, %d, %d, %d' % img_test.shape)

    data = {'train': (img_train, ceil_train, y_train , ),
            'valid': (img_valid, ceil_valid, y_valid),
            'test': (img_test, ceil_test, y_test),
            'label_encoder': encoder}

    return data


# file_dir: Directorio en el que se encuentran los archivos a cargar y guardar
# model_name: Nombre del modelo usado
# encoder: Objeto usado para transformar la clase a columnas
# normalizer: DataGenerator para normalizar las entradas
# test_data: Datos de evaluacion
# 'encoder' es el LabelBinarizer que transforma la clave en columnas
def save_results(file_dir, model_name, encoder, normalizer, test_data):
    # Cargar y evaluar el mejor modelo
    model = load_model('%s/%s' % (file_dir, '%s_model.h5' % model_name))
    img_test, ceil_test, y_test = test_data

    # Extraer las predicciones del modelo
    img_test = normalizer.standarize(img_test)
    model_test_predictions = model.predict([img_test, ceil_test])
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
