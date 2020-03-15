import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pickle
import copy
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

from imblearn.over_sampling import SMOTE
from imblearn.keras import BalancedBatchGenerator

from tensorflow.keras.models import load_model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from random import seed
from time import time

# file_dir: Directorio en el que se encuentran los archivos feat_file e img_file
# _prop: Proporcion entrenamiento y validacion (test por omision)
# _file: Nombre del fichero
# Devuelve un diccionario con 'train' 'valid' 'test' con tuplas de tipo (imagen,ceilometro,etiqueta)
# 'label_encoder' es el LabelBinarizer que transforma la clave en columnas
def load_data(file_dir, train_prop=.7, valid_prop=.1,
              feat_file='cloud_features.csv', img_file='images.npz',
              oversampler=SMOTE(), random_state=int(time())):

    oversampler.random_state = random_state

    features = pd.read_csv('%s/%s' % (file_dir, feat_file), sep=';', decimal=',')
    cloud_type = np.array(features['cloud.type'])

    ceil_info = np.array(features[["ceil.height0", "ceil.height1",
                                   "ceil.height2", "ceil.depth0",
                                   "ceil.depth1", "ceil.depth2",
                                   "ceil.layers"]])

    del features['date']
    del features['file']
    del features['camnum']
    del features['cloud.type']

    cols = features.columns

    # De string a una matriz de clases para la red
    encoder = LabelBinarizer()
    cloud_encoded = encoder.fit_transform(cloud_type)

    # Escalado de las variables de entrada
    features = pd.DataFrame(scale(features, copy=False), columns=cols)
    ceil_info = scale(ceil_info, copy=False)

    in_train, in_test = train_test_split(np.array(range(0,len(cloud_type))),
                                         train_size=train_prop + valid_prop,
                                         stratify=cloud_type, random_state=random_state)
    in_train, in_valid = train_test_split(in_train, train_size=train_prop / (train_prop + valid_prop),
                                          stratify=cloud_type[in_train],
                                          random_state=random_state)

    y_train, y_test, y_valid = cloud_encoded[in_train], cloud_encoded[in_test], cloud_encoded[in_valid]

    if oversampler is not None:
        print("%s oversampling" % oversampler.__class__.__name__)
        in_train_reshaped = in_train.reshape(-1, 1)
        in_train_reshaped, y_train = oversampler.fit_resample(in_train_reshaped, y_train)
        print(y_train)
        in_train = in_train_reshaped.reshape(-1)
        y_train = cloud_encoded[in_train]

    ceil_train, ceil_test, ceil_valid = ceil_info[in_train], ceil_info[in_test], ceil_info[in_valid]
    aux_train, aux_test, aux_valid = features.iloc[in_train, :], features.iloc[in_test, :], features.iloc[in_valid, :]

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

    data = {'train': (img_train.astype('float32'), ceil_train, y_train),
            'valid': (img_valid.astype('float32'), ceil_valid, y_valid),
            'test': (img_test.astype('float32'), ceil_test, y_test),
            'features': (aux_train, aux_valid, aux_test),
            'label_encoder': encoder,
            'indices': (in_train, in_valid, in_test)}
    



    return data


# Crea un ImageDataGenerator generico a partir de los datos de entrenamiento.
def make_data_generator(train_data):
    img_train, ceil_train, y_train = train_data
    dgen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
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
              max_epochs=1000, batch_size=64, lr=1e-4, n_outputs=1, include_class_weights=True,
              features = None, include_ceilometer=True):
    print("Unpacking train and validation tests")
    img_train, ceil_train, y_train = train_data
    img_valid, ceil_valid, y_valid = valid_data

    if features is not None:
        ceil_train, ceil_valid, _ = features

    # Declarar pesos de clase
    class_weights = None
    if include_class_weights:
        class_weights = class_weight.compute_class_weight('balanced',
                                                          classes=np.unique(y_train),
                                                          y=y_train.reshape(-1,))

    label_num = y_train.shape[1]
    ceil_features = ceil_train.shape[1]
    img_shape = img_train.shape[1:]

    print("Building the network")
    model = model_builder(img_shape, ceil_features, label_num)  # Model([base.input, ceil_input], predictions)

    # Declarar losses y accuracies en caso de que haya mas de una salida
    losses = {'out': 'categorical_crossentropy'}
    accuracies = {'out': 'accuracy'}

    if n_outputs > 1:
        losses.update({'out_%d' % i: 'categorical_crossentropy' for i in range(n_outputs-1)})
        accuracies.update({'out_%d' % i: 'accuracy' for i in range(n_outputs-1)})

    # Compilar el modelo
    print("Compiling the network")
    model.compile(loss=losses,
                  optimizer=Adam(lr=lr, clipnorm=1.),
                  metrics=accuracies)
    print('Layers: %d' % len(model.layers))
    # model.summary()

    # Entrenar el modelo
    seed(1)

    monitored_metric = 'val_acc'
    if n_outputs > 1:
        monitored_metric = 'val_out_acc'

    execution_id = str(time()).replace('.','')

    model_file_name =  '%s_%s_model.h5' % (execution_id, model_name)

    callback_list = [ModelCheckpoint('%s/%s' % (model_dir, model_file_name),
                                     monitor=monitored_metric, save_best_only=True),
                     EarlyStopping(monitor=monitored_metric, min_delta=0.0001, patience=25)]

    x_train = img_train
    x_valid = img_valid

    if include_ceilometer:
        x_train = (img_train, ceil_train)
        x_valid = (img_valid, ceil_valid)

    history = None
    print("Fitting the network %d" % n_outputs )
    if n_outputs > 1:
        # Generar el flow de salidas (No soporta multisalida por defecto DataGenGenerator)
        def multi_output_generator(gen, X, y, bs=batch_size):
            gen_x = gen.flow(X, y, seed=1, batch_size=bs)
            while True:
                x_next, y_next = gen_x.next()
                yield x_next, [y_next[:] for _ in range(n_outputs)]

        history = model.fit_generator(multi_output_generator(data_generator, x_train, y_train),
                            steps_per_epoch=len(img_train) / batch_size,
                            epochs=max_epochs,
                            verbose=1,
                            validation_data=multi_output_generator(data_generator, x_valid, y_valid),
                            validation_steps=len(img_valid) / batch_size,
                            callbacks=callback_list,
                            class_weight=class_weights)

    else:

        history = model.fit_generator(data_generator.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=len(img_train) / batch_size,
                            epochs=max_epochs,
                            verbose=1,
                            validation_data=data_generator.flow(x_valid, y_valid, batch_size=batch_size),
                            validation_steps=len(img_valid) / batch_size,
                            callbacks=callback_list,
                            class_weight=class_weights)
        
    model_history_file_name_pickle =  '%s_%s_history.pickle' % (execution_id, model_name)
    model_history_file_name_json =  '%s_%s_history.json' % (execution_id, model_name)
    
    with open(os.path.join(model_dir, model_history_file_name_pickle), 'wb') as f:
        pickle.dump(history.history, f)
        
    with open(os.path.join(model_dir, model_history_file_name_json), 'w') as f:
        pd.DataFrame(history.history).to_json(f)
        

    return model, model_file_name




def generate_classification_report(file_dir, model_file_name, model, encoder, img_set, ceil_set, y_set, data_generator, include_ceilometer, n_outputs, set_name):


    print("Getting classification report of %s set..." % set_name)

    # Extraer las predicciones del modelo
    standard_img_set = data_generator.standardize(copy.deepcopy(img_set))
    x_set = standard_img_set
    if include_ceilometer:
        x_set = [standard_img_set, ceil_set]

    model_set_predictions = model.predict(x_set)

    if len(model_set_predictions) == n_outputs and len(model_set_predictions) > 1:
        model_set_predictions = model_set_predictions[0]

    decoded_predictions = encoder.inverse_transform(model_set_predictions)
    decoded_observations = encoder.inverse_transform(y_set)
    pred_obs = pd.DataFrame(data={'pred': decoded_predictions, 'obs': decoded_observations})

    # Evaluar el Modelo
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    matrix = confusion_matrix(pred_obs['obs'], pred_obs['pred'])
    names = pred_obs['obs'].unique()

    sns.heatmap(matrix, annot=True, cbar=False, xticklabels=names, yticklabels=names)
    classification_report_matrix = classification_report(pred_obs['obs'], pred_obs['pred'], output_dict=True)
    # print(classification_report_matrix)

    
    classification_dict = {k: classification_report_matrix[k] for k in ('accuracy', 'macro avg', 'weighted avg')}

    summary_dict = {}

    for x in classification_dict.keys():
        if isinstance(classification_dict[x], dict):
            for x2 in list(classification_dict[x].keys()):           
                summary_dict[x.replace(" ", "_") + "-" + x2] = classification_dict[x][x2]
        else:
            summary_dict[x] = classification_dict[x]
        
    # print(str())
                
    with open('%s/%s' % (file_dir, model_file_name.replace("_model.h5", '_%s_summary_acc.csv' % set_name )),'w') as f:
        w = csv.writer(f)
        w.writerow(summary_dict.keys())
        w.writerow(summary_dict.values())

    with open('%s/%s' % (file_dir, ("%s_%s_classification_report.txt" % (set_name, model_file_name.replace("_model.h5", "")))), "w") as outputfile:
        outputfile.write(str(classification_report(pred_obs['obs'], pred_obs['pred'])))

    # Almacenar las predicciones del modelo entrenado
    pred_obs.to_csv('%s/%s' % (file_dir, model_file_name.replace("_model.h5", '_%s_preds.csv' % set_name)))



# file_dir: Directorio en el que se encuentran los archivos a cargar y guardar
# model_name: Nombre del modelo usado
# encoder: Objeto usado para transformar la clase a columnas
# normalizer: DataGenerator para normalizar las entradas
# test_data: Datos de evaluacion
# 'encoder' es el LabelBinarizer que transforma la clave en columnas
def save_results(file_dir, model_file_name, data, data_generator, n_outputs=1, features=None, include_ceilometer=True):
    # Cargar y evaluar el mejor modelo
    model = load_model('%s/%s' % (file_dir, model_file_name))
    
    test_data = data['test']
    train_data = data['train']
    val_data = data['valid']
    indices = data["indices"]
    encoder = data['label_encoder']

    img_test, ceil_test, y_test = test_data
    if features is not None:
        _, _, ceil_test = features
    generate_classification_report(file_dir=file_dir,
                                    model_file_name=model_file_name,
                                    model=model,
                                    encoder=encoder,
                                    img_set=img_test,
                                    ceil_set=ceil_test,
                                    y_set=y_test,
                                    data_generator=data_generator,
                                    include_ceilometer=include_ceilometer,
                                    n_outputs=n_outputs,
                                    set_name="test")

    if train_data is not None:
        img_train, ceil_train, y_train = train_data
        generate_classification_report(file_dir=file_dir,
                                        model_file_name=model_file_name,
                                        model=model,
                                        encoder=encoder,
                                        img_set=img_train,
                                        ceil_set=ceil_train,
                                        y_set=y_train,
                                        data_generator=data_generator,
                                        include_ceilometer=include_ceilometer,
                                        n_outputs=n_outputs,
                                        set_name="train")



    if val_data is not None:
        img_val, ceil_val, y_val = val_data
        generate_classification_report(file_dir=file_dir,
                                model_file_name=model_file_name,
                                model=model,
                                encoder=encoder,
                                img_set=img_val,
                                ceil_set=ceil_val,
                                y_set=y_val,
                                data_generator=data_generator,
                                include_ceilometer=include_ceilometer,
                                n_outputs=n_outputs,
                                set_name="valid")

    # Saving indices
    

    with open('%s/%s' % (file_dir, model_file_name.replace("_model.h5", '_indices.pickle' )), 'wb') as f:
        pickle.dump(data["indices"], f)

    # Use this to read
    #with open('temp.pickle', 'rb') as f:
    #    indices = pickle.load(f)
    

