import pandas as pd
import matplotlib.pyplot as plt
import csv
import pickle
import os
import math
import json
import re
import numpy as np
import seaborn as sns

from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.plotting import generate_bar_plot
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


from tensorflow.keras.models import load_model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from random import seed
from time import time
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def train_model(model_name, data_loaded, batch_size, model_builder,
                freeze_prop, dataset_name, max_epochs=1000, lr=1e-4):

    directory = f"results/results_{dataset_name:s}"

    if not os.path.exists("results"):
        os.makedirs("results")
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    model, model_file_name = fit_model(
        data_train=data_loaded["train"],
        data_validation=data_loaded["validation"],
        train_size=data_loaded["train_size"],
        valid_size=data_loaded["valid_size"],
        img_shape=data_loaded["img_shape"],
        num_labels=data_loaded["num_labels"],
        model_builder=model_builder,
        model_name=model_name,
        batch_size=batch_size,
        freeze_prop=freeze_prop,
        max_epochs=max_epochs,
        lr=1e-4,
        model_dir=directory
    )

    save_results(
        file_dir=directory,
        model_file_name=model_file_name,
        data_train=data_loaded["train"],
        data_valid=data_loaded["validation"],
        data_test=data_loaded["test"],
        y_train=data_loaded["y_train"],
        y_valid=data_loaded["y_valid"],
        y_test=data_loaded["y_test"],
        train_size=data_loaded["train_size"],
        valid_size=data_loaded["valid_size"],
        test_size=data_loaded["test_size"],
        batch_size=batch_size,
        num_labels=data_loaded["num_labels"],
    )


def fit_model(data_train, data_validation, train_size, valid_size, img_shape, num_labels,
              model_builder, model_name, model_dir='./results', verbose=1, 
              max_epochs=1000, batch_size=64, lr=1e-4, include_class_weights=False, freeze_prop=0.0):
    """
    Entrena un modelo construido a partir de un constructor de Model

    :param data_train: this is a first param
    :param data_validation: this is a second param
    :param size_train:
    :param size_valid:
    :param img_shape:
    :param label_num:
    :param model_builder:
    :param model_name:
    :returns: model
    :returns: model file name
    """
    # Declarar pesos de clase
    # class_weights = None
    # if include_class_weights:
    #    class_weights = class_weight.compute_class_weight('balanced',
    #                                                      classes=np.unique(y_train),
    #                                                      y=y_train.reshape(-1,))

    print("Building the network")
    model = model_builder(img_shape, None, num_labels)  # Model([base.input, ceil_input], predictions)

    # Declarar losses y accuracies en caso de que haya mas de una salida
    losses = {'out': 'categorical_crossentropy'}
    accuracies = {'out': 'accuracy'}

    # if n_outputs > 1:
    #    losses.update({'out_%d' % i: 'categorical_crossentropy' for i in range(n_outputs-1)})
    #    accuracies.update({'out_%d' % i: 'accuracy' for i in range(n_outputs-1)})

    # Compilar el modelo
    print("Compiling the network")
    model.compile(loss=losses,
                  optimizer=Adam(lr=lr, clipnorm=1.),
                  metrics=accuracies)
    print('Layers: %d' % len(model.layers))
    # model.summary()

    # Entrenar el modelo
    seed_value = time()

    seed(seed_value)
    # TODO check seed

    monitored_metric = 'val_accuracy'

    execution_id = str(seed_value).replace('.','')

    model_file_name = '%s_%s_model.h5' % (execution_id, model_name)

    callback_list = [ModelCheckpoint('%s/%s' % (model_dir, model_file_name),
                                     monitor=monitored_metric, save_best_only=True),
                     EarlyStopping(monitor=monitored_metric, min_delta=0.0001, patience=25),
                     tf.keras.callbacks.TensorBoard(log_dir=model_dir, profile_batch=0)]

    print("Fitting the network")
    history = model.fit(data_train,
                        steps_per_epoch=math.ceil(train_size/batch_size),
                        epochs=max_epochs,
                        validation_data=data_validation,
                        validation_steps=math.ceil(valid_size/batch_size),
                        callbacks=callback_list,
                        verbose=1)

    model_history_file_name_pickle = '%s_%s_history.pickle' % (execution_id, model_name)
    model_history_file_name_json = '%s_%s_history.json' % (execution_id, model_name)

    info_execution = '%s_%s_info.json' % (execution_id, model_name)

    with open(os.path.join(model_dir, model_history_file_name_pickle), 'wb') as f:
        pickle.dump(history.history, f)

    with open(os.path.join(model_dir, model_history_file_name_json), 'w') as f:
        pd.DataFrame(history.history).to_json(f)

    info_dict = {"model_name": model_name,
                 "max_epochs": max_epochs,
                 "batch_size": batch_size,
                 "lr": lr,
                 "include_class_weights": include_class_weights,
                 "img_shape": str(img_shape),
                 "execution_id": execution_id,
                 "freeze_prop": freeze_prop,
                 "seed": seed_value}
    with open(os.path.join(model_dir, info_execution), "w") as file:
        file.write(json.dumps(info_dict))

    return model, model_file_name


def generate_classification_report(file_dir, model_file_name, model, x_data, y_data, batch_size, size, set_name):

    print("Getting classification report of %s set..." % set_name)

    # Extraer las predicciones del modelo
    model_hot_predictions = model.predict(x_data, steps=math.ceil(size/batch_size))

    decoded_predictions = tf.argmax(model_hot_predictions, axis=1)
    #decoded_observations = tf.argmax(y_data, axis=1)

    pred_obs = pd.DataFrame(data={'pred': decoded_predictions, 'obs': y_data})

    # Evaluar el Modelo
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    matrix = confusion_matrix(pred_obs['obs'], pred_obs['pred'])
    # names = pred_obs['obs'].unique()

    # sns.heatmap(matrix, annot=True, cbar=False, xticklabels=names) # , yticklabels=names)
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
def save_results(file_dir, model_file_name, data_train, data_valid, data_test, 
                 y_train, y_valid, y_test,
                 train_size, valid_size, test_size,
                 batch_size, num_labels=2):
    # Cargar y evaluar el mejor modelo
    model = load_model('%s/%s' % (file_dir, model_file_name))

    generate_classification_report(file_dir=file_dir,
                                   model_file_name=model_file_name,
                                   model=model,
                                   x_data=data_train,
                                   y_data=y_train,
                                   batch_size=batch_size,
                                   size=train_size,
                                   set_name="train")

    generate_classification_report(file_dir=file_dir,
                                   model_file_name=model_file_name,
                                   model=model,
                                   x_data=data_valid,
                                   y_data=y_valid,
                                   batch_size=batch_size,
                                   size=valid_size,
                                   set_name="valid")

    generate_classification_report(file_dir=file_dir,
                                   model_file_name=model_file_name,
                                   model=model,
                                   x_data=data_test,
                                   y_data=y_test,
                                   batch_size=batch_size,
                                   size=test_size,
                                   set_name="test")

    # Saving indices


    #with open('%s/%s' % (file_dir, model_file_name.replace("_model.h5", '_indices.pickle' )), 'wb') as f:
    #    pickle.dump(data["indices"], f)

        
    #with open('%s/%s' % (file_dir, model_file_name.replace("_model.h5", '_encoder.pickle' )), 'wb') as f:
    #    pickle.dump(encoder, f)

    # Use this to read
    #with open('temp.pickle', 'rb') as f:
    #    indices = pickle.load(f)




def get_cnn_predictions_by_model(PATH, dataset_name, data_loaded, BATCH_SIZE, file_best_exec_id):

    

    x_data_train = data_loaded["train"]
    y_data_train = data_loaded["y_train"]

    x_data_valid = data_loaded["validation"]
    y_data_valid = data_loaded["y_valid"]

    x_data_test = data_loaded["test"]
    y_data_test = data_loaded["y_test"]

    size_train = data_loaded["train_size"]
    size_valid = data_loaded["valid_size"]
    size_test = data_loaded["test_size"]

    ##############################################################
    # Loading original tf data
    ##############################################################


    
    
    path_predictions_train = os.path.join(PATH, file_best_exec_id + "_train_predictions.pickle")
    path_predictions_valid = os.path.join(PATH, file_best_exec_id + "_valid_predictions.pickle")
    path_predictions_test = os.path.join(PATH, file_best_exec_id + "_test_predictions.pickle")
    
    if os.path.exists(path_predictions_train) and os.path.exists(path_predictions_valid) and os.path.exists(path_predictions_test):
        print("PREDICTIONS FOUND!!")
        with open(path_predictions_train, 'rb') as handle:
            cnn_hot_preds_train = pickle.load(handle)

        with open(path_predictions_valid, 'rb') as handle:
            cnn_hot_preds_valid = pickle.load(handle)
            
        with open(path_predictions_test, 'rb') as handle:
            cnn_hot_preds_test = pickle.load(handle)

    else:
        model = load_model(os.path.join(PATH, file_best_exec_id + "_model.h5"))
        print("PREDICTIONS NOT FOUND. GENERATING...")
        cnn_hot_preds_train = model.predict(x_data_train, steps=math.ceil(size_train / BATCH_SIZE), verbose=1)
        cnn_hot_preds_valid = model.predict(x_data_valid, steps=math.ceil(size_valid / BATCH_SIZE), verbose=1)
        cnn_hot_preds_test = model.predict(x_data_test, steps=math.ceil(size_test / BATCH_SIZE), verbose=1)
        
        with open(path_predictions_train, 'wb') as handle:
            pickle.dump(cnn_hot_preds_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(path_predictions_valid, 'wb') as handle:
            pickle.dump(cnn_hot_preds_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(path_predictions_test, 'wb') as handle:
            pickle.dump(cnn_hot_preds_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


    cnn_hot_preds_train_labels = cnn_hot_preds_train.argmax(axis=-1)
    cnn_hot_preds_valid_labels = cnn_hot_preds_valid.argmax(axis=-1)
    cnn_hot_preds_test_labels = cnn_hot_preds_test.argmax(axis=-1)

    cl_report_dict_best_cnn = classification_report(y_pred=cnn_hot_preds_test_labels, y_true=y_data_test, digits=3, output_dict=True)
    report_classes_exp1 = pd.DataFrame(cl_report_dict_best_cnn).transpose().round(4)
    
    return report_classes_exp1, cnn_hot_preds_train_labels, cnn_hot_preds_valid_labels, cnn_hot_preds_test_labels, cnn_hot_preds_train, cnn_hot_preds_valid, cnn_hot_preds_test


def generate_confusion_matrix_and_report(y_pred, y_test_dec, 
                                         output_file_id, experiment_name, 
                                         output_dir, generate_plot=False):

    matrix_file_name = ("{}_confusion_matrix_{}.pdf".format(output_file_id, experiment_name))
    report_file_name_latex = ("{}_report_matrix_{}.tex".format(output_file_id, experiment_name))

    cl_report_dict = classification_report(y_pred=y_pred, y_true=y_test_dec, digits=3, output_dict=True)
    latex_table = pd.DataFrame(cl_report_dict).transpose().round(2).to_latex()

    with open(os.path.join(output_dir, report_file_name_latex),'w') as tf:
        tf.write(latex_table)

    conf_matrix = confusion_matrix(y_true=y_test_dec, y_pred=y_pred)

    if generate_plot:
        fig = plt.figure(figsize=(8.27, 6), dpi=100)
        ax = plt.axes()
        sns.heatmap(conf_matrix, annot=True, ax=ax, fmt='g', cmap="Blues")  #annot=True to annotate cells
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        ax.xaxis.set_ticklabels(list(set(y_pred)))
        ax.yaxis.set_ticklabels(list(set(y_pred)))
        fig.savefig(os.path.join(output_dir, matrix_file_name), bbox_inches='tight',
                    pad_inches=0.1)

    return cl_report_dict, conf_matrix


##############################################################
# Function to compare classifiers on training set
##############################################################
def train_classifiers_on_set(X_train, Y_train, X_test, Y_test, output_file_id,
                             experiment_name, output_dir, path, use_saved_results=True, classifiers_used=None):
    
    #n_estimators_bagging = 10
    n_jobs = 14
    models = []
    
    models_dict = {
        "LogisticRegression": LogisticRegression(),
        "LinearDiscriminantAnalysis" : LinearDiscriminantAnalysis(),
        "KNeighborsClassifier": KNeighborsClassifier(n_jobs=n_jobs),
        #"SVM-linear": SVC(kernel='linear', probability=True),
        "SVM-rbf": SVC(kernel='rbf', probability=True, random_state=int(time())),
        "SVM-sigmoid": SVC(kernel='sigmoid', probability=True, random_state=int(time())),
        #"SVM-linear": LinearSVC(),
        #"SVM-linear": OneVsRestClassifier(BaggingClassifier(SVC(kernel='poly', probability=True, class_weight=None), max_samples=0.03, n_estimators=n_estimators_bagging, n_jobs=n_jobs)),
        #"SVM-rbf":  OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', probability=True, class_weight=None), max_samples=0.1, n_estimators=n_estimators_bagging, n_jobs=n_jobs)),
        #"SVM-sigmoid": OneVsRestClassifier(BaggingClassifier(SVC(kernel='sigmoid', probability=True, class_weight=None), max_samples=0.5, n_estimators=n_estimators_bagging, n_jobs=n_jobs)),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=int(time()))
    }
    
    if classifiers_used is None:
        models = [(key, models_dict[key]) for key in models_dict.keys()]
    else:
        models = [(key, models_dict[key]) for key in models_dict.keys() if classifiers_used[key]]
    
    
    
    #if classifiers_used is None or classifiers_used["LogisticRegression"]:
    #    models.append(('LogisticRegression', LogisticRegression()))
    #if classifiers_used is None or classifiers_used["LinearDiscriminantAnalysis"]:
    #    models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
    #if classifiers_used is None or classifiers_used["KNeighborsClassifier"]:
    #    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
    #if classifiers_used is None or classifiers_used["SVM-linear"]:
    #    models.append(('SVM-linear', SVC(kernel='linear', probability=True)))
    #if classifiers_used is None or classifiers_used["SVM-rbf"]:
    #    models.append(('SVM-rbf', SVC(kernel='rbf', probability=True)))
        
    #if classifiers_used is None or classifiers_used["SVM-sigmoid"]:
    #    models.append(('SVM-sigmoid', SVC(kernel='sigmoid', probability=True)))
    #if classifiers_used is None or classifiers_used["RandomForestClassifier"]:
    #    models.append(('RandomForestClassifier', RandomForestClassifier(n_estimators=500)))
        
        
        #models.append(('SVM-poly', SVC(kernel='poly', probability=True)))
    
    #models.append(('SVM-poly-Bagging', OneVsRestClassifier(BaggingClassifier(SVC(kernel='poly', probability=True, class_weight=None), max_samples=0.2, n_estimators=n_estimators_bagging, n_jobs=-1))))
     #models.append(('SVM-rbf-Bagging', OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', probability=True, class_weight=None), max_samples=0.2, n_estimators=n_estimators_bagging, n_jobs=-1))))
    #models.append(('SVM-sigmoid-Bagging', OneVsRestClassifier(BaggingClassifier(SVC(kernel='sigmoid', probability=True, class_weight=None), max_samples=0.2, n_estimators=n_estimators_bagging, n_jobs=-1))))
    #     models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
#     models.append(('GaussianNB', GaussianNB()))


    #if not os.path.exists(path + "/2_classification_experiments/"):
    #    os.makedirs(path + "/2_classification_experiments/")



    results_df_file_name = os.path.join(path, "results_classifiers_" + experiment_name + ".pickle")
    best_model_file_name = os.path.join(path, "best_model_classifiers_" + experiment_name + ".pickle")
    
    best_macro_avg_weighted_precision_test = 0.0
    best_model = None
    if not use_saved_results or not os.path.exists(results_df_file_name):
        
        #print("Training classifiers " + experiment_name)
        #print("PATH" + results_df_file_name)

        results_df = pd.DataFrame(columns=["Classifier",
                                           "Train_accuracy",
                                           "Test_accuracy",
                                           "Train_macro_avg_precision",
                                           "Test_macro_avg_precision",
                                           "Train_macro_avg_weighted_precision",
                                           "Test_macro_weighted_avg_precision", ])


        for name, model in models:

            # print(f'Running training for training {name:s}')

        
            model.fit(X_train, Y_train)

            preds_test = model.predict(X_test)
            preds_train = model.predict(X_train)

            acc_train = accuracy_score(Y_train, preds_train)
            acc_test = accuracy_score(Y_test, preds_test)

            macro_avg_precision_train = precision_score(Y_train, preds_train, average='macro')
            macro_avg_precision_test = precision_score(Y_test, preds_test, average='macro')

            macro_avg_weighted_precision_train = precision_score(Y_train, preds_train, average='weighted')
            macro_avg_weighted_precision_test = precision_score(Y_test, preds_test, average='weighted')

            if best_macro_avg_weighted_precision_test < macro_avg_weighted_precision_test:
                best_macro_avg_weighted_precision_test = macro_avg_weighted_precision_test
                best_model = model


            # msg = "%s: TRAIN: %f TEST: %f" % (name, acc_train, acc_test)
            results_df = results_df.append({"Classifier": name,
                                            "Train_accuracy": acc_train,
                                            "Test_accuracy": acc_test,
                                            "Preds_train": preds_train,
                                            "Preds_test": preds_test,
                                            "Train_macro_avg_precision": macro_avg_precision_train,
                                            "Test_macro_avg_precision": macro_avg_precision_test,
                                            "Train_macro_avg_weighted_precision": macro_avg_weighted_precision_train,
                                            "Test_macro_weighted_avg_precision": macro_avg_weighted_precision_test},
                                           ignore_index=True)

        # Saving results
        dbfile_results = open(results_df_file_name, 'ab')
        pickle.dump(results_df, dbfile_results)
        dbfile_results.close()

        dbfile_model = open(best_model_file_name, 'ab')
        pickle.dump(best_model, dbfile_model)
        dbfile_model.close()
        
    else:

        print("Results taken from " + results_df_file_name)
        dbfile_results = open(results_df_file_name, 'rb')
        results_df = pickle.load(dbfile_results)
        dbfile_results.close()

        dbfile_model = open(best_model_file_name, 'rb')
        best_model = pickle.load(dbfile_model)
        dbfile_model.close()

    best_row = results_df.iloc[results_df['Test_macro_weighted_avg_precision'].idxmax()]

    best_macro_avg_weighted_precision_test = best_row.Test_macro_weighted_avg_precision
    
    #print("RESULT: " + str(best_macro_avg_weighted_precision_test))
    best_model_name = best_row.Classifier
    best_model_decod_test = best_row.Preds_test



    #display(results_df.round(4))

    preds_test_best_model_hot = best_model.predict_proba(X_test)
    preds_train_best_model_hot = best_model.predict_proba(X_train)

    #report_classification_file_name = "{}_classifiers_report_{}.csv".format(output_file_id, experiment_name)

    #table_classification_latex_file_name = "{}_classifiers_latex_{}.tex".format(output_file_id, experiment_name)

    #results_df.round(4).to_csv(os.path.join(output_dir, report_classification_file_name))
    #results_df.round(4).to_latex(os.path.join(output_dir, table_classification_latex_file_name), index=False, bold_rows=True)

#     cl_report_dict, conf_matrix = generate_confusion_matrix_and_report(y_pred=preds_test_best_model, y_test_dec=Y_test, 
#                                          output_file_id=file_best_exec_id, experiment_name=experiment_name, 
#                                          output_dir=output_dir)
    cl_report_dict, conf_matrix = None, None

    return results_df, best_model_name, best_model_decod_test, preds_test_best_model_hot, preds_train_best_model_hot, cl_report_dict, conf_matrix


def train_rf(X_train, Y_train, X_test, Y_test, parameters_rf, file_best_exec_id, experiment_name, output_dir):

    print("TRAINING Random Forest")
    rf = RandomForestClassifier(random_state=1, max_features=.35, n_estimators=parameters_rf["n_estimators"][0])

    rf.fit(X_train, Y_train)

    preds_test_RF_hot = rf.predict_proba(X_test)
    preds_train_RF_hot = rf.predict_proba(X_train)

    preds_test_RF = rf.predict(X_test)
    preds_train_RF = rf.predict(X_train)

    print(classification_report(y_pred=preds_test_RF, y_true=Y_test, digits=4))

    cl_report_dict, conf_matrix = generate_confusion_matrix_and_report(y_pred=preds_test_RF, y_test_dec=Y_test, 
                                         output_file_id=file_best_exec_id, experiment_name=experiment_name, 
                                         output_dir=output_dir)

    return preds_test_RF_hot, preds_train_RF_hot, None, None, cl_report_dict, conf_matrix


##############################################################
# Function to train RF with different no. estimators
##############################################################
def grid_search_rf(X_train, Y_train, X_test, Y_test, parameters_rf, file_best_exec_id, experiment_name, output_dir):

    print("Grid search Random Forest")
    rf = RandomForestClassifier(random_state=1, max_features=.35)

    # THIS PERFORMS CROSS VALIDATION
    # REFIT TRUE, SO THE RF IS RETRAINED ON THE WHOLE ORIGINAL TRAINING DATASET WITH THE BEST PARAMETER CONFIGURAITON FOUND DURING THE CROSS VALIDATION
    grid_rf = GridSearchCV(rf, parameters_rf, verbose=10, n_jobs=-1, scoring='precision_weighted')
    grid_rf.fit(X_train, Y_train)

    pd.DataFrame(grid_rf.cv_results_).to_csv(os.path.join(output_dir, ("{}_rf_train_results_{}.csv".format(file_best_exec_id, experiment_name))))
    pd.DataFrame(grid_rf.best_params_, index=[0]).to_csv(os.path.join(output_dir, ("{}_rf_train_best_params_{}.csv".format(file_best_exec_id, experiment_name))))

    print("RF on training set results")
    #display(pd.DataFrame(grid_rf.cv_results_))

    ##############################################################
    # Plotting RF comparison no. estimators
    ##############################################################

    fig = plt.figure(figsize=(7.27, 4), dpi=100)
    ax = sns.lineplot(x="param_n_estimators", y="mean_test_score", marker="o", data=pd.DataFrame(grid_rf.cv_results_))
    ax.set(xlabel='No. estimators', ylabel='Mean weighted precision train set\n Cross validation')
    ax.set(xticks=np.asarray(parameters_rf["n_estimators"]), xticklabels=np.asarray(parameters_rf["n_estimators"]))
    ax.set_xticklabels(np.asarray(parameters_rf["n_estimators"]), rotation=45, horizontalalignment='right')
    # fig.savefig(os.path.join("plots/", ("{}_rf_train_results_no_estimators_{}.pdf".format(file_best_exec_id, experiment_name))), bbox_inches='tight', pad_inches=0.1)

    fig.savefig(os.path.join(output_dir, ("{}_rf_train_results_no_estimators_{}.pdf".format(file_best_exec_id, experiment_name))), bbox_inches='tight', pad_inches=0.1)

    preds_test_RF_hot = grid_rf.predict_proba(X_test)
    preds_train_RF_hot = grid_rf.predict_proba(X_train)

    preds_test_RF = grid_rf.predict(X_test)
    preds_train_RF = grid_rf.predict(X_train)

    print(classification_report(y_pred=preds_test_RF, y_true=Y_test, digits=4))

    cl_report_dict, conf_matrix = generate_confusion_matrix_and_report(y_pred=preds_test_RF, y_test_dec=Y_test, 
                                         output_file_id=file_best_exec_id, experiment_name=experiment_name, 
                                         output_dir=output_dir)

    return preds_test_RF_hot, preds_train_RF_hot, grid_rf.cv_results_, grid_rf.best_params_, cl_report_dict, conf_matrix


def get_best_cnn(path, plots_dir, cnn_filter="", dataset_filter=None):

    summary_files = [each for each in os.listdir(path) if each.endswith("_summary_acc.csv")]
    
    # Generating dataframe with all summary files. New columns with file name, model name and set name (train,valid,test)
    df_results_cnn = pd.concat((pd.read_csv(os.path.join(path, f)).assign(file=f).
                    assign(model=re.search("[0-9]_+(.+?)_summary_acc.csv", f).group(1).split("_")[0]).
                    assign(set=re.search("[0-9]_+(.+?)_summary_acc.csv", f.replace("no_ceil_", "")).
                           group(1).split("_")[1])for f in summary_files))
    if cnn_filter != "":
        df_results_cnn = df_results_cnn.loc[df_results_cnn['model']==cnn_filter]
    
    # Extracting train results to select best execution
    #df_train = df_results_cnn[df_results_cnn["set"] == "train"]
    #best_exec_acc_train = df_train[df_train["macro_avg-precision"] == max(df_train["macro_avg-precision"])]

    #print(df_results_cnn)
    # Extracting train results to select best execution
    df_validation = df_results_cnn[df_results_cnn["set"] == "valid"]
    best_exec_acc_validation = df_validation[df_validation["macro_avg-f1-score"] == max(df_validation["macro_avg-f1-score"])]

    
    if isinstance(best_exec_acc_validation["file"][0], str):
        file_best_exec = best_exec_acc_validation["file"][0]
    else:
        file_best_exec = best_exec_acc_validation["file"][0].head(1)[0]

    file_best_exec_id = re.search("(.+?)_[a-z]+_summary_acc.csv", file_best_exec).group(1)

    ##############################################################
    # General statistics CNN
    ##############################################################

    df_results_cnn["execution"] = df_results_cnn.groupby(["model", "set"]).cumcount()
    df_results_cnn["execution"].astype("int32")
    df_melt = pd.melt(df_results_cnn[df_results_cnn["set"] == "test"], id_vars=["model", "execution"], value_vars=["accuracy"])

    #generate_bar_plot(data=df_melt, x="execution", y="value", hue="model", path=plots_dir, file_name="1_cnn_comparison_barplot.pdf",
    #                 xlabel="Execution", ylabel="Test accuracy")

    df_results_cnn["set"] = df_results_cnn["set"].map({"test": "Test", "train": "Training", "valid": "Valid"})

    #generate_bar_plot(data=df_results_cnn, x="model", y="accuracy", hue="set", path=plots_dir, file_name="1_cnn_comparison_set_barplot.pdf",
    #                 xlabel="Execution", ylabel="Accuracy")

    test_summary_data_cnn = pd.read_csv(os.path.join(path, f"{file_best_exec_id:s}_test_summary_acc.csv" ), index_col=0)

    # print(f'Best CNN model found: {file_best_exec_id:s}')

    #display(test_summary_data_cnn)

    latex_table_summary = pd.DataFrame(test_summary_data_cnn).transpose().round(2).to_latex(os.path.join(plots_dir, "2_cnn_metrics_table" + cnn_filter + ".tex"))

    statistics_cnn = df_results_cnn[["accuracy", "macro_avg-precision", "model", "set"]].groupby(["model", "set"]).agg(["mean", "std"])

    statistics_cnn.round(4).to_latex(os.path.join(plots_dir, "3_cnn_statistics" + cnn_filter + ".tex"))

    #display(statistics_cnn.round(4) * 100)
    
    #print(os.path.join(path, f'{cnn_filter}_all_runs_results.csv'))
    df_results_cnn.to_csv(os.path.join(path, f'{cnn_filter:s}_all_runs_results.csv'))


    return file_best_exec_id, df_results_cnn


def generate_cnn_statistics(df_results_cnn, file_best_exec_id, path, plots_dir, cnn_filter="", show_tables=True):
    

    df_results_cnn["execution"] = df_results_cnn.groupby(["model", "set"]).cumcount()
    df_results_cnn["execution"].astype("int32")
    df_melt = pd.melt(df_results_cnn[df_results_cnn["set"] == "Test"], id_vars=["model", "execution"], value_vars=["accuracy"])
    
    # Generating bar plots
    comparison_barplot_name = "1_cnn_comparison_barplot"
    generate_bar_plot(data=df_melt, x="execution", y="value", hue="model", path=plots_dir, file_name=comparison_barplot_name + cnn_filter + ".pdf", xlabel="Execution", ylabel="Test accuracy")
    
    comparison_set_barplot_name = "1_cnn_comparison_set_barplot"
    generate_bar_plot(data=df_results_cnn, x="model", y="accuracy", hue="set", path=plots_dir, file_name=comparison_set_barplot_name + cnn_filter + ".pdf", xlabel="Execution", ylabel="Accuracy")
    
    # print("++ OUTPUT: Comparison barplot generated at: " + plots_dir + comparison_barplot_name)
    # print("++ OUTPUT: Comparison barplot generated at: " + plots_dir + comparison_set_barplot_name)
    
    ################################################################
    ################################################################
    
    test_summary_data_cnn = pd.read_csv(os.path.join(path, f"{file_best_exec_id:s}_test_summary_acc.csv" ), index_col=0)
    print(f'Best CNN model found: {file_best_exec_id:s}')
    
    ################################################################
    ################################################################

    if show_tables:
        display(test_summary_data_cnn)
    
    latex_table_summary_path = os.path.join(plots_dir, "2_cnn_metrics_table" + cnn_filter + ".tex")
    latex_table_summary = pd.DataFrame(test_summary_data_cnn).transpose().round(2).to_latex(latex_table_summary_path)
    # print("++ OUTPUT: Latex table summary: " + latex_table_summary_path)
    
    ################################################################
    ################################################################
    
    
    statistics_cnn_path = os.path.join(plots_dir, "3_cnn_statistics" + cnn_filter)
    statistics_cnn = df_results_cnn[["accuracy", "macro_avg-precision", "model", "set"]].groupby(["model", "set"]).agg(["mean", "std", "max"])
    statistics_cnn.round(4).to_latex(statistics_cnn_path + ".tex")
    statistics_cnn.round(4).to_csv(statistics_cnn_path + ".csv")
    
    # print("++ OUTPUT: Statistics CNN path: " + statistics_cnn_path + ".tex")
    # print("++ OUTPUT: Statistics CNN path: " + statistics_cnn_path + ".csv")
    
    if show_tables:
        display(statistics_cnn.round(4) * 100)
    
    return statistics_cnn
    