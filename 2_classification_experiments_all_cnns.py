# This file is adapted to obtain results of all cnn models in all datasets, instead of best cnn model from all datasets

import os
import pandas as pd
import numpy as np
import re
import pickle
import math
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer, scale
from IPython.display import display, Markdown
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from IPython.display import display, HTML
import warnings
from src.Utils import *

from src.datasets import load_dataset_by_name

from src.plotting import *
warnings.filterwarnings("ignore")


import fire


BATCH_SIZE = 64
N_RUNS = 10
##############################################################
# Directories, files, parameters
#############################################################



def run_experiments(dataset_name):

    data_loaded = load_dataset_by_name(dataset_name=dataset_name, batch_size=BATCH_SIZE)

    PATH = "results/results_" + dataset_name
    plots_dir = os.path.join(PATH, "plots_" + dataset_name)
    features_folder = "features_datasets/"

    PATH_CLASSIFICATION_RESULTS = os.path.join(PATH, "2_classification_experiments_ALL_cnn_execs")

    if not os.path.exists(PATH_CLASSIFICATION_RESULTS):
        os.makedirs(PATH_CLASSIFICATION_RESULTS)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # parameters_rf = {"n_estimators": list(range(100, 1000, 200))}
    parameters_rf = {'n_estimators': [100]}

    ##############################################################
    # Obtaining best CNN execution (in training set)
    ##############################################################

    file_best_exec_id_vgg19, df_results_cnn_vgg19 = get_best_cnn(path=PATH, cnn_filter="vgg19", plots_dir=plots_dir)
    file_best_exec_id_inceptionresnetv2, df_results_cnn_inceptionresnetv2 = get_best_cnn(path=PATH, cnn_filter="inceptionresnetv2", plots_dir=plots_dir)
    file_best_exec_id_inceptionv3, df_results_cnn_inceptionv3 = get_best_cnn(path=PATH, cnn_filter="inceptionv3", plots_dir=plots_dir)
    file_best_exec_id_densenet201, df_results_cnn_densenet201 = get_best_cnn(path=PATH, cnn_filter="densenet201", plots_dir=plots_dir)
    file_best_exec_id_xceptionv1, df_results_cnn_xceptionv1 = get_best_cnn(path=PATH, cnn_filter="xceptionv1", plots_dir=plots_dir)

    print(f'Best executions id: {file_best_exec_id_vgg19:s}')
    print(f'Best executions id: {file_best_exec_id_inceptionresnetv2:s}')
    print(f'Best executions id: {file_best_exec_id_inceptionv3:s}')
    print(f'Best executions id: {file_best_exec_id_densenet201:s}')
    print(f'Best executions id: {file_best_exec_id_xceptionv1:s}')

    ##############################################################
    # General statistics CNN
    ##############################################################

    #generate_cnn_statistics(df_results_cnn=df_results_cnn_vgg19, file_best_exec_id=file_best_exec_id_vgg19, path=PATH, plots_dir=plots_dir, cnn_filter="vgg19")
    #generate_cnn_statistics(df_results_cnn=df_results_cnn_inceptionresnetv2, file_best_exec_id=file_best_exec_id_inceptionresnetv2, path=PATH, plots_dir=plots_dir, cnn_filter="inceptionresnetv2")
    #generate_cnn_statistics(df_results_cnn=df_results_cnn_inceptionv3, file_best_exec_id=file_best_exec_id_inceptionv3, path=PATH, plots_dir=plots_dir, cnn_filter="inceptionv3")
    #generate_cnn_statistics(df_results_cnn=df_results_cnn_densenet201, file_best_exec_id=file_best_exec_id_densenet201, path=PATH, plots_dir=plots_dir, cnn_filter="densenet201")
    #generate_cnn_statistics(df_results_cnn=df_results_cnn_xceptionv1, file_best_exec_id=file_best_exec_id_xceptionv1, path=PATH, plots_dir=plots_dir, cnn_filter="xceptionv1")


    ##############################################################
    # Reading features
    ##############################################################
    print("Reading features...")
    train_features = pd.read_csv(os.path.join(features_folder, dataset_name + "_train_features.csv"))
    del train_features["Unnamed: 0"]
    del train_features["label"]

    valid_features = pd.read_csv(os.path.join(features_folder, dataset_name + "_valid_features.csv"))
    del valid_features["Unnamed: 0"]
    del valid_features["label"]

    test_features = pd.read_csv(os.path.join(features_folder, dataset_name + "_test_features.csv"))
    del test_features["Unnamed: 0"]
    del test_features["label"]



    x_data_train = data_loaded["train"]
    y_data_train = data_loaded["y_train"]

    x_data_valid = data_loaded["validation"]
    y_data_valid = data_loaded["y_valid"]

    x_data_test = data_loaded["test"]
    y_data_test = data_loaded["y_test"]

    size_train = data_loaded["train_size"]
    size_valid = data_loaded["valid_size"]
    size_test = data_loaded["test_size"]



    results_by_cnn_model = {}
    
    print("Obtaining results from CNNs")
    for model_file_id in [file_best_exec_id_vgg19, file_best_exec_id_inceptionresnetv2, file_best_exec_id_inceptionv3, file_best_exec_id_densenet201, file_best_exec_id_xceptionv1]:

        report_classes_exp1, cnn_hot_preds_train_labels, cnn_hot_preds_valid_labels, cnn_hot_preds_test_labels, cnn_hot_preds_train, cnn_hot_preds_valid, cnn_hot_preds_test = get_cnn_predictions_by_model(PATH=PATH,
                                 dataset_name=dataset_name,
                                 data_loaded=data_loaded,
                                 BATCH_SIZE=BATCH_SIZE,
                                 file_best_exec_id=model_file_id)

        results_by_cnn_model[model_file_id] = {"report_classes_exp1": report_classes_exp1,
                                                         "cnn_hot_preds_train_labels:": cnn_hot_preds_train_labels,
                                                         "cnn_hot_preds_valid_labels": cnn_hot_preds_valid_labels,
                                                         "cnn_hot_preds_test_labels": cnn_hot_preds_test_labels,
                                                         "cnn_hot_preds_train": cnn_hot_preds_train,
                                                         "cnn_hot_preds_valid": cnn_hot_preds_valid,
                                                         "cnn_hot_preds_test": cnn_hot_preds_test}


    print("Scaling statistics")
    scaler = MinMaxScaler()
    train_features = pd.DataFrame(scaler.fit_transform(train_features), columns=train_features.columns)

    test_features = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns)

    ##############################################################
    # Experiment 2: classification based on statistical features
    ##############################################################
    
    experiment_name = "EXP_2_statistical_features"
    print(experiment_name)
    (results_exp2, best_model_name_exp2, _, preds_test_best_model_exp2, 
     preds_train_best_model_exp2, cl_report_exp2, matrix_exp2) = train_classifiers_on_set(
        X_train=train_features,
        Y_train=y_data_train,
        X_test=test_features,
        Y_test=y_data_test,
        output_file_id=file_best_exec_id_vgg19.replace("vgg19", ""),
        experiment_name=experiment_name,
        output_dir=PATH_CLASSIFICATION_RESULTS,
        path=PATH,
    use_saved_results=True,)

    dict_experiment_2 = (results_exp2, best_model_name_exp2, _, preds_test_best_model_exp2, 
     preds_train_best_model_exp2, cl_report_exp2, matrix_exp2) 
    
    with open(os.path.join(PATH_CLASSIFICATION_RESULTS, "dict_exp2.pickle"), 'wb') as dbfile_exp2:
        pickle.dump(dict_experiment_2, dbfile_exp2)
        
    
    
    ##############################################################
    # Experiment 3: Average of best classifier over statistical features + CNN probs
    ##############################################################
    experiment_name = "EXP_3_avg_features_CNN_probs"
    print(experiment_name)
    dict_experiment_3 = {}
    for model_file_id in results_by_cnn_model.keys():
        
        preds_features_test_avg_features_CNN = np.argmax(((preds_test_best_model_exp2 + results_by_cnn_model[model_file_id]["cnn_hot_preds_test"]) / 2), axis=1)

        cl_report_dict_avg_features_cnn, conf_matrix_avg_features_cnn = generate_confusion_matrix_and_report(
            y_pred=preds_features_test_avg_features_CNN,
            y_test_dec=y_data_test,
            output_file_id=model_file_id,
            experiment_name=experiment_name,
            output_dir=PATH_CLASSIFICATION_RESULTS,
        )


        report_classes_exp3 = pd.DataFrame(cl_report_dict_avg_features_cnn).transpose().round(5)

        dict_experiment_3[model_file_id] = report_classes_exp3
    
    with open(os.path.join(PATH_CLASSIFICATION_RESULTS, "dict_exp3.pickle"), 'wb') as dbfile_exp3:
        pickle.dump(dict_experiment_3, dbfile_exp3)
    
    ##############################################################
    # Experiment 4: Standard classifiers on CNN predictions + best classifier over estimators
    ##############################################################
    experiment_name = "EXP_4_standard_classifiers_features_CNN"
    print(experiment_name)

    dict_experiment_4 = {}

    for model_file_id in results_by_cnn_model.keys():

        x_train_classifiers_features_cnn_predictions = np.concatenate((preds_train_best_model_exp2, results_by_cnn_model[model_file_id]["cnn_hot_preds_train"]), axis=1)
        x_test_classifiers_features_cnn_predictions = np.concatenate((preds_test_best_model_exp2, results_by_cnn_model[model_file_id]["cnn_hot_preds_test"]), axis=1)

        dict_experiment_4[model_file_id] = []
        for n_run in tqdm(range(N_RUNS)):
            
    
            (results_classifiers_features_cnn,
            best_model_name_exp4, best_model_decod_test_exp4,
            preds_test_best_model_hot_classifiers_features_cnn, 
            preds_train_best_model_hot_classifiers_features_cnns,
            _, _) = train_classifiers_on_set(
                                                                    X_train=x_train_classifiers_features_cnn_predictions,
                                                                    Y_train=y_data_train,
                                                                    X_test=x_test_classifiers_features_cnn_predictions,
                                                                    Y_test=y_data_test,
                                                                    output_file_id=model_file_id,
                                                                    experiment_name=experiment_name,
                                                                    output_dir=PATH_CLASSIFICATION_RESULTS,
                                                                    path=PATH,
                                                                    use_saved_results=False)
            
            #cl_report_dict_exp4 = classification_report(y_pred=best_model_decod_test_exp4, y_true=y_data_test, digits=3, output_dict=True)
            #report_classes_exp4 = pd.DataFrame(cl_report_dict_exp4).transpose().round(4)
            
            dict_experiment_4[model_file_id].append(results_classifiers_features_cnn)
    
    with open(os.path.join(PATH_CLASSIFICATION_RESULTS, "dict_exp4.pickle"), 'wb') as dbfile_exp4:
        pickle.dump(dict_experiment_4, dbfile_exp4)



if __name__ == '__main__':
  fire.Fire(run_experiments)