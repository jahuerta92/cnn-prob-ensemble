import os
import re
import math
import pickle
import warnings


from tqdm import tqdm 
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


from src.plotting import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from IPython.display import display, HTML, Markdown

from src.Utils import *
from src.datasets import load_dataset_by_name
import fire


warnings.filterwarnings("ignore")

##############################################################
# Directories, files, parameters
#############################################################

#############################################################
BATCH_SIZE = 64
N_RUNS = 10



def run_experiments(dataset_name, cnn_model_selected):


    # Loading dataset
    data_loaded = load_dataset_by_name(dataset_name=dataset_name, batch_size=BATCH_SIZE)

    CNN_RESULTS_PATH = "results/results_" + dataset_name
    
    PATH_CLASSIFICATION_RESULTS = os.path.join(CNN_RESULTS_PATH, "2_classification_experiments_ALL_cnn_execs")

    ABLATION_STUDY_PATH = "ablation_study_stat_out/results_" + dataset_name

    if not os.path.exists(ABLATION_STUDY_PATH):
        os.makedirs(ABLATION_STUDY_PATH)

    output_plots_dir = os.path.join(ABLATION_STUDY_PATH, "plots_" + dataset_name)
    features_folder = "features_datasets/"

    if not os.path.exists(output_plots_dir):
        os.makedirs(output_plots_dir)

    parameters_rf = {'n_estimators': [500]}

    ##############################################################
    # Obtaining best CNN execution (in training set) for each CNN model
    ##############################################################

    file_best_exec_id, df_results_cnn = get_best_cnn(path=CNN_RESULTS_PATH, cnn_filter=cnn_model_selected, plots_dir=output_plots_dir)

    ##############################################################
    # Reading features
    ##############################################################

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


    scaler = MinMaxScaler()

    train_features = pd.DataFrame(scaler.fit_transform(train_features), columns=train_features.columns)
    valid_features = pd.DataFrame(scaler.transform(valid_features), columns=valid_features.columns)
    test_features = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns)

    results_by_cnn_model = {}

    # tqdm([file_best_exec_id_vgg19, file_best_exec_id_inceptionresnetv2, file_best_exec_id_inceptionv3, file_best_exec_id_densenet201, file_best_exec_id_xceptionv1]):

    print("RUNNING ALL EXPERIMENTS FOR ABLATION WITH MODEL: " + file_best_exec_id)
    report_classes_exp1, cnn_hot_preds_train_labels, cnn_hot_preds_valid_labels, cnn_hot_preds_test_labels, cnn_hot_preds_train, cnn_hot_preds_valid, cnn_hot_preds_test = get_cnn_predictions_by_model(PATH=CNN_RESULTS_PATH,
                             dataset_name=dataset_name,
                             data_loaded=data_loaded,
                             BATCH_SIZE=BATCH_SIZE,
                             file_best_exec_id=file_best_exec_id)

    results_by_cnn_model[file_best_exec_id] = {"report_classes_exp1": report_classes_exp1,
                                                     "cnn_hot_preds_train_labels:": cnn_hot_preds_train_labels,
                                                     "cnn_hot_preds_valid_labels": cnn_hot_preds_valid_labels,
                                                     "cnn_hot_preds_test_labels": cnn_hot_preds_test_labels,
                                                     "cnn_hot_preds_train": cnn_hot_preds_train,
                                                     "cnn_hot_preds_valid": cnn_hot_preds_valid,
                                                     "cnn_hot_preds_test": cnn_hot_preds_test}

    # Only saving results of the cnn model used for ablation
    with open(os.path.join(ABLATION_STUDY_PATH, f'results_{file_best_exec_id}_cnn_model_{dataset_name}.pickle'), 'wb') as handle:
        pickle.dump(results_by_cnn_model, handle, protocol=pickle.HIGHEST_PROTOCOL)


    features_groups = [['green_mean-tex', 'red_mean-tex', 'blue_mean-tex'],
                       ['green_contrast', 'red_contrast', 'blue_contrast'],
                       ['green_std', 'red_std', 'blue_std'],
                       ['green_skew', 'red_skew', 'blue_skew'],
                       ['green_ASM', 'red_ASM', 'blue_ASM'],
                       ['green_correlation', 'red_correlation', 'blue_correlation'],
                       ['green_homogeneity', 'red_homogeneity', 'blue_homogeneity'],
                       ['green_hist-high', 'red_hist-high', 'blue_hist-high'],
                       ['green_hist-medium-low', 'red_hist-medium-low', 'blue_hist-medium-low'],
                       ['green_hist-medium', 'red_hist-medium', 'blue_hist-medium'],
                       ['green_mean', 'red_mean', 'blue_mean'],
                       ['green_variance', 'red_variance', 'blue_variance'],
                       ['green_hist-medium-high', 'red_hist-medium-high', 'blue_hist-medium-high'],
                       ['green_dissimilarity', 'red_dissimilarity', 'blue_dissimilarity'],
                       ['green_hist-low', 'red_hist-low', 'blue_hist-low'],
                       ['green_entropy', 'red_entropy', 'blue_entropy'],
                       ['red-green_difference', 'red-blue_difference', 'green-blue_difference'],
                       ['red-green_ratio', 'red-blue_ratio', 'green-blue_ratio']]

    features_groups_dict = {features_groups[x][0].split("_")[1]: features_groups[x] for x in range(len(features_groups))}

    ##############################################################
    # Ablation individual statistics
    ##############################################################

    # Remember to fix model to vgg19 for all experiments

    experiment_name = "ablation_study_stat_out"
    print("EXPERIMENT %s" % experiment_name)


    results_stats_classification = {}
    results_ablation = {}
    
    
    # RUN BEFORE 3_CLASSIFICATION_EXPERIMENTS_RESULTS_COLLECTION IN ORDER TO OBTAIN THE BEST CLASSIFIER FOR EACH CNN AND DATASET IN ORDER TO RECEIVE THE PERFORM ABLATION STUDY WITH THE CORRESPONDING COMBINATION
    with open("best_ensemble.pickle", "rb") as handle:
        best_ensemble = pickle.load(handle)

    for group in features_groups_dict.keys():
        
        print("GROUP: " + group)
        group_stats = features_groups_dict[group]
        
        best_classifiers = None
        
        # Classification based on statistical features
        print("1 Classification of statistical features")
        
        # Loading results of classification of statistical features
    
        with open(os.path.join(PATH_CLASSIFICATION_RESULTS, "dict_exp2.pickle"), 'rb') as dbfile_exp2:
            (results_stats, _, _, preds_test_best_model_stats, preds_train_best_model_stats, _, _)  = pickle.load(dbfile_exp2)

            
        results_stats_classification[group] = results_stats
        
        print("2 Ensemble")
        if best_ensemble[(dataset_name, cnn_model_selected)] == "Avg.":
            
            preds_features_test_avg_features_CNN = np.argmax(((preds_test_best_model_stats + results_by_cnn_model[file_best_exec_id]["cnn_hot_preds_test"]) / 2), axis=1)
            
            macro_avg_weighted_precision_test = precision_score(y_data_test, preds_features_test_avg_features_CNN, average='weighted')
            results_df = pd.DataFrame(columns=["Classifier",
                                   "Test_macro_weighted_avg_precision", ])
            results_ensemble = results_df.append({"Classifier": "Avg.",
                                "Test_macro_weighted_avg_precision": macro_avg_weighted_precision_test},
                               ignore_index=True)

            
        
        else:
            
            x_train_classifiers_features_cnn_predictions = np.concatenate((preds_train_best_model_stats, results_by_cnn_model[file_best_exec_id]["cnn_hot_preds_train"]), axis=1)
            x_test_classifiers_features_cnn_predictions = np.concatenate((preds_test_best_model_stats, results_by_cnn_model[file_best_exec_id]["cnn_hot_preds_test"]), axis=1)

            best_classifier_ensemble = best_ensemble[(dataset_name, cnn_model_selected)]

            classifiers_used = {"LogisticRegression":False, 
                              "LinearDiscriminantAnalysis":False,
                              "KNeighborsClassifier":False,
                              "SVM-linear":False,
                              "SVM-rbf":False,
                              "SVM-sigmoid":False,
                              "RandomForestClassifier":False}

            classifiers_used[best_classifier_ensemble] = True

            

            # Classification based on ensemble (group of statistics + cnn probabilities)
            results_ensemble = []
            for n_run in tqdm(range(N_RUNS)):
                (results_ensemble_run, _, _, _, _, _, _) = train_classifiers_on_set(
                                                                        X_train=x_train_classifiers_features_cnn_predictions,
                                                                        Y_train=y_data_train,
                                                                        X_test=x_test_classifiers_features_cnn_predictions,
                                                                        Y_test=y_data_test,
                                                                        output_file_id=experiment_name,
                                                                        experiment_name=experiment_name,
                                                                        output_dir=output_plots_dir,
                                                                        path=ABLATION_STUDY_PATH,
                                                                        use_saved_results=False,
                                                                        classifiers_used=classifiers_used)

                results_ensemble.append(results_ensemble_run)
        results_ablation[group] = results_ensemble
        print("\n")


    with open(os.path.join(ABLATION_STUDY_PATH, f'results_ablation_stat_out_{dataset_name}_{file_best_exec_id}.pickle'), 'wb') as handle:
        pickle.dump(results_ablation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(ABLATION_STUDY_PATH, f'results_stats_classification_stat_out_{dataset_name}_{file_best_exec_id}.pickle'), 'wb') as handle:
        pickle.dump(results_stats_classification, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        
        
   
if __name__ == '__main__':
  fire.Fire(run_experiments)