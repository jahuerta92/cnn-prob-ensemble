import os
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display, HTML
from sklearn.preprocessing import LabelBinarizer, scale
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data_with_indices(file_dir, in_train, in_valid, in_test, feat_file='cloud_features.csv', img_file='images.npz'):

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

    y_train, y_test, y_valid = cloud_encoded[in_train], cloud_encoded[in_test], cloud_encoded[in_valid]

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

##############################################################
# Function to generate confusion matrix and report of results
##############################################################
def generate_confusion_matrix_and_report(y_pred, y_test_dec, output_file_id, experiment_name, output_dir):
    
    matrix_file_name=("{}_confusion_matrix_{}.pdf".format(output_file_id, experiment_name))
    report_file_name_csv=("{}_report_matrix_{}.csv".format(output_file_id, experiment_name))
    report_file_name_latex=("{}_report_matrix_{}.tex".format(output_file_id, experiment_name))

    
    cl_report_dict = classification_report(y_pred= y_pred, y_true= y_test_dec, digits= 3, output_dict=True)
    a = pd.DataFrame(cl_report_dict).transpose().round(2).to_csv(os.path.join(output_dir, report_file_name_csv))
    latex_table = pd.DataFrame(cl_report_dict).transpose().round(2).to_latex()

    with open(os.path.join(output_dir, report_file_name_latex),'w') as tf:
        tf.write(latex_table)
    
    conf_matrix = confusion_matrix(y_true=y_test_dec, y_pred=y_pred)

    fig = plt.figure(figsize=(5.27, 4), dpi=100)
    ax = plt.axes()

    sns.heatmap(conf_matrix, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    ax.xaxis.set_ticklabels(list(set(y_pred)));     ax.yaxis.set_ticklabels(list(set(y_pred)))
   
    fig.savefig(os.path.join(output_dir, matrix_file_name), bbox_inches='tight', pad_inches=0.1)

##############################################################
# Function to compare classifiers on training set
##############################################################
def train_classifiers_on_set(X_train, Y_train, X_test, Y_test, output_file_id, experiment_name, output_dir, encoder):
    
    #kfold = model_selection.KFold(n_splits=10)
    models = []
    models.append(('LogisticRegression', LogisticRegression()))
    models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    models.append(('GaussianNB', GaussianNB()))
    models.append(('SVM-linear', SVC(kernel='linear', probability=True)))
    models.append(('SVM-poly', SVC(kernel='poly', probability=True)))
    models.append(('SVM-rbf', SVC(kernel='rbf', probability=True)))
    models.append(('SVM-sigmoid', SVC(kernel='sigmoid', probability=True)))
    models.append(('RandomForestClassifier', RandomForestClassifier(n_estimators=500)))

    scoring = 'accuracy'
  
    results_df = pd.DataFrame(columns=["Classifier", "Train_accuracy", "Test_accuracy"])
    
    for name, model in models:
      
        model.fit(X_train, Y_train)

        preds_test = model.predict_proba(X_test)
        preds_train = model.predict_proba(X_train)      

        decod_test = encoder.inverse_transform(preds_test)
        decod_train = encoder.inverse_transform(preds_train)

        acc_train = accuracy_score(Y_train, decod_train)
        acc_test = accuracy_score(Y_test, decod_test)

        msg = "%s: TRAIN: %f TEST: %f" % (name, acc_train, acc_test)
        results_df = results_df.append({"Classifier":name, "Train_accuracy":acc_train, "Test_accuracy":acc_test}, ignore_index=True)

    display(results_df.round(4))
        
    report_classification_file_name=("{}_classifiers_report_{}.csv".format(output_file_id, experiment_name))

    results_df.round(4).to_csv(os.path.join(output_dir, report_classification_file_name))

    
##############################################################
# Function to train RF with different no. estimators
##############################################################

def grid_search_rf(X_train, Y_train, X_test, Y_test, encoder, parameters_rf, file_best_exec_id, experiment_name, output_dir):
    
    
    print("Grid search Random Forest")
    rf = RandomForestClassifier(random_state=1, max_features=.35)
    
    # THIS PERFORMS CROSS VALIDATION
    # REFIT TRUE, SO THE RF IS RETRAINED ON THE WHOLE ORIGINAL TRAINING DATASET WITH THE BEST PARAMETER CONFIGURAITON FOUND DURING THE CROSS VALIDATION
    grid_rf = GridSearchCV(rf, parameters_rf, verbose=10, n_jobs=-1)
    grid_rf.fit(X_train, Y_train)
    

    
    pd.DataFrame(grid_rf.cv_results_).to_csv(os.path.join(output_dir, ("{}_rf_train_results_{}.csv".format(file_best_exec_id, experiment_name))))
    pd.DataFrame(grid_rf.best_params_, index=[0]).to_csv(os.path.join(output_dir, ("{}_rf_train_best_params_{}.csv".format(file_best_exec_id, experiment_name))))
    
    print("RF on training set results")
    display(pd.DataFrame(grid_rf.cv_results_))

    ##############################################################
    # Plotting RF comparison no. estimators
    ##############################################################
    
    fig = plt.figure(figsize=(5.27, 4), dpi=100)
    ax = sns.lineplot(x="param_n_estimators", y="mean_test_score", marker="o", data=pd.DataFrame(grid_rf.cv_results_))
    ax.set(xlabel='No. estimators', ylabel='Mean test score')
    fig.savefig(os.path.join(output_dir, ("{}_rf_train_results_no_estimators_{}.pdf".format(file_best_exec_id, experiment_name))), bbox_inches='tight', pad_inches=0.1)
    
    preds_test_RF_hot = grid_rf.predict_proba(X_test)
    preds_train_RF_hot = grid_rf.predict_proba(X_train)

    preds_test_RF = encoder.inverse_transform(preds_test_RF_hot)
    preds_train_RF = encoder.inverse_transform(preds_train_RF_hot)
    
    #y_test_dec = encoder.inverse_transform(Y_test)

    print(classification_report(y_pred=preds_test_RF, y_true=Y_test, digits=4))
    
    generate_confusion_matrix_and_report(y_pred=preds_test_RF, y_test_dec=Y_test, 
                                         output_file_id=file_best_exec_id, experiment_name=experiment_name, 
                                         output_dir=output_dir)

    return preds_test_RF_hot, preds_train_RF_hot