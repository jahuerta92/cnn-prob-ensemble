import os
from sklearn.metrics import classification_report
import pandas as pd
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

##############################################################
# Function to generate confusion matrix and report of results
##############################################################
def generate_confusion_matrix_and_report(y_pred, y_test_dec, output_file_id, experiment_name, output_dir):
    
    matrix_file_name=("{}_confusion_matrix_{}.pdf".format(output_file_id, experiment_name))
    report_file_name_csv=("{}_report_matrix_{}.csv".format(output_file_id, experiment_name))
    report_file_name_latex=("{}_report_matrix_{}.tex".format(output_file_id, experiment_name))

    
    cl_report_dict = classification_report(y_pred= y_pred, y_true= y_test_dec, digits= 3, output_dict=True)
    pd.DataFrame(cl_report_dict).transpose().round(2).to_csv(os.path.join(output_dir, report_file_name_csv))
    latex_table = pd.DataFrame(cl_report_dict).transpose().round(2).to_latex()

    with open(os.path.join(output_dir, report_file_name_latex),'w') as tf:
        tf.write(latex_table)
    
    conf_matrix = confusion_matrix(y_true=y_test_dec, y_pred=y_pred)

    fig = plt.figure(figsize=(8.27, 6), dpi=300)
    ax = plt.axes()

    sns.heatmap(conf_matrix, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    ax.xaxis.set_ticklabels(list(set(y_pred)));     ax.yaxis.set_ticklabels(list(set(y_pred)))
   

    fig.savefig(os.path.join(output_dir, matrix_file_name), bbox_inches='tight', pad_inches=0.1)
    #print(latex_table)

##############################################################
# Function to compare classifiers on training set
##############################################################
def train_classifiers_on_set(X, Y, output_file_id, experiment_name, output_dir):
    
    kfold = model_selection.KFold(n_splits=10)
    models = []
    models.append(('LogisticRegression', LogisticRegression()))
    models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    models.append(('GaussianNB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('RandomForestClassifier', RandomForestClassifier()))
    # evaluate each model
    results = []
    names = []
    scoring = 'accuracy'
    results_df = pd.DataFrame(columns=["Classifier", "Mean", "Std"])
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        results_df = results_df.append({"Classifier":name, "Classifier":cv_results.mean(), "Std":cv_results.std()}, ignore_index=True)
        print(msg)
        
    report_classification_file_name=("{}_classifiers_report_{}.csv".format(output_file_id, experiment_name))

    results_df.round(4).to_csv(os.path.join(output_dir, report_classification_file_name))
