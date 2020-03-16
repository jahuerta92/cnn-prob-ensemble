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
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display, HTML

#cross_validation /train_test_split
method = "train_test_split"

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
    
    kfold = model_selection.KFold(n_splits=10)
    models = []
    models.append(('LogisticRegression', LogisticRegression()))
    models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    models.append(('GaussianNB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('RandomForestClassifier', RandomForestClassifier(n_estimators=500)))
    # evaluate each model
    #results = []
    #names = []
    
    
    results_df = None
    
    le = preprocessing.LabelEncoder()
    le.fit(Y_train)
    
    scoring = 'accuracy'
    if method == "cross_validation":
        results_df = pd.DataFrame(columns=["Classifier", "Mean", "Std"])
    else:
        results_df = pd.DataFrame(columns=["Classifier", "Train_accuracy", "Test_accuracy"])
    for name, model in models:
       
        # print(name)
        msg = None
        if method == "cross_validation":
            kfold = model_selection.KFold(n_splits=10)
            cv_results = model_selection.cross_val_score(model, X_train, le.transform(Y_train), cv=kfold, scoring=scoring)
            #results.append(cv_results)
            #names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            results_df = results_df.append({"Classifier":name, "Mean":cv_results.mean(), "Std":cv_results.std()}, ignore_index=True)
        else:
            model.fit(X_train, Y_train)

            preds_test = model.predict(X_test)
            preds_train = model.predict(X_train)      
            
            acc_train = accuracy_score(preds_train, Y_train)
            acc_test = accuracy_score(preds_test, Y_test)

            msg = "%s: TRAIN: %f TEST: %f" % (name, acc_train, acc_test)
            results_df = results_df.append({"Classifier":name, "Train_accuracy":acc_train, "Test_accuracy":acc_test}, ignore_index=True)
            
        #print(msg)
    display(results_df.round(4))
        
    report_classification_file_name=("{}_classifiers_report_{}.csv".format(output_file_id, experiment_name))

    results_df.round(4).to_csv(os.path.join(output_dir, report_classification_file_name))

    
##############################################################
# Function to train RF with different no. estimators
##############################################################

def grid_search_rf(X_train, Y_train, X_test, Y_test, encoder, parameters_rf, file_best_exec_id, experiment_name, output_dir):
    
    print("Grid search Random Forest")
    rf = RandomForestClassifier()
    
    # THIS PERFORMS CROSS VALIDATION
    # REFIT TRUE, SO THE RF IS RETRAINED ON THE WHOLE ORIGINAL TRAINING DATASET WITH THE BEST PARAMETER CONFIGURAITON FOUND DURING THE CROSS VALIDATION

    grid_rf = GridSearchCV(rf, parameters_rf, verbose=10, n_jobs=10)
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

    preds_test_RF_hot = grid_rf.predict(X_test)
    preds_train_RF_hot = grid_rf.predict(X_train)

    preds_test_RF = encoder.inverse_transform(preds_test_RF_hot)
    preds_train_RF = encoder.inverse_transform(preds_train_RF_hot)
    
    y_test_dec = encoder.inverse_transform(Y_test)

    print(classification_report(y_pred=preds_test_RF, y_true=y_test_dec, digits=3))
    
    generate_confusion_matrix_and_report(y_pred=preds_test_RF, y_test_dec=y_test_dec, 
                                         output_file_id=file_best_exec_id, experiment_name=experiment_name, 
                                         output_dir=output_dir)

    return preds_test_RF_hot, preds_train_RF_hot