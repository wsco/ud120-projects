#!/usr/bin/python

import sys
import math
sys.path.append("../tools/")
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tester import dump_classifier_and_data, test_classifier
from feature_format import featureFormat, targetFeatureSplit

###Examine Data
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

available_features=[]

for people, k in data_dict.iteritems():
    for key, value in k.iteritems():
        if key not in available_features:
            available_features.append(key)
print 'Number of employees in dataset:', len(data_dict),'\n'
print 'Number of Features in dataset:',len(available_features)\
,'\ncomprised of:\n', available_features,'\n'

print 'POI''s: ', sum([1 for p in data_dict if data_dict[p]['poi']==True]), \
'NonPOI''s:', sum([1 for p in data_dict if data_dict[p]['poi']==False]), '\n'

def count_unique_and_null(dictionary):
    feature_summary = []
    for feature in [f for f in available_features if f!='poi']:
        entry_unique= []
        nan_count = 0
        for entry in dictionary:
            if dictionary[entry][feature] == 'NaN':
                nan_count += 1
            elif dictionary[entry][feature] not in entry_unique and dictionary[entry][feature] !='NaN' :
                entry_unique.append(entry)
        feature_summary.append({'feature':feature
                                ,'unique count':len(entry_unique)
                                ,'nans': nan_count})
    return feature_summary

df = pd.DataFrame(count_unique_and_null(data_dict))
print 'NaN / Unique Values'
print df.sort_values(by = 'nans', ascending = False)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = [ 'poi','salary','to_messages','total_payments','exercised_stock_options'
                 ,'bonus','restricted_stock','shared_receipt_with_poi','total_stock_value'
                 ,'expenses','from_messages','other','from_this_person_to_poi'
                 ,'long_term_incentive','from_poi_to_this_person','ratio_from_poi'
                 ,'ratio_to_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

#find employees with nan for all data
all_nan = []
for person, features in data_dict.items():
    notNaN = False
    for feature, val in features.items():
        if feature != 'poi':
            if val != 'NaN':
                notNaN = True
                break
    if not notNaN:
        all_nan.append(person)
print 'Users with all ''NaN'' data', all_nan

#Remove people identified above in code and visual inspection of employee names
for person in ['TOTAL','THE TRAVEL AGENCY IN THE PARK','LOCKHART EUGENE E']:
    data_dict.pop(person)
    

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

def computeRatio( poi_messages, all_messages ):
    if not math.isnan(float(poi_messages)) and not math.isnan(float(all_messages)):
        fraction = round(float(poi_messages)/float(all_messages),3)
    else:
        fraction =0
    return fraction

for person in data_dict:
    from_poi = data_dict[person]['from_poi_to_this_person']
    total_inbox = data_dict[person]['to_messages']
    to_poi = data_dict[person]['from_this_person_to_poi']
    total_outbox = data_dict[person]['from_messages']
    
    data_dict[person]['ratio_from_poi'] = computeRatio( from_poi, total_inbox )
    data_dict[person]['ratio_to_poi']  = computeRatio( to_poi, total_outbox )

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

#Examine new features
print 'max to poi ratio: ', max([data_dict[p]['ratio_to_poi'] for p in data_dict])
print 'max from poi ratio: ', max([data_dict[p]['ratio_from_poi'] for p in data_dict])
print 'Person with 100% emails sent to POI',[p for p in data_dict if data_dict[p]['ratio_to_poi']==1.]

ratiosdf = pd.DataFrame(data)
ratiosdf.columns = features_list

plt.scatter(ratiosdf.ratio_to_poi, ratiosdf.ratio_from_poi
            , s=ratiosdf.from_this_person_to_poi+ratiosdf.from_poi_to_this_person
            , alpha=0.5)

plt.legend(ratiosdf.from_this_person_to_poi+ratiosdf.from_poi_to_this_person)
plt.title('From POI vs To POI ratios\nby Total emails with POI')
plt.xlabel('ratio to poi')
plt.ylabel('ratio from poi')
plt.show()

###Selecting Feautures to process with algorithms with Select K Best
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
def selectKbest(feat_count):
    kbest = SelectKBest(k = feat_count)
    gaus = GaussianNB()
    pipeline = Pipeline(steps = [('kbest', kbest), ('gaus', gaus)])
    pipeline.fit(features, labels)

    #Get Index for kbest
    kbest_index = kbest.get_support(indices = True)

    #add scores to dictionary with feature names as keys
    scores = {}
    for i in kbest_index:
        scores[features_list[i + 1]] = round(kbest.scores_[i],2)
    
    selected_features = scores.keys()
    selected_features.insert(0, 'poi')
    return selected_features, scores

#For Now I'm scoring each feature, when tuning 
#I'll loop through including additional features
#allow me to see which combination produces the 
#best accuracy, precision, recall and consequently 
#f1 score
selected_features, scores = selectKbest('all')
print '\nFeature Select K Best Scores:\n', pd.Series(scores).sort_values(ascending = False)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def split_data(dataset, feature_list):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
    return features_train, features_test, labels_train, labels_test

#create testing and training features and labels with 
#all selected features first then try variety in tuning later.
# features_train, features_test, labels_train, labels_test = split_data(data_dict, selected_features)        
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
def modelScoring(pred, test_labels):
    accuracy = accuracy_score(pred, test_labels)
    precision = precision_score(pred, test_labels)
    recall = recall_score(pred, test_labels)
    f1 = f1_score(pred, test_labels)
    confusion = confusion_matrix(pred, test_labels)
    
    scores = {'Accuracy': accuracy
              ,'Precision': precision 
              ,'Recall': recall 
              ,'F_Score': f1
              ,'Num of Predictions':len(test_labels)
              ,'True positives': confusion[1,1]
              ,'True negatives': confusion[0,0]
              ,'False positives': confusion[0,1]
              ,'False negatives': confusion[1,0]
              }
    return scores

def finalScores(scores):
    cols = ['Accuracy','Precision','Recall','F_Score','Num of Predictions','True positives','False positives','False negatives','True negatives']
    final_scores = pd.DataFrame(scores,cols).transpose()
    final_scores = final_scores.sort_values(['F_Score','Accuracy'],ascending=False)
    return final_scores

###Classifiers:
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(features_train,labels_train)
pred_nb = clf_nb.predict(features_test)
nb_scores = modelScoring(pred_nb, labels_test)

#Dtree
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier()
clf_dt.fit(features_train,labels_train)
pred_dt = clf_dt.predict(features_test)
dt_scores = modelScoring(pred_dt, labels_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
clf_lr= LogisticRegression()
clf_lr.fit(features_train,labels_train)
pred_lr = clf_lr.predict(features_test)
lr_scores = modelScoring(pred_lr, labels_test)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf_rfc = RandomForestClassifier(n_estimators = 500, n_jobs = -1)
clf_rfc.fit(features_train, labels_train)
pred_rfc = clf_rfc.predict(features_test)
rfc_scores = modelScoring(pred_rfc, labels_test)

#K Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier()
clf_knn.fit(features_train, labels_train)
pred_knn = clf_knn.predict(features_test)
knn_scores = modelScoring(pred_knn, labels_test)

scores = {'Naive Bayes': nb_scores
           ,'Decision Tree': dt_scores
           ,'Logistic Reg': lr_scores
           ,'Random Forest': rfc_scores
           ,'K Nearest': knn_scores}
print '\nFinal Scores for each untuned Classifier:\n', finalScores(scores)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#Naive Bayes Tuning
from sklearn.model_selection import GridSearchCV
pipeline = Pipeline([('scaler', MinMaxScaler())
                    ,('kbest', SelectKBest())
                    ,('clf', GaussianNB())
                   ])

kFeatures = range(1,len(features_list))
parameters = {'kbest__k': kFeatures}
nb_grid = GridSearchCV(estimator = pipeline
                      ,param_grid = parameters
                      ,scoring = 'f1'
                      ,cv = 10)
print '\nTraining NB Classifier\n'
nb_grid.fit(features_train, labels_train)

clf = nb_grid.best_estimator_
print '\nPredicting with NB Best Estimator\n'
pred = clf.predict(features_test)
print '\nTesting NB Classifier:\n', test_classifier(clf, my_dataset, features_list)

#Logistic Regression Tuning
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([('scaler', MinMaxScaler())
                    ,('kbest', SelectKBest())
                    ,('clf', LogisticRegression())
                   ])

kFeatures = range(1,len(features_list))
parameters = {'kbest__k': kFeatures
#              ,'clf__C':(.02,.04,.06,.08,.1,.12,.14,.16,.18,.2)
             ,'clf__C' : [1,1.1,1.2,1.3,1.4, 1.5,1.6,1.7,1.8,1.9,2] 
             ,'clf__solver':('newton-cg','lbfgs','liblinear','sag')
             }
print '\ngrid search LR\n'
lr_grid = GridSearchCV(estimator = pipeline
                      ,param_grid = parameters
                      ,scoring = 'f1'
                      ,cv = 10)
print '\nTraining LR Classifier\n'
lr_grid.fit(features_train, labels_train)

clf = lr_grid.best_estimator_
print '\nPredicting with LR Best Estimator\n'
pred = clf.predict(features_test)
print '\nTesting LR Classifier:\n', test_classifier(clf, my_dataset, features_list)

#K Nearest Neighbors Tuning
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([('scaler', MinMaxScaler())
                    ,('kbest', SelectKBest())
                    ,('clf', KNeighborsClassifier())
                   ])

kFeatures = range(1,len(features_list))
parameters = {'kbest__k': kFeatures
              ,'clf__n_neighbors':range(2,10)
              ,'clf__weights':['uniform','distance']
             }
print '\ngrid search KNN\n'
knn_grid = GridSearchCV(estimator = pipeline
                      ,param_grid = parameters
                      ,scoring = 'f1'
                      ,cv = 10
                       )

print '\nTraining KNN Classifier\n'
knn_grid.fit(features_train, labels_train)

clf = knn_grid.best_estimator_
print '\nPredicting with KNN Best Estimator\n'
pred = clf.predict(features_test)
print '\nTesting KNN Classifier:\n', test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(clf, my_dataset, features_list)
print '\nclf, my_dataset, features_list dumped'