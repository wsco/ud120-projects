# %load svm_author_id.py
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score

def train_and_predict(type, svm, features_train, labels_train, features_test, labels_test):
    t0 = time()
    svm.fit(features_train, labels_train)
    print 'training time with svm''s '+ type +' kernel', time()-t0

    t1 = time()
    pred = svm.predict(features_test)
    print 'prediction time with svm''s '+ type +' kernel', time() - t1

    acc = accuracy_score(labels_test, pred)
    print acc , '\n'

linear_kernel_svm = svm.SVC(kernel = 'linear')    
train_and_predict('linear', linear_kernel_svm, features_train, labels_train, features_test, labels_test)
    
features_train_sample = features_train[:len(features_train)/100]
labels_train_sample = labels_train[:len(labels_train)/100]

train_and_predict('linear', linear_kernel_svm, features_train_sample,
                  labels_train_sample, features_test, labels_test)
    
rbf_kernel_svm = svm.SVC(kernel = 'rbf')
train_and_predict('rbf', rbf_kernel_svm, features_train_sample, 
                  labels_train_sample, features_test, labels_test)
rbf_kernel_svm = svm.SVC(kernel = 'rbf', C=10)
train_and_predict('rbf', rbf_kernel_svm, features_train_sample, 
                  labels_train_sample, features_test, labels_test)

rbf_kernel_svm = svm.SVC(kernel = 'rbf', C=100)
train_and_predict('rbf', rbf_kernel_svm, features_train_sample, 
                  labels_train_sample, features_test, labels_test)

rbf_kernel_svm = svm.SVC(kernel = 'rbf', C=1000)
train_and_predict('rbf', rbf_kernel_svm, features_train_sample, 
                  \labels_train_sample, features_test, labels_test)

rbf_kernel_svm = svm.SVC(kernel = 'rbf', C=10000)
train_and_predict('rbf', rbf_kernel_svm, features_train_sample, 
                  labels_train_sample, features_test, labels_test)

#########################################################

def time_with_power(power, people, times):
    results = nd.random.power(power, people)
    for i in range(times):
        results += nd.random.power(power, 100)
    return results

