# This Program Will Be Demonstrating Using SVM for Data Modeling

import csv  # library used for importing CSV files
import numpy as np
from sklearn import svm  # library for creating SVM model
from sklearn.model_selection import cross_val_score  # library for running k-fold cross validation
from sklearn import preprocessing  # library for preprocessing the data before model training
import timeit

# Import CSV file heart_failure_clinical_records_dataset.csv
f = open('heart_failure_clinical_records_dataset.csv', newline='')
feature_names = f.readline()
with f as csvfile:
    data = list(csv.reader(csvfile))

# turn data to a 2-d array
data2d = np.array(data)

# separate labels from features:
features = data2d[:, 1: 12]  # acquire the first 12 rows as feature
label = data2d[:, 12]  # acquire row 13 as the label

# data standardization
scaler = preprocessing.StandardScaler().fit(features)
features_scaled = scaler.transform(features)

# using linear kernel
model = svm.SVC(kernel='linear')

# run 10-fold CV for the linear SVC model:
print("now computing the accuracy for the linear SVC kernel: ")
start = timeit.default_timer()
scores = cross_val_score(model, features_scaled, label, scoring='accuracy', cv=10)
stop = timeit.default_timer()
print("The result of the 10-fold CV is: ")
print(scores)
print("The average classification accuracy for the linear SVM model under the 10-fold CV is: " + str(scores.mean()))
print("Model training time: " + str(stop-start))

# print message
print("")

# using polynomial kernel
model = svm.SVC(kernel='poly')

# run 10-fold CV for the polynomial SVC model:
print("now computing the accuracy for the polynomial SVC kernel: ")
start = timeit.default_timer()
scores = cross_val_score(model, features_scaled, label, scoring='accuracy', cv=10)
stop = timeit.default_timer()
print("The result of the 10-fold CV is: ")
print(scores)
print("The average classification accuracy for the polynomial SVM model under the 10-fold CV is: " + str(scores.mean()))
print("Model training time: " + str(stop-start))

# output:
# now computing the accuracy for the linear SVC kernel:
# The average classification accuracy for the linear SVM model under the 10-fold CV is: 0.7556321839080461

# now computing the accuracy for the polynomial SVC kernel:
# The average classification accuracy for the polynomial SVM model under the 10-fold CV is: 0.6789655172413793
