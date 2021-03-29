# This Program Will Be Demonstrating Using SVM for Data Modeling

import csv  # library used for importing CSV files
import numpy as np
from sklearn import svm  # library for creating SVM model
from sklearn.model_selection import cross_val_score  # library for running k-fold cross validation

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

# using linear kernel
model = svm.SVC(kernel='linear')

# run 10-fold CV for the linear SVC model:
print("now computing the accuracy for the linear SVC kernel: ")
scores = cross_val_score(model, features, label, scoring='accuracy', cv=10)
print("The average classification accuracy for the linear SVM model under the 10-fold CV is: " + str(scores.mean()))

# print message
print("")

# using polynomial kernel
model = svm.SVC(kernel='poly')

# run 10-fold CV for the polynomial SVC model:
print("now computing the accuracy for the polynomial SVC kernel: ")
scores = cross_val_score(model, features, label, scoring='accuracy', cv=10)
print("The average classification accuracy for the polynomial SVM model under the 10-fold CV is: " + str(scores.mean()))

# output:
# now computing the accuracy for the linear SVC kernel:
# The average classification accuracy for the linear SVM model under the 10-fold CV is: 0.7556321839080461

# now computing the accuracy for the polynomial SVC kernel:
# The average classification accuracy for the polynomial SVM model under the 10-fold CV is: 0.6789655172413793
