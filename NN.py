# This Program Will Be Demonstrating Using Neural Network for Data Modeling

import csv  # library used for importing CSV files
import numpy as np
from sklearn.neural_network import MLPClassifier  # lirbray for MLP training
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

# using the logistic activation function for logistic MLP model training
model = MLPClassifier(activation='logistic')

# run 10-fold CV for the logistic MLP model model:
print("now computing the accuracy for the logistic MLP model: ")
scores = cross_val_score(model, features, label, scoring='accuracy', cv=10)
print("The average classification accuracy for the logistic MLP model is: " + str(scores.mean()))

# using the logistic activation function for Tanh MLP model training
model = MLPClassifier(activation='tanh')

# run 10-fold CV for the logistic MLP model model:
print("now computing the accuracy for the Tanh MLP model: ")
scores = cross_val_score(model, features, label, scoring='accuracy', cv=10)
print("The average classification accuracy for the Tanh MLP model is: " + str(scores.mean()))

# output
# now computing the accuracy for the logistic MLP model:
# The average classification accuracy for the logistic MLP model is: 0.6789655172413793
# now computing the accuracy for the Tanh MLP model:
# The average classification accuracy for the Tanh MLP model is: 0.6789655172413793
