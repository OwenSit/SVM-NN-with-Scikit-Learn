# This Program Will Be Demonstrating Using Neural Network for Data Modeling

import csv  # library used for importing CSV files
import numpy as np
from sklearn.neural_network import MLPClassifier  # lirbray for MLP training
from sklearn.model_selection import cross_val_score  # library for running k-fold cross validation
from sklearn import preprocessing # library for preprocessing the data before model training
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

# using the logistic activation function for logistic MLP model training
model = MLPClassifier(activation='logistic')

# run 10-fold CV for the logistic MLP model model:
print("now computing the accuracy for the logistic MLP model: ")
start = timeit.default_timer()
scores = cross_val_score(model, features, label, scoring='accuracy', cv=10)
stop = timeit.default_timer()
print("The result of the 10-fold CV is: ")
print(scores)
print("The average classification accuracy for the logistic MLP model is: " + str(scores.mean()))
print("Model training time: " + str(stop-start))

# using the logistic activation function for Tanh MLP model training
model = MLPClassifier(activation='tanh')

# run 10-fold CV for the logistic MLP model model:
print("now computing the accuracy for the Tanh MLP model: ")
start = timeit.default_timer()
scores = cross_val_score(model, features, label, scoring='accuracy', cv=10)
stop = timeit.default_timer()
print("The result of the 10-fold CV is: ")
print(scores)
print("The average classification accuracy for the Tanh MLP model is: " + str(scores.mean()))
print("Model training time: " + str(stop-start))

# output
# now computing the accuracy for the logistic MLP model:
# The average classification accuracy for the logistic MLP model is: 0.6789655172413793
# now computing the accuracy for the Tanh MLP model:
# The average classification accuracy for the Tanh MLP model is: 0.6789655172413793
