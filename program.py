# Training and Testing data should be fed to the algorithm in the following format:
# 1) Input, i.e, X is a list of strings where each string represents a path at which the text file is located
# 2) Result, i.e, Y is a 1-dimensional iterable whose length must be equal to the length of the Input

import os
from MultinomialNB import MultiNomialNBClass
from sklearn import model_selection
from sklearn.metrics import classification_report
from time import time

start_time = time()
X = []
Y = []
for subfolder in os.listdir("20_newsgroups"):
    for filename in os.listdir("20_newsgroups/{sf}".format(sf = subfolder)):
        X.append("20_newsgroups/{sf}/{fn}".format(sf = subfolder, fn = filename))
        Y.append(subfolder)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, random_state=1)

clf = MultiNomialNBClass(filter="minimum_count")
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("Classification Report for self implementation of MultinomialNB algorithm:\n")
print(classification_report(Y_test, Y_pred))
end_time = time()
print("---------------Execution Time: {tt} sec---------------".format(tt = end_time - start_time))