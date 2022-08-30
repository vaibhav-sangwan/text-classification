# Training and Testing data should be fed to the algorithm in the following format:
# 1) Input, i.e, X is a 2-D numpy array which contains M data points and N features/words
# 2) Result, i.e, Y is a 1-dimensional iterable whose length must be equal to the length of the Input(M)

import os
import numpy as np
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from MultinomialNB import MultiNomialNBClass
from time import time

start_time = time()
X = []
Y = []
for subfolder in os.listdir("20_newsgroups"):
    for filename in os.listdir("20_newsgroups/{sf}".format(sf = subfolder)):
        X.append("20_newsgroups/{sf}/{fn}".format(sf = subfolder, fn = filename))
        Y.append(subfolder)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, random_state=1)

alg = MultiNomialNBClass(max_words=30000)
vocab = alg.make_vocabulary(X_train)

M = len(Y_train)
n = len(vocab)
features_list = list(vocab)
features_dict = {features_list[i]:i for i in range(n)}
x_train_df = np.zeros((M, n), dtype=int)
for i in range(M):
    with open(X_train[i], mode="r") as curr_file:
        words = curr_file.read().split()
    for word in words:
        if(word in features_dict):
            x_train_df[i, features_dict[word]] += 1

M = len(Y_test)
x_test_df = np.zeros((M, n), dtype=int)
for i in range(M):
    with open(X_test[i], mode="r") as curr_file:
        words = curr_file.read().split()
    for word in words:
        if(word in features_dict):
            x_test_df[i, features_dict[word]] += 1

alg = MultinomialNB()
alg.fit(x_train_df, Y_train)
Y_pred = alg.predict(x_test_df)
print("Classification Report for SKLearn's implementation of MultinomialNB algorithm:\n")
print(classification_report(Y_test, Y_pred))
end_time = time()
print("---------------Execution Time: {tt} sec---------------".format(tt = end_time - start_time))