import numpy as np
from stop_words import stop_words #A list of common words whose use in an article doesn't really help in classifying it. For ex:- a, to, the, here

class MultiNomialNBClass():
    def __init__(self, filter = "max_words", max_words = 5000, minimum_count = 10):
        self.max_words = max_words #the maximum number of words to have in vocabulary
        self.minimum_count = minimum_count  #the minimum frequency a word must have to be considered in vocabulary
        self.filter = filter #the parameter used for the filtering process. Allowed Values -> "max_words", "minimum_count"

    def fit(self, X, Y):
        self.vocab_words = self.make_vocabulary(X)
        dictionary = dict()
        for i in range(len(X)):
            if Y[i] not in dictionary:
                dictionary[Y[i]] = {}
            with open(X[i], mode="r") as curr_file:
                words = curr_file.read().split()
            for word in words:
                if word in self.vocab_words:
                    dictionary[Y[i]][word] = dictionary[Y[i]].get(word, 0) + 1
                    dictionary[Y[i]]["total words in current class"] = dictionary[Y[i]].get("total words in current class", 0) + 1
                    dictionary["total word count"] = dictionary.get("total word count", 0) + 1
        self.dictionary = dictionary
    
    def get_probability(self, x, curr_class, words_contained):
        output = np.log(self.dictionary[curr_class]["total words in current class"]) -  np.log(self.dictionary["total word count"])
        for word in words_contained:
            num = self.dictionary[curr_class].get(word, 0) + 1
            den = self.dictionary[curr_class]["total words in current class"] + len(self.vocab_words)
            prob = np.log(num) - np.log(den)
            output = output + prob
        return output

    def predict_point(self, x):
        max_prob = -1000
        best_class = -1
        first_run = True
        words_contained = set()
        with open(x, mode = "r") as curr_file:
            words = curr_file.read().split()
        for word in words:
            if word in self.vocab_words:
                words_contained.add(word)
        for curr_class in self.dictionary.keys():
            if curr_class == "total word count":
                continue
            prob_curr_class = self.get_probability(x, curr_class, words_contained)
            if (first_run or (prob_curr_class > max_prob)):
                max_prob = prob_curr_class
                best_class = curr_class
            first_run = False
        return best_class

    def predict(self, X):
        Y = []
        for i in range(len(X)):
            Y.append(self.predict_point(X[i]))
        return Y

    def make_vocabulary(self, X):
        complete_vocab = dict()
        for file in X:
            with open(file, mode="r") as curr_file:
                words = curr_file.read().split()
                for word in words:
                    if word not in stop_words:
                        complete_vocab[word] = complete_vocab.get(word, 0) + 1
        #print("Size of vocabulary(before filtering):", len(complete_vocab))
        if self.filter == "minimum_count":
            vocab = set()
            for key in complete_vocab:
                if(complete_vocab[key] >= self.minimum_count):
                    vocab.add(key)
        else:
            key_list = np.array(list(complete_vocab.keys()))
            value_list = np.array(list(complete_vocab.values()))
            sorted_value_index = np.argsort(value_list)
            sorted_keys = key_list[sorted_value_index]
            vocab = set(sorted_keys[-1:-self.max_words:-1])
        #print("Length of vocabulary(after filtering):", len(vocab))
        return vocab