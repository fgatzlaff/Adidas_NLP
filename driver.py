from TextCleaner import CorporaAnalyzer as CorporaAnalyzer
from NaiveBayes import Classifier as NBClassifier
from LinReg import Classifier as LinRegClassifier
import numpy as np
import itertools

"""
General variables
"""

#all reviews
f = open(r'/home/ji78remu/Schreibtisch/CaseStudySeminar/txtSaver/con80_white.txt')
all_reviews = f.readlines()
f.close()

#positive reviews
f = open(r'/home/ji78remu/Schreibtisch/CaseStudySeminar/txtSaver/con80_white\pos.txt')
pos_reviews = f.readlines()
f.close()

#create class label                 #True == positive sentiment; False == negative sentiment
classLabel = [(review in pos_reviews) for review in all_reviews]


#NB preprocess step
Preprocessor = CorporaAnalyzer(all_reviews)
tokenized_corpora = Preprocessor.tokenize_corpora(all_reviews)
cleaned_corpora = Preprocessor.clean(all_reviews)
most_frequent_words = Preprocessor.most_frequent_words(cleaned_corpora)

#NB classifier
Classifier = NBClassifier()
vectorized_reviews = Classifier.feature_vectorization(all_reviews, 
most_frequent_words)

data_set = np.column_stack((vectorized_reviews, classLabel))
train_set = data_set[:300]
test_set = data_set[300:]

NB_accuracy = Classifier.fit(train_set,test_set)
#highest accuracy achieved with: accuracy= 85.84905660377359  tok= 1  stem= 3  lem= 0  stop= 1  dist= 1
#                                accuracy= 85.84905660377359  tok= 2           lem= 1  stop= 1  dist= 0
#                                accuracy= 85.84905660377359  tok= 2           lem= 1  stop= 1  dist= 2 
#                                accuracy= 85.84905660377359  tok= 2  stem= 3  lem= 0  stop= 1  dist= 0

#LinReg classifier
Classifier = LinRegClassifier()

data_set = np.column_stack((all_reviews, classLabel))
train_set = data_set[:300]
test_set = data_set[300:]

LR_accuracy = Classifier.fit(all_reviews,train_set,test_set)
#highest accuracy achieved with: accuracy= 85.71428571428571 penalty= 'l1'  reg_strength= 1 
#highest accuracy achieved with: accuracy= 82.85714285714286 penalty= 'l2'  reg_strength= 1 
#highest accuracy achieved with: accuracy= 84.76190476190476 penalty= 'elasticnet'  reg_strength= 1 '
#highest accuracy achieved with: accuracy= 83.80952380952381 penalty= 'none'  reg_strength= 1 
