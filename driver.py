from TextCleaner import CorporaAnalyzer as CorporaAnalyzer
from NaiveBayes import Classifier as NBClassifier
from LinReg import Classifier as LinRegClassifier
import numpy as np

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

#TODO: tokenizer = 1; tokenizer = 2
#TODO: stemmer = 1, stemmer = 2
#preprocess all_reviews
Preprocessor = CorporaAnalyzer(all_reviews, lemmatization_enabled=False, stopwords_removal=True)
tokenized_corpora = Preprocessor.tokenize_corpora(all_reviews)
cleaned_corpora = Preprocessor.clean(all_reviews)
most_frequent_words = Preprocessor.most_frequent_words(cleaned_corpora)

"""
#NB classifier
Classifier = NBClassifier()
vectorized_reviews = Classifier.feature_vectorization(all_reviews, most_frequent_words)

data_set = np.column_stack((vectorized_reviews, classLabel))
train_set = data_set[:300]
test_set = data_set[300:]

Classifier.fit(train_set,test_set,distribution=0)
"""

#LinReg classifier
Classifier = LinRegClassifier()

data_set = np.column_stack((all_reviews, classLabel))
train_set = data_set[:300]
test_set = data_set[300:]

Classifier.fit(all_reviews,train_set,test_set)
