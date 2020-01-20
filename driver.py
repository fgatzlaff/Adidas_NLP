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

accuracy = 0
tok = 0
stem = 0
dist = 0
lem = 0
stop = 0

for i, j, x, l, s in itertools.product(range(3),range(4), range(3), range(2), range(2)):
    #preprocess all_reviews
    Preprocessor = CorporaAnalyzer(all_reviews, tokenizer=i, stemmer=j, lemmatization_enabled=l, stopwords_removal=s)
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

    running_accuracy = Classifier.fit(train_set,test_set,distribution=x)

    if running_accuracy > accuracy: 
        accuracy = running_accuracy
        tok = i; stem = j; dist = x; lem = l; stop = s

print("highest accuracy achieved with: accuracy=",accuracy," tok=",tok, " stem=",stem, " lem= ",lem, " stop=", stop, " dist=",dist)

"""
#LinReg classifier
Classifier = LinRegClassifier()

data_set = np.column_stack((all_reviews, classLabel))
train_set = data_set[:300]
test_set = data_set[300:]

Classifier.fit(all_reviews,train_set,test_set)
"""