from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter

class Classifier(object):
    def __init__(self):
        """
        Initialize the class with parameter with desired tokenizer, stemmer and lemmatizer
        """   

    def fit(self, corpora, train_set, test_set, magnitude = 100):
        """

        """
        corpora = np.array(corpora)
        split = np.size(train_set,0) / np.size(corpora,0)

        ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
        self.vectorizer = ngram_vectorizer.fit(corpora)
        
        data_set = np.concatenate((train_set, test_set), axis=0)
        text_data = data_set[:,0].tolist()
        classlabel = np.array(data_set)[:,1]
        transposed_text_set = ngram_vectorizer.transform(text_data)
        
        self.most_common_ngrams = Counter(ngram_vectorizer.vocabulary_).most_common(magnitude)

        X_train, X_val, y_train, y_val = train_test_split(
            transposed_text_set, classlabel, train_size = split
        )
        #logistic regression
        final_accuracy = 0
        for c in [0.01, 0.05, 0.25, 0.5, 1]:
            lr = LogisticRegression(C=c)
            classifier = lr.fit(X_train, y_train)

            running_accuracy = accuracy_score(y_val, lr.predict(X_val))

            if(running_accuracy > final_accuracy):
                final_accuracy = running_accuracy
                self.classifier = classifier

        print("nGram Logistic Regression accuracy:",final_accuracy)

    def predict(self, reviews):
        """
        Classification step

        X has shape (M,2). It contains M feature vectors to classify in 2-d feature space

        Returns:
            A vector of shape (M,) with M classification results (class labels)
        """
        vectorized_reviews = self.vectorizer.transform(reviews)
        return self.classifier.predict(vectorized_reviews)