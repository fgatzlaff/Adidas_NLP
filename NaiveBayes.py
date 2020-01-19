from nltk import classify
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

class Classifier(object):
    def __init__(self):
        """
        Initialize the class with parameter with desired tokenizer, stemmer and lemmatizer
        """
        self.most_frequent_words = None

    def feature_vectorization(self, tokenized_reviews, most_frequent_words):
        """
        Classification step

        tokenized_review has shape (N). It contains N words as vector strings
        most_frequent_words has shape (M). It contains M words as vector strings

        Returns:
            A dict of shape (M,) with M classification results in regards to N(True:False)

        Creates a feature vector for the given review based on the most frequent words among positive reviews 
        """
        self.most_frequent_words = most_frequent_words
        
        vectorized_reviews = []

        for tokenized_review in tokenized_reviews:
            features = {}
            for frequent_word in most_frequent_words:
                features[frequent_word] = (frequent_word in tokenized_review)
            vectorized_reviews.append(features)

        return vectorized_reviews

    def fit(self, train_set, test_set, distribution = 0):
        """
        Training step: Initializes this class with desired distribution
        0: Gaussian
        1: Multinomial
        2: Bernoulli

        train_set has shape (N,2). It contains N feature vectors and class-label
        test_set has shape (N,2). It contains N feature vectors and class-label
        """
        if distribution >= 0 and distribution <= 2:
            if distribution == 0:
                self.classifier = classify.NaiveBayesClassifier.train(train_set)
                print("GaussianNB accuracy:",(classify.accuracy(self.classifier, test_set))*100,"%")
            elif distribution == 1:
                self.classifier = SklearnClassifier(MultinomialNB()).train(train_set)
                print("MultinomialNB accuracy:",(classify.accuracy(self.classifier, test_set))*100,"%")
            else:
                self.classifier = SklearnClassifier(BernoulliNB()).train(train_set)
                print("BernoulliNB accuracy:",(classify.accuracy(self.classifier, test_set))*100,"%")
        else:
            raise ValueError('invalid distribution value')
        
    def predict(self, reviews):
        """
        Classification step

        reviews has shape (M,N). It contains M reviews to classify in N-d feature space

        Returns:
            A vector of shape (M,) with M classification results (class labels)
        """
        vectorized_reviews = self.feature_vectorization(reviews, self.most_frequent_words)
        return self.classifier.classify_many(vectorized_reviews)