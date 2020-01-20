from nltk.tokenize import *
from nltk.stem import *
from nltk.corpus import stopwords 
from nltk.probability import FreqDist
from nltk import pos_tag as pos_tag
from collections import Counter
import numpy as np

class CorporaAnalyzer(object):
    def __init__(self, corpora, tokenizer=0, stemmer=0, lemmatization_enabled=True, stopwords_removal=True):
        """
        Initialize the class with with desired tokenizer, stemmer and lemmatizer
        """    
        self.corpora = corpora
        self.cleaned_corpora_set = []
        self.tokenized_corpora =   []
        self.all_words = []

        print("tokenizer: ", tokenizer)
        print("stemmer: ", stemmer)
        print("lemmatizer: ", lemmatization_enabled)
        print("stopwords: ", stopwords_removal)

        if tokenizer <= 2:
            if tokenizer == 0:
                self.tokenizer = RegexpTokenizer(r'\w+')
            elif tokenizer == 1:
                self.tokenizer = TreebankWordTokenizer()
            else:
                self.tokenizer = TweetTokenizer()
        else:
            assert tokenizer <= 2,"you used the wrong tokenizer value"

        if stemmer <= 3:
            if stemmer == 0:
                self.stemmer = PorterStemmer()
            elif stemmer == 1:
                self.stemmer = LancasterStemmer()
            elif stemmer == 2:
                self.stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)                       #manually modifiable stemmer
            else:
                self.stemmer = Cistem(case_insensitive=False)                                 #favorite german stemmer
        else:
            assert stemmer <= 3,"you used the wrong stemmer value"

        if lemmatization_enabled:
            self.lemmatization_enabled = True
            self.stemmer = WordNetLemmatizer()
        else:
            self.lemmatization_enabled = False
            #print("no lemmatization was selected")  

        if stopwords_removal:
            self.stopwords_removal = True
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stopwords_removal = False
            #print("no stopword removal was selected")   

    def tokenize_corpora(self, corpora):
        tokenized_corpora =   []
        for review in corpora:
            review_vector = self.tokenizer.tokenize(review)   #tokenize
            tokenized_corpora.append(review_vector)
        return tokenized_corpora

    def clean(self, corpora):
        """
        preprocesses the whole corpora (cleans the reviews based on the chosen technique)
        """
        assert corpora != None, "no defined corpora"

        cleaned_corpora_set = []

        for review in corpora:
            review_vector = self.tokenizer.tokenize(review)   #tokenize 
            if self.lemmatization_enabled:
                review_vector = [self.stemmer.lemmatize(word) for word in review_vector] #lem
            else:    
                review_vector = [self.stemmer.stem(word) for word in review_vector] #stem
            for word in review_vector:
                word.lower()
                self.all_words.append(word)
            if self.stopwords_removal:
                review_vector = [w for w in review_vector if not w in self.stop_words]
            review = ' '.join(review_vector)
            cleaned_corpora_set.append(review)
        self.cleaned_corpora_set = cleaned_corpora_set
        return cleaned_corpora_set
        
    def most_frequent_words(self, corpora, magnitude=100):
        """
        returns the most frequent words of the cleaned corpora
        """
        all_words = []
        for review in corpora:
            review_vector = self.tokenizer.tokenize(review)   #tokenize 
            if self.lemmatization_enabled:
                review_vector = [self.stemmer.lemmatize(word) for word in review_vector] #lem
            else:    
                review_vector = [self.stemmer.stem(word) for word in review_vector] #stem
            for word in review_vector:
                word.lower()
                all_words.append(word)
        return np.array(FreqDist(all_words).most_common(magnitude))[:,0]

    def most_frequent_Nouns(self, tagger=0, magnitude=20):
        """
        returns the most frequent nouns of the cleaned corpora
        """
        dict_nouns = Counter()
        for tokenized_review in self.tokenized_corpora:
            part_of_speech = np.array(pos_tag(tokenized_review))
            part_of_speech_nouns_only = np.where(np.logical_or(part_of_speech == 'NN', part_of_speech == 'NNS'))
            nouns_indexes = part_of_speech_nouns_only[0]
            for i in nouns_indexes:
                noun = tokenized_review[i]
                dict_nouns[noun] += 1
        return dict_nouns.most_common(magnitude)

    def most_frequent_Adjectives(self, magnitude=100):
        """
        returns the most frequent words of the cleaned corpora
        """
        dict_adjectives = Counter()
        for tokenized_review in self.tokenized_corpora:
            part_of_speech = np.array(pos_tag(tokenized_review))
            part_of_speech_adjectives_only = np.where(part_of_speech == 'JJ')
            adjectives_indexes = part_of_speech_adjectives_only[0]
            for i in adjectives_indexes:
                adjective = tokenized_review[i]
                dict_adjectives[adjective] += 1
        return dict_adjectives.most_common(magnitude)