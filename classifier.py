from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk import classify
from nltk.stem import WordNetLemmatizer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import nltk
import numpy

"""
Current workflow:
- select tokenizer --> Regex erases punctuations, word_tokenize not (reminder change tokenizer)
- open reviews 
- label reviews based on manual selection
- get general word distribution for feature_vector creation
- create feature vector for each review
- train classifier
"""


#define tokenizer, lemmatizer
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

#getm all the reviews
f = open(r'/home/ji78remu/Schreibtisch/CaseStudySeminar/txtSaver/con80_white.txt')
x = f.readlines()
f.close()

#positive reviews
f = open(r'/home/ji78remu/Schreibtisch/CaseStudySeminar/txtSaver/con80_white\pos.txt')
p_x = f.readlines()
f.close()

#negative reviews
f = open(r'/home/ji78remu/Schreibtisch/CaseStudySeminar/txtSaver/con80_white\neg.txt')
n_x = f.readlines()
f.close()

#create labeled trainingSet
x_Set = []

#create adjectives dict
dict_adjectives = Counter()
dict_nouns = Counter()

for review in x:
    classLabel = review in p_x          #True == positive sentiment; False == negative sentiment

    words_array = tokenizer.tokenize(review.replace('\n','').replace('1/2', 'halth'))   #tokenize 
    
    #part of speech tagging
    part_of_speech = numpy.array(nltk.pos_tag(words_array)) #.pos_tag() should be a valid approach, since sentence structure is fairly simple 
    adjectives_sql = numpy.where(part_of_speech == 'JJ') #numpy.asarray(part_of_speech == 'JJ').nonzero()
    nouns_sql = numpy.where(numpy.logical_or(part_of_speech == 'NN', part_of_speech == 'NNS'))
    adjectives_indexes = adjectives_sql[0]
    nouns_indexes = nouns_sql[0]
 
    for i in adjectives_indexes:
        adjective = words_array[i]
        dict_adjectives[adjective] += 1

    for i in nouns_indexes:
        noun = words_array[i]
        dict_nouns[noun] += 1

    words_array = [stemmer.stem(word) for word in words_array]                          #stem
    words_array = [lemmatizer.lemmatize(word) for word in words_array]                  #lem

    review = ' '.join(words_array)

    x_Set.append((review, classLabel))

print(dict_adjectives.most_common(10))
print(dict_nouns.most_common(20))



#get words distribution for feature vector creation
all_words = []
for sentence in x:
    
    sentence = tokenizer.tokenize(sentence.replace('\n','').replace('1/2', 'halth').replace('0.5', 'halth'))    #tokenize
    for word in sentence:
        word = stemmer.stem(word)                                                       #stem
        word = lemmatizer.lemmatize(word)                                               #lem
        all_words.append(word.lower())


#stopword removal (not within tutotrial)
stop_words = set(stopwords.words('english'))
filtered_all_words = [w for w in all_words if not w in stop_words]


#transposition of word to words + freq_dist
filtered_all_words = FreqDist(filtered_all_words); print(filtered_all_words.most_common(15))

#feature extraction
word_features = numpy.array(filtered_all_words.most_common(100))[:,0] #most commonly used words

#create feature_vectors for the each review
def find_features(review):
    words = tokenizer.tokenize(review)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
featureSet = [(find_features(x[0]), x[1]) for x in x_Set]


#n-gram approach with logistic regression
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(x)
train_set = ngram_vectorizer.transform(x)
test_set = ngram_vectorizer.transform(x)
target = numpy.array(x_Set)[:,1]

n_gram_vocab = ngram_vectorizer.vocabulary_
n_gram_vocab_new = Counter(n_gram_vocab).most_common(100)
print(n_gram_vocab_new)

X_train, X_val, y_train, y_val = train_test_split(
    train_set, target, train_size = 0.75
)

#logistic regression
for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))


#split data
train_set = featureSet[:300]
test_set = featureSet[300:]

#train classifier 
classifier = classify.NaiveBayesClassifier.train(train_set)

print("Naive Bayes accuracy percent:",(classify.accuracy(classifier, test_set))*100)
#classifier.show_most_informative_features(15)

#save classifier to pickel file
#save_classifier = open("naivebayes.pickle","wb")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()

#other classifiers
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(train_set)
print("MultinomialNB accuracy percent:",classify.accuracy(MNB_classifier, test_set)*100)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(train_set)
print("BernoulliNB accuracy percent:",classify.accuracy(BNB_classifier, test_set)*100)

"""
#how to open classifier
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()"""