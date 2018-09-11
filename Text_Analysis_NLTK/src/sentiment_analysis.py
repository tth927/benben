#%%
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:\MyWorkspace\python_pj\Text_Analysis_NLTK\datasets\\train.tsv", sep='\t')
# data.head()

# data.Sentiment.value_counts()

# Sentiment_count=data.groupby('Sentiment').count()
# plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
# plt.xlabel('Review Sentiments')
# plt.ylabel('Number of Review')
# plt.show()

#### Feature Generation using Bag of Words
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.tokenize import RegexpTokenizer
# #tokenizer to remove unwanted elements from out data like symbols and numbers
# token = RegexpTokenizer(r'[a-zA-Z0-9]+')
# cv = CountVectorizer(lowercase=True,stop_words='english',
#                      ngram_range = (1,1),
#                      tokenizer = token.tokenize)
# text_counts= cv.fit_transform(data['Phrase'])
# print(text_counts)

# # Split train and test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     text_counts, data['Sentiment'], test_size=0.3, random_state=1)

# # Model Building and Evaluation
# from sklearn.naive_bayes import MultinomialNB
# #Import scikit-learn metrics module for accuracy calculation
# from sklearn import metrics
# # Model Generation Using Multinomial Naive Bayes
# clf = MultinomialNB().fit(X_train, y_train)
# predicted= clf.predict(X_test)
# print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
##########################################

##### Feature Generation using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

phaseData  = data['Phrase']
targetData = data['Sentiment']

#### Improve Accuracy <START>
# 1. Broke the documents in list of words.
# 2. Removed stop words, punctuations.
# 3. Performed stemming.
# 4. Replaced numerical values with '#num#' to reduce vocabulary size.

# Tokenize
from nltk.tokenize import word_tokenize
phaseData_token =[word_tokenize(p) for p in phaseData]
# print(phaseData[1])
# print(phaseData_token[1])

# Remove Stopwords
# Stemming
# from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# stop_words=set(stopwords.words("english"))

# stemmed_phaseData = stemmed_phaseData.apply(lambda x : )
stemmed_phaseData=[]
for phase in phaseData_token:
    stem_item = []
    for w in phase:
        # if w not in stop_words:
        w = ps.stem(w)
        stem_item.append(w)
    stemmed_phaseData.append(stem_item)
# print("Filterd Sentence:",stemmed_phaseData[:50])

# Detokenize
from sacremoses import MosesDetokenizer
detokenizer = MosesDetokenizer()

phaseData_cleaned = [detokenizer.detokenize(s, return_str=True) for s in stemmed_phaseData]
# print(phaseData_cleaned[:50])

# Speech Tags

# print(phaseData[:5])
# a = stemmed_phaseData[:5]
# print(a)
# len(a)
# b = [[detokenizer.detokenize(s, return_str=True)] for s in a]
# print(b)


#### Improve Accuracy <END>

#%%
import datetime
import src.common_function as cf
tf=TfidfVectorizer(analyzer='word', token_pattern=r'[a-zA-Z]+', 
                   stop_words='english', ngram_range=(1,3))
# tf=TfidfVectorizer()
text_tf= tf.fit_transform(phaseData_cleaned)
# print(type(text_tf)) #scipy.sparse.csr.csr_matrix
# print(text_tf.shape) # (156060, 5000)
# tf.get_feature_names()


# trainDF = pd.SparseDataFrame(text_tf, columns=tf.get_feature_names())
# trainDF.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, targetData, test_size=0.3, random_state=123)

from sklearn.naive_bayes import MultinomialNB
# from sklearn import metrics
# # Model Generation Using Multinomial Naive Bayes
# clf = MultinomialNB().fit(X_train, y_train)
# predicted= clf.predict(X_test)
# print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

accuracy = cf.train_model(MultinomialNB(), X_train, X_test, y_train, y_test)
print(datetime.datetime.now().strftime("%H:%M:%S"),"MultinomialNB Accuracy:", accuracy)


##########################################

