#%%
##############################
# ONE TIME: Download the punkt corpus
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
###############################


text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome. 
The sky is pinkish-blue. You shouldn't eat cardboard"""

# Sentence tokenizer
from nltk.tokenize import sent_tokenize
tokenized_sent=sent_tokenize(text)
print(tokenized_sent)

# Word tokenizer
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)

# Frequency Distribution
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)
fdist.most_common(2)

# Frequency Distribution Plot
import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()

# Stopwords
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)

# Remove Stopwords
filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_sent)
print("Filterd Sentence:",filtered_sent)

# Stemming
from nltk.stem import PorterStemmer

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)


#Lexicon Normalization
#performing stemming and Lemmatization

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))

word = "better"
print("Lemmatized Word:",lem.lemmatize(word, pos="a"))
print("Stemmed Word:",stem.stem(word))

# POS Tagging
sent = "Albert Einstein was born in Ulm, Germany in 1879, he is very clever and good"
tokens=word_tokenize(sent)
print(tokens)

nltk.pos_tag(tokens)