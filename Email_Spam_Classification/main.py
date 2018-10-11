#%%
import os
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer


# According to protocol, email headers and bodies are separated by a blank line (NEWLINE)
NEWLINE = '\n'
SKIP_FILES = {'cmds'}


def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    f = open(file_path, encoding="latin-1")
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content


def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame



#%%
# A:: Prepare Data
HAM = 'ham'
SPAM = 'spam'

SOURCES = [
    # ('data/spam',        SPAM),
    # ('data/easy_ham',    HAM),
    # ('data/hard_ham',    HAM),
    ('C:\\MyWorkspace\\python_data\\Email_Spam_Classification\\data\\beck-s', HAM),
    # ('data/farmer-d',    HAM),
    # ('data/kaminski-v',  HAM),
    # ('data/kitchen-l',   HAM),
    # ('data/lokay-m',     HAM),
    # ('data/williams-w3', HAM),
    ('C:\\MyWorkspace\\python_data\\Email_Spam_Classification\\data\\BG', SPAM)
    # ('data/GP',          SPAM),
    # ('data/SH',          SPAM)
]

data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

# use DataFrame’s reindex to shuffle the whole dataset. 
# Otherwise, we’d have contiguous blocks of examples from each source. 
# This is important for validating prediction accuracy later.
data = data.reindex(np.random.permutation(data.index))


#%%
# B:: Extract Features
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(data['text'].values)

#%%
# C:: Train Classification
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

#%%
# C1: Test
examples = ['Free Viagra call today!', "I'm going to attend the Linux users group tomorrow."]
example_counts = count_vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
predictions # [1, 0]

#%%
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  MultinomialNB()) ])

# Validation
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

k_fold = KFold(n=len(data), n_folds=6)
scores = []
confusion = np.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    X_train = data.iloc[train_indices]['text'].values
    y_train = data.iloc[train_indices]['class'].values

    X_test = data.iloc[test_indices]['text'].values
    y_test = data.iloc[test_indices]['class'].values

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    confusion += confusion_matrix(y_test, predictions)
    score = f1_score(y_test, predictions, pos_label=SPAM)
    scores.append(score)

print('Total emails classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)

#%%
# ZZ: Dummy Code
data.head()
data.groupby('class').size()

counts.shape