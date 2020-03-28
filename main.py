import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import gc
import string


warnings.filterwarnings("ignore")
warnings.filterwarnings("error", message=".*check_inverse*.", category=UserWarning, append=False)

vocab_size = 10000
embedding_dim = 64
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
max_length = 200

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_labels = pd.read_csv("test_labels.csv")
test = test.merge(test_labels, on='id', how='inner')

del test_labels
gc.collect()

"""
def clean_text(text):
  cleaned_text=pd.Series

  for _, text in text.items():
    cleaned = []
    # replacing newlines and punctuations with space
    text =text.replace('\t', ' ').replace('\n', ' ')
    for punctuation in string.punctuation:
      text = text.replace(punctuation, ' ')
    text = text.split()

    # removing stop words and Stemming the remaining words in the resume
    stemmer = SnowballStemmer("english")
    for word in text:
      if word not in stopwords.words('english') and not word.isdigit():
        cleaned.append(word.lower())#stemmer.stem(word))

    cleaned = ' '.join(cleaned)
    cleaned_text.append(to_append=cleaned)

  return cleaned_text


train.comment_text = clean_text(train.comment_text)
test.comment_text = clean_text(test.comment_text)
"""


def clean_text(text):
  cleaned = []
  # replacing newlines and punctuations with space
  text = text.replace('\t', ' ').replace('\n', ' ')
  for punctuation in string.punctuation:
    text = text.replace(punctuation, ' ')
  text = text.split()
  # removing stop words and Stemming the remaining words in the resume
  stemmer = SnowballStemmer("english")
  for word in text:
    if word not in stopwords.words('english') and not word.isdigit():
      cleaned.append(word.lower())  # stemmer.stem(word))
  cleaned = ' '.join(cleaned)
  return cleaned


train.comment_text = train.comment_text.apply(lambda x: clean_text(x))
test.comment_text = test.comment_text.apply(lambda x: clean_text(x))


okenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
Tokenizer.fit_on_texts(train.comment_text)

word_index = Tokenizer.word_index

Train_sequences = Tokenizer.texts_to_sequences(train.comment_text)
Train_padded = pad_sequences(Train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

train['labels'] = train.apply(lambda x: np.array(x.toxic, x.severe_toxic, x.obscene, x.threat, x.insult, x.identity_hate))


