import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.utils import plot_model
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import gc
import string
import re


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

filter = train["comment_text"] != ""
train = train[filter]
train = train.dropna()

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
      #stemmer.stem(word)
      cleaned.append(word.lower())
  cleaned = ' '.join(cleaned)
  return cleaned
"""

def clean_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


train.comment_text = train.comment_text.apply(lambda x: clean_text(x))
test.comment_text = test.comment_text.apply(lambda x: clean_text(x))


"""
X = []
sentences = list(train["comment_text"])
for sen in sentences:
    X.append(clean_text(sen))

toxic_comments_labels = toxic_comments[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
y = toxic_comments_labels.values
"""

trainTokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
trainTokenizer.fit_on_texts(train.comment_text)

trainWord_index = trainTokenizer.word_index

Train_sequences = trainTokenizer.texts_to_sequences(train.comment_text)
Train_padded = pad_sequences(Train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# train['labels'] = train.apply(lambda x: np.array([x['toxic'], x['severe_toxic'], x['obscene'], x['threat'], x['insult'], x['identity_hate']]))
# train['labels'] = train.assign(np.array([train['toxic'], train['severe_toxic'], train['obscene'], train['threat'], train['insult'], train['identity_hate']]))

trainLabels = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values#.tolist()
#train['labels'] = train['labels'].apply(lambda x: np.asarray(x)).to_numpy()


testTokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
testTokenizer.fit_on_texts(test.comment_text)

testWord_index = testTokenizer.word_index

Test_sequences = testTokenizer.texts_to_sequences(test.comment_text)
Test_padded = pad_sequences(Test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#test['labels'] = test.apply(lambda x: np.array([x['toxic'], x['severe_toxic'], x['obscene'], x['threat'], x['insult'], x['identity_hate']]))
#test['labels'] = test.assign(np.array([test['toxic'], test['severe_toxic'], test['obscene'], test['threat'], test['insult'], test['identity_hate']]))

test[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = test[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].replace(0,1)
test[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = test[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].replace(-1,0)

testLabels = test[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values#.tolist()
#test['labels'] = test['labels'].apply(lambda x: np.asarray(x)).to_numpy()

#test['labels'] = test['labels'].replace(0, 1)
#test['labels'] = test['labels'].replace(-1, 0)

#train.labels = np.asarray(train.labels)
#test.labels = np.asarray(test.labels)
#train.labels = train.labels.to_numpy()
#test.labels = test.labels.to_numpy()


BUFFER_SIZE = 10000
BATCH_SIZE = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(6, activation='sigmoid')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)

NUM_EPOCHS = 5
history = model.fit(Train_padded, trainLabels, epochs=NUM_EPOCHS, validation_data=(Test_padded, testLabels), verbose=1)
#history = model.fit(np.array(Train_padded), np.array(trainLabels), epochs=NUM_EPOCHS, validation_data=(np.array(Test_padded), np.array(testLabels)), verbose=1)

results = model.evaluate(Test_padded, testLabels, batch_size=BATCH_SIZE)

gc.collect()

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

