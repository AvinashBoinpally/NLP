from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding
from keras.layers import Bidirectional, LSTM, GlobalMaxPool1D
from keras.models import Model

from sklearn.metrics import roc_auc_score


EMBEDDING_DIM = 100
MAX_VOC = 20000
SEQ_LENGTH = 200

word2Vec = {}
with open(os.path.join(f'../generic_data/glove.6B/glove.6B.{EMBEDDING_DIM}d.txt')) as f:
  for line in f:
    row = line.split()
    word2Vec[row[0]] = np.asarray(row[1:], dtype='float32')
print(f'Total words {len(word2Vec)}')


train = pd.read_csv('toxic-comment-classification/train.csv')
comments = train['comment_text']
tar = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
target = train[tar].values


tokenizer = Tokenizer(num_words=MAX_VOC)
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)

print(f'Max words in text {max(len(s) for s in sequences)}')
print(f'Min words in text {min(len(s) for s in sequences)}')

word2idx = tokenizer.word_index
data = pad_sequences(sequences, maxlen=SEQ_LENGTH)
num_words = min(MAX_VOC, len(word2idx) + 1)

embedded_mtx = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word2idx.items():
  if i < MAX_VOC:
    embedded_vec = word2Vec.get(word)
    if embedded_vec is not None:
      embedded_mtx[i] = embedded_vec


embedded_lyr = Embedding(num_words,EMBEDDING_DIM,
weights =  [embedded_mtx], trainable=False,
input_length=200)


input = Input(shape=(200,))
x = embedded_lyr(input)
x = Bidirectional(LSTM(15, activation='relu', return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(tar), activation= 'sigmoid')(x)

model = Model(input, output)

model.compile(
  loss='binary_crossentropy',
  optimizer='rmsprop',
  metrics = ['accuracy']
)

res = model.fit(
  data,
  target,
  batch_size=512,
  epochs=10,
  validation_split=0.2
)

plt.plot(res.hisory['loss'], label= 'loss')
plt.plot(res.hisory['val_loss'], label='Val_loss')
plt.legend()
plt.show()


plt.plot(res.hisory['accuracy'], label= 'acc')
plt.plot(res.hisory['val_accuracy'], label='Val_acc')
plt.legend()
plt.show()


pred = model.predict(data)
aucs = []
for u in range(6):
  auc = roc_auc_score(target[:, u], pred[:, u])
  aucs.append(auc)
print(np.mean(aucs))


















