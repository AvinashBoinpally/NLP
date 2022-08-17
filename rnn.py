from msilib import sequence
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Embedding, LSTM, Dense, Input
from keras.layers import GlobalMaxPool1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.metrics import roc_auc_score

from .toxic_cnn import BATCH_SIZE, EMBEDDING_DIM, EPOCHS, MAX_VOC, VALIDATION_SPLIT

EMBEDDING_DIM = 100
MAX_VOC = 20000
BATCH_SIZE = 128
EPOCHS = 10
VALIDATION_SPLIT = 0.2

word2Vec = {}

with open(os.path.join(f'../generic_data/glove.6B/glove.6B.{EMBEDDING_DIM}d.txt')) as f:
  for line in f:
    row = line.split(' ')
    word2Vec[row[0]] = np.asarray(row[1:], dtype='float32')
print(f'total words {len(word2Vec)}')

train = pd.read_csv('toxic-comment-classification/train.csv')
comments = train['comment_text'].values
tar = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
target = train[tar].values

tokenizer = Tokenizer(num_words=MAX_VOC)
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)

word2idx = tokenizer.word_index

data = pad_sequences(sequences, maxlen=EMBEDDING_DIM)
num_words = min(MAX_VOC, len(word2idx) + 1)

embedded_mtx = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < num_words:
    embedded_vct = word2Vec.get(word)
    if embedded_vct is not None:
      embedded_mtx[i] = embedded_vct

embedded_lyr  = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedded_mtx],
  input_length=EMBEDDING_DIM,
  trainable = False
)

input_ = Input(shape=(EMBEDDING_DIM,))
x = embedded_lyr(input_)
x = LSTM(128, activation='relu', return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(tar), activation='sigmoid')(x)

model = Model(input_, output)
model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

res = model.fit(
  data,
  target,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split= VALIDATION_SPLIT
)


# plot some data
plt.plot(res.history['loss'], label='loss')
plt.plot(res.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(res.history['accuracy'], label='acc')
plt.plot(res.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# plot the mean AUC over each label
p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(target[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))


