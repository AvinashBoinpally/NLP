from msilib import sequence
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding
from keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D
from keras.models import Model
from sklearn.metrics import roc_auc_score

EMBEDDING_DIM = 100
MAX_VOC = 20000
BATCH_SIZE = 128
EPOCHS = 10
VALIDATION_SPLIT = 0.2

word2Vec = {}
with open(os.path.join(f'../generic_data/glove.6B/glove.6B.{EMBEDDING_DIM}d.txt')) as f:
  for line in f:
    row = line.split()
    word2Vec[row[0]] = np.asarray(row[1:], dtype ='float32')
print(f'Total words {len(word2Vec)}')

# Reading the comments and categories
train = pd.read_csv('toxic-comment-classification/train.csv')
comments = train['comment_text'].values
tar = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
target = train[tar].values

# Processing the comments
tokenizer = Tokenizer(num_words=MAX_VOC)
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)

print(f'max words in text {max(len(s) for s in sequences)}')
print(f'min words in text {min(len(s) for s in sequences)}')

word2idx = tokenizer.word_index

data = pad_sequences(sequences, maxlen=EMBEDDING_DIM)


num_words  = min(MAX_VOC, len(word2idx) + 1)
embedded_mtx = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOC:
    embedded_vec = word2Vec.get(word)
    if embedded_vec is not None:
      embedded_mtx[i] = embedded_vec

embedded_lyr = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights = [embedded_mtx],
  input_length= EMBEDDING_DIM,
  trainable = False
)

input = Input(shape=(EMBEDDING_DIM,))
x = embedded_lyr(input)
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPool1D(3)(x)  
x = Conv1D(128, 3, activation='relu')(x)
x = MaxPool1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(tar), activation='sigmoid')(x)

model = Model(input, output)
model.compile(
  loss='binary_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)


mod_res = model.fit(
  data,
  target,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split = VALIDATION_SPLIT
)

# plot some data

plt.plot(mod_res.history['loss'], label='loss')
plt.plot(mod_res.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(mod_res.history['accuracy'], label='acc')
plt.plot(mod_res.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# plot the mean AUC over each label
p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(target[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))






