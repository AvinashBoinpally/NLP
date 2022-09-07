
import pandas as pd
import numpy as np
from keras.layers import Dense, LSTM, Bidirectional, Input, Embedding, Bidirectional, Concatenate, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils.vis_utils import plot_model

import os


EMBEDDED_DIM = 100
MAX_VOC = 30000
LATENT_DIM = 64




data = pd.read_csv('./cnn_dailymail/train.csv', nrows=1000)
summaries = list(data['highlights'])
articles = list(data['article'])

summaries_pref =  ['<sos> ' + x +' <eos>' for x in summaries]

all_data = summaries_pref + articles

print(f'Articles: {len(articles)} Summaries: {len(summaries_pref)} All_data: {len(all_data)}')


word2Vec = {}

with open(os.path.join(f'../generic_data/glove.6B/glove.6B.{EMBEDDED_DIM}d.txt')) as f:
  for line in f:
    row = line.split(' ')
    word2Vec[row[0]] = np.array(row[1:], dtype='float32')

print(f'Total word Vects {len(word2Vec)}')

max_len_sent = np.max([len(sent.split(" ")) for sent in articles])
max_len_summary = np.max([len(sent.split(" ")) for sent in summaries_pref])

print(f'Max lengt of sentence {max_len_sent} & Summaries {max_len_summary}')
unique_words = []
for sent in all_data:
  for word in sent.split(' '):
    if word not in unique_words:
      unique_words.append(word)
print(f'Total unique words present {len(unique_words)}')

num_wrds = min(MAX_VOC, len(unique_words) + 1)


tokenizer = Tokenizer(num_words=num_wrds, filters='')
tokenizer.fit_on_texts(all_data)
article_seq =  tokenizer.texts_to_sequences(articles)
summaries_seq = tokenizer.texts_to_sequences(summaries_pref)
word2idx = tokenizer.word_index

articles_padded =  pad_sequences(article_seq, maxlen=max_len_sent, padding='post')
summary_padded =  pad_sequences(summaries_seq, maxlen=max_len_summary, padding='post')


embedded_mtx = np.zeros((num_wrds, EMBEDDED_DIM))

for word, i in word2idx.items():
  if i < num_wrds:
    embedded_vect = word2Vec.get(word)
    if embedded_vect is not None:
      embedded_mtx[i] = embedded_vect

embedded_lyr = Embedding(
  num_wrds,
  EMBEDDED_DIM,
  weights = [embedded_mtx],
  trainable=False
)


embedded_lyr_2 = Embedding(
  num_wrds,
  EMBEDDED_DIM,
  weights = [embedded_mtx],
  trainable=False
)


ec_input = Input(shape=(max_len_sent,))
enc = embedded_lyr(ec_input)
lstm = Bidirectional(LSTM(LATENT_DIM, return_state=True))
enc_output, en_fh, en_fc, en_bh, en_bc = lstm(enc)
en_h = Concatenate(axis=-1, name='en_h')([en_fh, en_bh])
en_c = Concatenate(axis=-1, name='en_c')([en_fc, en_bc])
# Decoder
de_input = Input(shape=(None,))
de = embedded_lyr_2(de_input)
de_lstm = LSTM(LATENT_DIM*2, return_state=True, return_sequences=True)
de_out, _, _ = de_lstm(de, initial_state=[en_h, en_c])

de_dense = TimeDistributed(Dense(num_wrds, activation='softmax'))
de_output = de_dense(de_out)

model = Model([ec_input, de_input], de_output)
model.summary()

plot_model(
  model,
  to_file='./text_summary_model.png',
  show_shapes=True,
  show_layer_names=True,
  rankdir='TB',
  expand_nested=False,
  dpi=96
)

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

i_h = np.zeros((len(summary_padded) ,LATENT_DIM))

model.fit([articles_padded, summary_padded[:, :-1]], 
summary_padded.reshape(summary_padded.shape[0], summary_padded.shape[1],1)[:,1:],
epochs=10,  batch_size=128)































