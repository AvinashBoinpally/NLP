import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam, SGD





EMBEDDING_DIM = 100
MAX_VOCAB_SIZE = 3000
LATENT_DIM = 25
BATCH_SIZE = 128
EPOCHS = 2000

input_text = []
target_text = []

with open('robert.txt', encoding='utf-8') as f:
  for line in f:
    line = line.rstrip()
    if not line:
      continue
    input_line = '<sos> ' +  line
    target_line = line + ' <eos>'
    input_text.append(input_line)
    target_text.append(target_line)


word2Vec = {}
with open(os.path.join(f'../generic_data/glove.6B/glove.6B.{EMBEDDING_DIM}d.txt'), encoding='utf-8') as f:
  for line in f:
    row = line.split(" ")
    word2Vec[row[0]] = np.asarray(row[1:], dtype='float32')
print(f'lenght of the words {len(word2Vec)}')

all_text = input_text + target_text

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer.fit_on_texts(all_text)
input_sequences = tokenizer.texts_to_sequences(input_text)
target_sequences= tokenizer.texts_to_sequences(target_text)

max_sentence_length = max([len(line) for line in input_sequences])
print(f'Max sentence length {max_sentence_length}')

input_sequences = pad_sequences(input_sequences, maxlen=max_sentence_length,padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_sentence_length, padding='post')

wordidx = tokenizer.word_index
assert('<sos>' in wordidx)
assert('<eos>' in wordidx)

num_words = min(MAX_VOCAB_SIZE, len(wordidx) + 1)
embedding_mtx = np.zeros((num_words, EMBEDDING_DIM))

for word, i in wordidx.items():
  if i < num_words:
    word_vec = word2Vec.get(word)
    if word_vec is not None:
      embedding_mtx[i] = word_vec


target_one_hot = np.zeros((len(target_sequences), max_sentence_length, num_words))

for i, target_seq in enumerate(target_sequences):
  for j, word in enumerate(target_seq):
    target_one_hot[i, j, word] = 1


embedded_lyr = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights = [embedding_mtx]
)

_input = Input(shape=(max_sentence_length,))
input_h = Input(shape=(LATENT_DIM,))
input_c = Input(shape=(LATENT_DIM,))
x = embedded_lyr(_input)
lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
x, _, _ = lstm(x, initial_state=[input_h, input_c])
dense = Dense(num_words, activation='softmax')
output = dense(x)

model = Model([_input, input_h, input_c], output)

model.compile(
  loss='categorical_crossentropy',
  optimizer=Adam(lr=0.01),
  metrics=['accuracy']
)

h_c_states = np.zeros((len(input_sequences) ,LATENT_DIM))

res = model.fit(
  [input_sequences, h_c_states, h_c_states],
  target_one_hot,
  batch_size= BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=0.2
)

input_2 = Input(shape=(1,))
x = embedded_lyr(input_2)
x, h, c = lstm(x, initial_state=[input_h, input_c])
output = dense(x)

sample_model = Model([input_2, input_h, input_c], [output, h, c])

indx2word = {v:k for k, v in wordidx.items()}

def sample_line():
  in_input = np.array([[wordidx['<sos>']]])
  h = np.zeros((1, LATENT_DIM))
  c = np.zeros((1, LATENT_DIM))
  eos = wordidx['<eos>']
  output_sentences = []
  for _ in range(max_sentence_length):
    o, h, c = sample_model.predict([in_input, h, c])
    probs = o[0,0]
    if np.argmax(probs) == 0:
      print('Printing <sos>')
    probs[0] = 0

    probs /= np.sum(probs)
    idx = np.random.choice(len(probs), p=probs)
    if idx == eos:
      break

    output_sentences.append(indx2word.get(idx))
    in_input[0,0] = idx
  return ' '.join(output_sentences)


for _ in range(4):
  sample_line() 

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








