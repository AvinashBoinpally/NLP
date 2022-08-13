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