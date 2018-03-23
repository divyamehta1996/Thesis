import numpy as np
import pandas as pd
import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import sys
import getopt
import codecs
import time
import os
import csv
import operator
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import KFold

chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']

def tokenize_corpus(path):
  stopWords = stopwords.words("english")
  classes = []
  samples = []
  docs = []
  words = {}
  f = open(path, 'r')
  lines = f.readlines()
  for line in lines:
    raw = line.decode('latin1')
    raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
    tokens = word_tokenize(raw)
    tokens = [w for w in tokens if w not in stopWords]  
    # print tokens
    for t in tokens: 
      try:
            words[t] = words[t]+1
        except:
            words[t] = 1 
    docs.append(tokens)
  return words

def transformData(words):
  sorted_words = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
  i = 1
  vocab = {}
  # key = each[0], word 
  #value = vocab[each[0]] = i, #freq
  for each in sorted_words:
    vocab[each[0]] = i 
    i = i + 1

  # print len(vocab)
  f = open("processedData1.csv", 'r')
  stopWords = stopwords.words("english")
  lines_old = f.readlines()
  lines = lines_old[1:]
  transform_data = []
  max_line = 0
    for line in lines:
      newLine = []
      raw = line.decode('latin1')
      raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
      tokens = word_tokenize(raw)
      tokens = [w for w in tokens if w not in stopWords]
      if len(tokens) > max_line:
        max_line = len(tokens)
      for each in tokens:
        newLine.append(vocab[each])
      transform_data.append(newLine)

    return transform_data


def main():
  words = tokenize_corpus("processedData1.csv")
  transform_data = transformData(words)

    # 3726 by 84 #14748 vocab #or by 57 without stopwords
    trainX = sequence.pad_sequences(transform_data)
    trainY = pd.read_csv("processedLabels1.csv", header = 0, low_memory=False)

    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.4, random_state=0)
  
  # trainX = np.array(X)
  # trainY = np.array(Y)
  # kf = KFold(n_splits=5)
  # for train_index, test_index in kf.split(trainX):
  #   # print("TRAIN:", train_index, "TEST:", test_index)
  #   X_train, X_test = trainX[train_index], trainX[test_index]
  #   y_train, y_test = trainY[train_index], trainY[test_index]

  # create the model
  embedding_vecor_length = 32
  model = Sequential()
  model.add(Embedding(14749, embedding_vecor_length, input_length=57))
  model.add(LSTM(200))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(model.summary())
  model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
  # Final evaluation of the model
  scores = model.evaluate(X_test, y_test, verbose=0)
  print("Accuracy: %.2f%%" % (scores[1]*100))


if __name__ == "__main__":
  main()