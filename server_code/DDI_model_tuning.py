import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
import optuna

def my_text_to_word_sequence(sentence):
    return keras.preprocessing.text.text_to_word_sequence(sentence,
                                                          filters='!"#%&()*+,-./:;<=>?[\\]^_`\'{|}~\t\n',
                                                          lower=True)

train = pd.read_csv('train.tsv', sep='\t')

test = pd.read_csv('test.tsv', sep='\t')

with open("word2index.pkl", 'rb') as output:
    w2i = pickle.load(output)
with open("embedding_matrix.pkl", 'rb') as output:
    embedding_matrix = pickle.load(output)

sentences = [my_text_to_word_sequence(sentence) for sentence in train['sentence']]
sentences_test = [my_text_to_word_sequence(sentence) for sentence in test['sentence']]

max_index, max = (-1, -1)
for i, sentence in enumerate(sentences):
    max_index, max = (i, len(sentence)) if len(sentence) > max else (max_index, max)

for i, sentence in enumerate(sentences_test):
    max_index, max = (i, len(sentence)) if len(sentence) > max else (max_index, max)

embedded_trainset = np.zeros(shape=(len(sentences), max, 300))
for i, sentence in enumerate(sentences):
    for j, word in enumerate(sentence):
        try:
            embedded_trainset[i, j, :] = embedding_matrix[w2i[word]]
        except KeyError:
            pass

embedded_testset = np.zeros(shape=(len(sentences_test), max, 300))
for i, sentence in enumerate(sentences_test):
    for j, word in enumerate(sentence):
        try:
            embedded_testset[i, j, :] = embedding_matrix[w2i[word]]
        except KeyError:
            pass

five_hot_train = OneHotEncoder(sparse=False).fit_transform(
  train.label.to_numpy().reshape(-1, 1)
)

five_hot_test = OneHotEncoder(sparse=False).fit_transform(
  test.label.to_numpy().reshape(-1, 1)
)

def objective(trial):
    activation = trial.suggest_categorical('activation', ['tanh', 'relu', 'gelu'])
    units = trial.suggest_int('units', 16, 70)
    recurrent_dropout = trial.suggest_float('dropout', 0.2, 0.8, step=0.01)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01, step=0.0001)

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(max, 300)))
    model.add(keras.layers.LSTM(units=units,
                                recurrent_dropout=recurrent_dropout,
                                activation=activation))
    model.add(keras.layers.Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    batch_size = trial.suggest_int('batch_size', 25, 50)
    epochs = trial.suggest_int('epochs', 5, 15)
    result = model.fit(embedded_trainset,
                       five_hot_train,
                       epochs=epochs,
                       batch_size=batch_size)

    return model.evaluate(embedded_testset, five_hot_test)[1]

study = optuna.create_study(direction='maximize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=1800, n_jobs=30)

with open('optuna_result.pkl', 'wb') as result:
    pickle.dump(study.trials_dataframe(), result)

