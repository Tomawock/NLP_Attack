import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
import optuna

def my_text_to_word_sequence(sentence):
    return keras.preprocessing.text.text_to_word_sequence(sentence,
                                                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`\'{|}~\t\n',
                                                          lower=True)

test = pd.read_json('ate_absita_gold.ndjson', lines=True)

train = pd.read_json('ate_absita_training.ndjson', lines=True)

with open("word2index.pkl", 'rb') as output:
    w2i = pickle.load(output)
with open("embedding_matrix.pkl", 'rb') as output:
    embedding_matrix = pickle.load(output)

train.drop(columns=['id_sentence','polarities','aspects_position','aspects'], inplace=True)
test.drop(columns=['id_sentence','polarities','aspects_position','aspects'], inplace=True)

train["review_type"] = train["score"].apply(lambda x: "neg" if x < 5 else "pos")
test["review_type"] = test["score"].apply(lambda x: "neg" if x < 5 else "pos")

train.drop(columns=['score'], inplace=True)
test.drop(columns=['score'], inplace=True)

sentences = [my_text_to_word_sequence(sentence) for sentence in train['sentence']]
sentences2 = [my_text_to_word_sequence(sentence) for sentence in test['sentence']]

max_index, max = (-1, -1)
for i, sentence in enumerate(sentences):
    max_index, max = (i, len(sentence)) if len(sentence) > max else (max_index, max)

embedded_trainset = np.zeros(shape=(len(sentences), max, 300))
for i, sentence in enumerate(sentences):
    for j, word in enumerate(sentence):
        try:
            embedded_trainset[i, j, :] = embedding_matrix[w2i[word]]
        except KeyError:
            pass

embedded_testset = np.zeros(shape=(len(sentences2), max, 300))
for i, sentence in enumerate(sentences2):
    for j, word in enumerate(sentence):
        try:
            embedded_testset[i, j, :] = embedding_matrix[w2i[word]]
        except KeyError:
            pass

one_hot_train = OneHotEncoder(sparse=False).fit_transform(
  train.review_type.to_numpy().reshape(-1, 1)
)

one_hot_test = OneHotEncoder(sparse=False).fit_transform(
  test.review_type.to_numpy().reshape(-1, 1)
)

def objective(trial):
    rec_act = trial.suggest_categorical('recurrent_act', ['sigmoid', 'tanh', 'relu', 'gelu'])
    units = trial.suggest_int('units', 16, 70)
    recurrent_dropout = trial.suggest_loguniform('dropout', 0.2, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 0.004, 0.008, step=0.0001)

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(max, 300)))
    model.add(keras.layers.LSTM(units=units,
                                recurrent_dropout=recurrent_dropout,
                                activation='tanh',
				recurrent_activation=rec_act))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    batch_size = trial.suggest_int('batch_size', 10, 35)
    epochs = trial.suggest_int('epochs', 13, 27)
    result = model.fit(embedded_trainset,
                       one_hot_train,
                       epochs=epochs,
                       batch_size=batch_size)

    return model.evaluate(embedded_testset, one_hot_test)[1]

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3000, n_jobs=-1)

with open('optuna_result_2.pkl', 'wb') as result:
    pickle.dump(study.trials_dataframe(), result)
