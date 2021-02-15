import numpy as np
import pandas as pd
import pickle
import optuna

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder

train = pd.read_json('ate_absita_training.ndjson', lines=True)

print(train.columns)

train.drop(columns=['id_sentence','polarities','aspects_position','aspects'], inplace=True)
print(f'Contains {len(train)} sentences')

train["review_type"] = train["score"].apply(lambda x: "neg" if x < 5 else "pos")

print(f'TRAIN::\n{train.review_type.value_counts()}')

train.drop(columns=['score'], inplace=True)

with open("word2index.pkl", 'rb') as output:
    w2i = pickle.load(output)
with open("embedding_matrix.pkl", 'rb') as output:
    embedding_matrix = pickle.load(output)

def my_text_to_word_sequence(sentence):
    return keras.preprocessing.text.text_to_word_sequence(sentence,
                                                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`\'{|}~\t\n',
                                                          lower=True)

sentences = [my_text_to_word_sequence(sentence) for sentence in train['sentence']]

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

embedded_trainset = tf.convert_to_tensor(embedded_trainset)

one_hot_train = tf.convert_to_tensor(
    OneHotEncoder(sparse=False).fit_transform(
        train.review_type.to_numpy().reshape(-1, 1)
        )
    )

def objective(trial):
    units = trial.suggest_int('units', 40, 140)
    recurrent_dropout = trial.suggest_float('dropout', 0.2, 0.8, step=0.01)

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(max, 300)))
    model.add(keras.layers.Bidirectional(layer=keras.layers.LSTM(units=units,
                                                                 recurrent_dropout=recurrent_dropout,
                                                                 activation='tanh')))
    model.add(keras.layers.Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(0.001),
                  metrics=['accuracy'])

    batch_size = trial.suggest_int('batch_size', 50, 128)
    result = model.fit(embedded_trainset,
                       one_hot_train,
                       epochs=100,
                       batch_size=batch_size,
                       callbacks=[keras.callbacks.EarlyStopping(monitor='loss',
                                                                patience=10)])

    return model.evaluate(embedded_trainset, one_hot_train)[1]

study = optuna.create_study(direction='maximize',storage='sqlite:///models.db', study_name='ATE')
study.optimize(objective, n_trials=200, n_jobs=14)
