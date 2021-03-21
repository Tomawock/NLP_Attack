import pandas as pd
import optuna
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from tensorflow import keras

"""## CONFRONTO TRAINING SET E TEST SET"""

dev = pd.read_csv('dev.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')
train = pd.read_csv('train.tsv', sep='\t')


print(f'TRAIN:\n{train.label.value_counts()}')
print(f'TEST:\n{test.label.value_counts()}')

"""## FASE PRELIMINARE MODELLO"""

with open("word2index.pkl", 'rb') as output:
  w2i = pickle.load(output)
with open("embedding_matrix.pkl", 'rb') as output:
  embedding_matrix = pickle.load(output)

categories = [['DDI-false', 'DDI-mechanism', 'DDI-effect', 'DDI-advise','DDI-int']]
five_hot_train = OneHotEncoder(sparse=False, categories=categories).fit_transform(
  train.label.to_numpy().reshape(-1, 1)
)

five_hot_dev = OneHotEncoder(sparse=False, categories=categories).fit_transform(
  dev.label.to_numpy().reshape(-1, 1)
)

my_text_to_word_sequence = lambda sen: keras.preprocessing.text.text_to_word_sequence(sen,
                                                                                      filters='!"#&()*+,-./:;<=>?[\\]^_`\'{|}~\t\n',
                                                                                      lower=True)

sentences_train = [my_text_to_word_sequence(sentence) for sentence in train['sentence']]
sentences_dev = [my_text_to_word_sequence(sentence) for sentence in dev['sentence']]

max_index, max = (-1, -1)
for i, sentence in enumerate(sentences_train):
  max_index, max = (i, len(sentence)) if len(sentence) > max else (max_index, max)

for i, sentence in enumerate(sentences_dev):
  max_index, max = (i, len(sentence)) if len(sentence) > max else (max_index, max)

print(f'Il massimo è {max}')

embedded_trainset = np.zeros(shape=(len(sentences_train), max, 300))
for i, sentence in enumerate(sentences_train):
  for j, word in enumerate(sentence):
    try:
      embedded_trainset[i, j, :] = embedding_matrix[w2i[word]]
    except KeyError:
      pass

embedded_devset = np.zeros(shape=(len(sentences_dev), max, 300))
for i, sentence in enumerate(sentences_dev):
  for j, word in enumerate(sentence):
    try:
      embedded_devset[i, j, :] = embedding_matrix[w2i[word]]
    except KeyError:
      pass

"""## OPTUNA"""

def metrics_2(t_labels, t_predictions):
  numeric_labels = np.argmax(t_labels, axis=1)
  numeric_predictions = np.argmax(t_predictions, axis=1)
  matrix = confusion_matrix(numeric_labels, numeric_predictions)
  print(matrix)
  FP = (matrix.sum(axis=0) - np.diag(matrix))[1:]
  FN = (matrix.sum(axis=1) - np.diag(matrix))[1:]
  TP = (np.diag(matrix))[1:]
  overall_fp = np.sum(FP)
  overall_fn = np.sum(FN)
  overall_tp = np.sum(TP)
  overall_precision = overall_tp / (overall_tp + overall_fp)
  overall_recall = overall_tp / (overall_tp + overall_fn)
  overall_f_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
  return overall_f_score

def objective(trial):
    units = trial.suggest_int('units', 40, 140)
    recurrent_dropout = trial.suggest_float('dropout', 0.2, 0.8, step=0.01)

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(max, 300)))
    model.add(keras.layers.Bidirectional(layer=keras.layers.LSTM(units=units,
                                                                 recurrent_dropout=recurrent_dropout,
                                                                 activation='tanh')))

    model.add(keras.layers.Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    batch_size = trial.suggest_int('batch_size', 50, 128)
    result = model.fit(embedded_trainset,
                       five_hot_train,
                       epochs=100,
                       batch_size=batch_size,
                       callbacks=[keras.callbacks.EarlyStopping(monitor='loss',
                                                                patience=10)])

    return metrics_2(model.predict(embedded_devset), five_hot_dev)

study = optuna.create_study(direction='maximize',storage="sqlite:///models.db", study_name="DDI")
study.optimize(objective, n_trials=300, n_jobs=-1)
