# nostri import
import random
import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import tensorflow_text #necessaria per hub.load
from sklearn.preprocessing import OneHotEncoder

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

df = pd.concat([pd.read_csv('../amazon_sinonimi.csv'), pd.read_csv('../reducedReviews.csv')], ignore_index=True)
df = df.sample(frac=1, random_state=RANDOM_SEED)

print(f"COLONNE:{df.columns}")
print(f"DIMENSIONE:{df.shape}")

df['review'] = df['Summary']+df['Text']
del df['Summary']
del df['Text']
df.review.fillna("",inplace = True)
df.head()

"""Creazione review type al posto dello score, per avere tutte le frasi con valutazione minore di 4 come negative e le rimanneti come positive"""

df["review_type"] = df["Score"].apply(lambda x: "negative" if x < 4 else "positive")

del df['Score']

"""Bilanciamento delle sentence positive con quelle negative, in modo da avere un dataset bilanciato, fatto 50 50 poiche si hanno a disposizione molti dati """

positive_reviews = df[df.review_type == "positive"]
negative_reviews = df[df.review_type == "negative"]

positive_df = positive_reviews.sample(n=min(len(negative_reviews), len(positive_reviews)), random_state=RANDOM_SEED)
negative_df = negative_reviews.sample(n=min(len(negative_reviews), len(positive_reviews)), random_state=RANDOM_SEED)

review_df = positive_df.append(negative_df).reset_index(drop=True)
review_df.shape
print(review_df.columns)

"""## **Modello unico**

Creazione one hot encoding per il review_type
"""

type_one_hot = OneHotEncoder(sparse=False).fit_transform(
  review_df.review_type.to_numpy().reshape(-1, 1)
)

"""Creazione test e train set per il modello, generato tramite **train_test_split**,  70/30 con seed identico a quello usato per il campionamemto dei dati per il bilancaiamento del dataset"""

train_reviews, test_reviews, y_train, y_test =\
  train_test_split(
    review_df.review,
    type_one_hot,
    test_size=.3,
    random_state=RANDOM_SEED
  )

print(train_reviews.shape)
print(test_reviews.shape)

"""Caricamento dell'embedding e creazione della funzione usata dal modello per usarlo all'interno dello stesso e non come preprocessing del dataset"""

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

model = keras.Sequential()

model.add(keras.layers.Input(shape=(1,), dtype=tf.string))
model.add(keras.layers.Lambda(lambda x: tf.squeeze(tf.cast(x, tf.string))))
model.add(hub.KerasLayer(handle=embed,output_shape=512)) # pre trained Convolutional Neural Net.
model.add(keras.layers.Dense(units=256, activation='relu'))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001), metrics=['accuracy'])
model.summary()

history = model.fit(
    train_reviews, y_train,
    epochs=15,
    batch_size=16,
    validation_split=0.1,
    verbose=1,
    shuffle=True
)

"""Salva il modello per poterlo utilizzare per i vari test

"""

model.save_weights('model.h5')

print("TUTTO OK XD")
