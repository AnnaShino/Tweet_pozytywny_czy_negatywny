# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 14:19:40 2021

@author: pauli
"""

import pandas as pd
import numpy as np
import tweepy as tw
import tensorflow as tf
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import texttable as tt
from wordcloud import WordCloud

#działanie na danych (wczytywanie / podgląd / wstępna analiza)
df = pd.read_csv('./train.csv', index_col=0)
print()
print('Podgląd danych treningowych')
print()
print(df.head())


print()
print('Dystrybucja danych między klasami')
print()
print(df.label.value_counts())

print()

lengths = []
for tweet in df['tweet']:
    lengths.append(len(tweet))
print('Maksymalna długości tweeta: ', max(lengths))
del lengths

#mapa najczęscniej występujacych slow
words = ' '.join([tweet for tweet in df['tweet']])
wordCloud = WordCloud(width=600, height=400).generate(words)
plt.imshow(wordCloud)
plt.savefig('./wyniki/mapa_slow.png')
plt.show()

print()
print('train_test_split')
print()
Train, Test = train_test_split(df, test_size=0.3, random_state=101)
print(Train.head())

#Konwertowanie ramki danych pandy na zestaw danych tensorflow...
def df_to_dataset(dataframe, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('label')
    ds = tf.data.Dataset.from_tensor_slices((dataframe['tweet'], labels))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

train_ds = df_to_dataset(Train)
test_ds = df_to_dataset(Test)

print()
print('Testowanie, czy zestaw danych został utworzony poprawnie')
print()
for text_batch, label_batch in train_ds.take(1):
    for i in range(3):
        print("Tweet", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])


#Tworzenie warstwy wektoryzacji tekstu    
vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=10000, output_mode='int',
                                                                               output_sequence_length=274)
#Tworzenie zbioru danych zawierającego tylko tekst pasujący do warstwy wektoryzacji kodowania
train_text = train_ds.map(lambda x, y: x)

#Dopasowanie warstwy
vectorize_layer.adapt(train_text)

print()
print('Testowanie wektoryzowanego tekstu')
print()
text_batch, label_batch = next(iter(train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print(first_review)
text = tf.expand_dims(first_review, -1)
print(vectorize_layer(text))

def vectorize_text(text, label):
    text = tf.expand_dims(text[0], -1)
    return vectorize_layer(text), label

#Wstępne przetwarzanie zbiorów danych
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Przygotowanie modelu
model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(10000, 20),
    tf.keras.layers.Dense(256),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)])

#Dodanie punktu kontrolnego, aby zapisać model z najlepszymi wynikami i najmniejszym dopasowaniem
#jeżeli model tworzony jest na danych w innym języku niż angielski, należy zmienić w nazwie 'pliki_weryfikacyjne' skrót językowy
checkpoint_val_acc = tf.keras.callbacks.ModelCheckpoint(
        'pliki_weryfikacyjne_en.tf', monitor='val_binary_accuracy', verbose=1, save_best_only=True,
        save_weights_only=False,  save_freq='epoch')

#Kompilowanie modelu
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

print()
print('Trenowanie modelu')
print()
epochs = 10
history = model.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=[checkpoint_val_acc])


#Importowanie modelu testowego
g = df['tweet'][5000:7000]
label = df['label'][5000:7000]
labels = list(label)
tweets = list(g)
tweets

#Tworzenie funkcji, która konwertuje listę tweetów do TF Dataset
def sample_list_to_dataset(dataframe, batch_size=20):
    ds = tf.data.Dataset.from_tensor_slices(dataframe)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

#Działanie na modelu
#wczytywany jest model o danym języku - tu też należy zmienić w przypadku języka innego niż angielki
model_2 = tf.keras.models.load_model('pliki_weryfikacyjne_en.tf')
predict_ds = sample_list_to_dataset(tweets)
score = model_2.predict(predict_ds)

#Otrzymane wartoci
predicted_1 = 0
for i in score:
    if i > 0.5:
        predicted_1+=1        
actual_1 = 0
for i in labels:
    if i > 0:
        actual_1+=1
 
#tworzenie wyników w formie tabeli
print()
table = tt.Texttable()
table.set_cols_align(["c", "c"])
table.set_cols_valign(["t", "i"])
table.add_rows([["Cecha", "Wartosc"],
                ["Przewidywane wartosci negatywne",predicted_1],
                ["Aktualne wartosci negatywne", actual_1],
                ["Przewidywane wartosci pozytywne",(len(score)-predicted_1)],
                ["Aktualne wartosci pozytywne", (len(score)-actual_1)],
                ["Iloć elementów", len(score)],
                ["Dokładnosc w %",(((len(score)-abs(actual_1-predicted_1))/len(score))*100)]])
print (table.draw() + "\n")
        

#tworzenie wyników w formie graficznej
df = pd.DataFrame({'Wartosc': score[:,0]})
def getTextAnalysis(a):
    if a >= 0.5:
        return "Negatywne"
    else:
        return "Pozytywne"
df['Charakter'] = df['Wartosc'].apply(getTextAnalysis)
labels = df.groupby('Charakter').count().index.values
values = df.groupby('Charakter').size().values
plt.bar(labels, values)
plt.suptitle('Ilosc negatywnych i pozytywnych Tweetów')
plt.savefig('./wyniki/pozytywy_negatywy.png')
plt.show()
