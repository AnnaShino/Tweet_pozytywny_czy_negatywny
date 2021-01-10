
import numpy as np
import pandas as pd
import silence_tensorflow.auto
import tensorflow as tf
import sys
import tweepy as tw
from tweepy import OAuthHandler
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
from wordcloud import WordCloud
import texttable as tt
import re
import os

# Argumenty które należy wprowadzić w konsolii podczas wywoływania
try:
    phrase = '#'+sys.argv [1] #->  Wyszukiwany hashtag
except:
    sys.exit("Dane wejciowe muszą się składać z 5 elementów:\n"
                 "Zerowy - python .\zadanie_dodatkowe_tweety.py \n"
                 "Pierwszy - wyszukiwane słowo \n"
                 "Drugi - data w formacie RRRR-MM-DD\n"
                 "Trzeci - liczba wyszukiwanych tweetów\n"
                 "Czwarty - język wykorzystywanego modelu (np. en)\n"
                 "np. python .\zadanie_dodatkowe_tweety.py duda 2019-12-20 40 en \n")
try:
    date_since = sys.argv [2] #-> Data w formacie RRRR-MM-DD
except:
    sys.exit("Dane wejciowe muszą się składać z 5 elementów:\n"
                 "Zerowy - python .\zadanie_dodatkowe_tweety.py \n"
                 "Pierwszy - wyszukiwane słowo \n"
                 "Drugi - data w formacie RRRR-MM-DD\n"
                 "Trzeci - liczba wyszukiwanych tweetów\n"
                 "Czwarty - język wykorzystywanego modelu (np. en)\n"
                 "np. python .\zadanie_dodatkowe_tweety.py duda 2019-12-20 40 en \n")
try:    
    num_elem = int (sys.argv [3]) #-> Liczba tweetów (jeśli 0, to bez limitu)
except:
    sys.exit("Dane wejciowe muszą się składać z 5 elementów:\n"
                 "Zerowy - python .\zadanie_dodatkowe_tweety.py \n"
                 "Pierwszy - wyszukiwane słowo \n"
                 "Drugi - data w formacie RRRR-MM-DD\n"
                 "Trzeci - liczba wyszukiwanych tweetów\n"
                 "Czwarty - język wykorzystywanego modelu (np. en)\n"
                 "np. python .\zadanie_dodatkowe_tweety.py duda 2019-12-20 40 en \n")
try:
    model_tr = 'pliki_weryfikacyjne_'+ sys.argv [4] +'.tf'  #-> wybór języka zastosowanego modelu
except:
    sys.exit("Dane wejciowe muszą się składać z 5 elementów:\n"
                 "Zerowy - python .\zadanie_dodatkowe_tweety.py \n"
                 "Pierwszy - wyszukiwane słowo \n"
                 "Drugi - data w formacie RRRR-MM-DD\n"
                 "Trzeci - liczba wyszukiwanych tweetów\n"
                 "Czwarty - język wykorzystywanego modelu (np. en)\n"
                 "np. python .\zadanie_dodatkowe_tweety.py duda 2019-12-20 40 en \n")

#Kod w przypadku bezporedniego działania na kodzie
#phrase = '#thor'
#date_since = "2020-12-01"
#num_elem = 1000
#model_tr = 'pliki_weryfikacyjne_en.tf'

#Import kluczy dostępu
keys = pd.read_csv('./klucze/dostep.csv', names=['KeyName','Key'])
consumer_key= keys['Key'][0]
consumer_secret=  keys['Key'][1]
access_token=  keys['Key'][2]
access_token_secret=  keys['Key'][3]

#drugi sposób to bezporednie zamieszczenie kluczy
#consumer_key= 'tu_wpisz_swoj_kod'
#consumer_secret= 'tu_wpisz_swoj_kod'
#access_token= 'tu_wpisz_swoj_kod'
#access_token_secret= 'tu_wpisz_swoj_kod'

#Zastosowanie kluczy dostepu do logowania w Twitter API
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

print('\n PROSZĘ CZEKAĆ - PROCES MOŻE CHWILĘ TRWAĆ!!! \n')

#pobieranie zdefiniowanej ilosci tweetów
if num_elem>0:
    tweets = tw.Cursor(api.search,
                  q=phrase,
                  lang="en",
                  since=date_since).items(num_elem)
else:
    tweets = tw.Cursor(api.search,
                  q=phrase,
                  lang="en",
                  since=date_since).items()
tweets
        

#zastępowanie nazw użytkowników przez „@user”
tweets_list = []

for tweet in tweets:
    tweet_text = tweet.text
    text_list = tweet_text.split()
    for idx, item in enumerate(text_list):
        if '@' in item:
            text_list[idx] = '@user'

    text = ' '.join([str(elem) for elem in text_list])
    tweets_list.append(text)

#kończenie procesu w przypadku nie znalezienia tresci o zadanym hastagu
if len(tweets_list)==0:
    sys.exit('Nie znaleziono tresci o zadanym #')
    
#zapisywanie tresci tweetow do pliku
tweety = pd.DataFrame(data=tweets_list)
tweety.to_csv('./wyniki/'+phrase+'lista_tweetow.csv')
#print(tweety.head)
#print()

#mapa najczęsciej padajacych slow    
words = ' '.join([tweet for tweet in tweets_list])
wordCloud = WordCloud(width=600, height=400).generate(words)
plt.imshow(wordCloud)
plt.savefig('./wyniki/'+phrase+'_mapa_słów.png')
plt.show()

#Tworzenie funkcji, która konwertuje listę tweetów do TF Dataset
def sample_list_to_dataset(dataframe, batch_size=20):
    ds = tf.data.Dataset.from_tensor_slices(dataframe)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

#Działanie na modelu
m_1 = tf.keras.models.load_model(model_tr)

#Tworzenie zestawu danych prognozy
predict_ds = sample_list_to_dataset(tweets_list)

#Przewidywania
score = m_1.predict(predict_ds)
y_pred = []
for i in score:
    if i >=0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

#wyniki koncowe        
negative_count = sum(y_pred)
negative_percent = (negative_count/len(y_pred))*100
positive_percent = ((len(y_pred)-negative_count)/len(y_pred))*100

#tworzenie wyników w formie tabeli
print()
table = tt.Texttable()
table.set_cols_align(["c", "c"])
table.set_cols_valign(["t", "i"])
table.add_rows([["Cecha", "Wartosc"],
                ["Zastosowany #hashtag", phrase],
                ["Data od której rozpoczęto przeszukiwanie",date_since],
                ["Przetworzona ilosc tweetów", len(y_pred)],
                ["Wartosci negatywne w %",negative_percent],
                ["Wartosci pozytywne w %",positive_percent]])
print (table.draw() + "\n")
print()
print('Wyniki w formie graficznej, mapę słów oraz plik csv z wykorzystanymi tweetami znajdziesz w folderze ./wyniki/')

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
plt.suptitle('Ilosc negatywnych i pozytywnych Tweetów zawierających '+phrase)
plt.savefig('./wyniki/'+phrase+'_pozytywy_negatywy.png')
plt.show()
