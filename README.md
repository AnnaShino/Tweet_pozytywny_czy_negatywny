# Tweet_pozytywny_czy_negatywny

**PRACA Z KODEM**
1) Utwórz folder o nazwie „Tweety” 
2) Umieść w folderze _„Tweety”_ plik _tweety.py_ oraz _trening.py_
3) Utwórz w folderze _„Tweety”_ foldery o nazwach _„klucze”_ oraz _„wyniki”_

**Poniżej przedstawiona struktura folderu użytkowego**

![Struktura folderu](https://github.com/AnnaShino/Tweet_pozytywny_czy_negatywny/blob/main/STRUKTURA%20FOLDERU.png)

4) W folderze „klucze” umieść klucze dostępu do Twitter API (otrzymane podczas rejestracji konta developerskiego Twitter: http://apps.twitter.com/) 
5) Stwórz model treningowy za pośrednictwem kodu _"trening.py"_ - przykładowe dane: https://github.com/shwetachandel/Twitter-Sentiment-Analysis

**Poniżej przedstawiona skuteczność modelu wytrenowanego na powyższych danych**

![Model](https://github.com/AnnaShino/Tweet_pozytywny_czy_negatywny/blob/main/MODEL.png)

6) Po stworzeniu danych treningowych możesz przejść do działania z _"tweety.py"_ (poprzez terminal lub dowolną aplikację np. Spyder)

**W przypadku działania w terminalu, komenda wraz z argumentami to:**  
  * python .\tweety.py [wyszukiwane slowo] [data] [liczba wyszukań] [język modelu]  
  * np. python .\tweety.py thor 2019-12-20 40 en

## WYMAGANE
* pip install numby 
* pip install pandas 
* pip install tensorflow 
* pip install texttable 
* pip install tweepy 
* pip install wordcloud
* pip install silence_tensorflow 


## Ciekawe linki (pomocne przy tworzeniu tych kodów)
1) https://monkeylearn.com/blog/sentiment-analysis-of-twitter/ 
2) https://towardsdatascience.com/twitter-sentiment-analysis-in-python-1bafebe0b566 
3) https://medium.com/better-programming/twitter-sentiment-analysis-15d8892c0082 
4) https://medium.com/@himanshu_23732/tweet-text-analysis-3b7b3e2531 
5) https://www.earthdatascience.org/courses/use-data-open-source-python/intro-to-apis/twitter-data-inpython/
