

import numpy as np
import pandas as pd
import nltk
# nltk doğal dil işlemede sıklıkla kullanılan bir kütüphanedir.

veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\Restaurant_Reviews.csv", error_bad_lines=False)
# veride bazı yerlerde sanırsam hata var bu yüzden yüklerken sıkıntı çıktı ama böyle yükledim böyle yükleyince de verinin tamamını yüklemedi nedenini
# öğren !!!!!!

#-----------------------------------------------------

## VERİMİZİ HAZIRLAMA NLP İÇİN

from nltk.stem.porter import PorterStemmer
# bu kütüphane kelimeleri köklerine ayırmaya yarayan bir kütüphane.
ps = PorterStemmer()
# PorterStemmer sınıfından bir obje oluşturduk.

# STOPWORDS : stopwords denen şey bi duygu ifade etmeyen kelimeler demektir
# yani mesela that, this gibi bunlar duygu ifade etmediği için ve bizde
# anlama yönelik sınıflandırma yapıcağımız için bu kelimeleri atmamız gerek.

nltk.download('stopwords')
# böyle dediğimiz zaman bizim içim stopwords listesi indiricek
# sonra'da dicezki bu kelimeleri gördüğünde alma dicez bu indirdiğimiz liste
# ingilizce için
from nltk.corpus import stopwords
## yukarıda sanırsam nltk'dan stopwords indirdik sonra'da nltk.corpus'dan stopwords import ettik.

import re
# re kütüphanesini noktalama işaretlerini felan temizlemek için kullanabiliriz

derlem = []
# boş bi liste tanımladık çünkü bu listeyi düzelttiğimiz yorumları biriktirmek için kullanıcaz

for i in range(716):
    # yukarıda for döngüsü tanımladık çünkü bu yaptığımız işlemi tüm satırlara yaptırmamız gerek.
    
    veri_1 = re.sub('[^a-zA-Z]',' ',veri["Review"][i])
    # yukarıda dediğimiz şey Review sütunundaki 0. satırdaki şeyi al onun içindeki a-z,A-z
    # harfleri hariç geri kalanları sil yerine boşluk koy dedik ^ bu işaret hariç anlamına geliyor.
    
    veri_1 = veri_1.lower()
    # tüm yazılanları küçük harfe çevirdik
    
    veri_1 = veri_1.split()
    # hepsini listeye çevirdik çünkü kelimelerin sayısı bize lazım
    
    veri_1 = [ps.stem(kelime) for kelime in veri_1 if not kelime in set(stopwords.words('english'))]
    # burda şunu dedik veri_1'imizin içindeki her bir kelime için stopwords mü değilmi diye bak şayet değil ise
    # köklerine ayır ve liste yap dedik.
    
    veri_1 = ' '.join(veri_1)
    # yukarıda yaptığımız işlem ise veri_1 içindeki ayırdığımız stop wordlerden arındırdığımız
    # ve son haline getirdiğimiz liste şeklinde olan veriyi aralarına boşluk koyarak string haline getir dedik.

    derlem.append(veri_1)
    # temizlenmiş herbir yorumu listeye append ettik.
    
#-----------------------------------------------------------------------------

## ÖZNİTELİK ÖLÇEKLENDİRME NLP İÇİN

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2000)
# biz import ettiğimiz şeyden bir tane obje oluşturduk oluşturduğumuz bu objenin
# max_features = 2000 gibi bir parametre vermemizin sebebi en fazla en çok kullanılan 2000 kelimeyi al dedik yoksa
# birsürü kelimeleri alabilirdi bu da rami öldürür.
# bu countvectorizer'a kelimeleri saydırcaz.

X = cv.fit_transform(derlem).toarray() # bu bağımsız değişken
# yukarıda yaptığımız şey ise derlem listesinin içindeki 
# verileri öğren ve dönüştür ve bunları bir array yap dedik cv objesine göre cv objesi de bunları sayıcak 
# en çok kullanılan 2000 kelimeyi gördüğü her yorumdaki kelimede işaretlicek 

Y = veri.iloc[:,1].values
# klasik bağımlı değişkeni çek dedik veri_1 'den 

from numpy import nan

Y = veri.iloc[:,1].replace(nan,0)
# ben burda nan veriler olduğu için o nan verilerin yerine 0 yaz dedim 
# eksik veri tamamladım yani

#-----------------------------------------------------------

## NLP İÇİN MAKİNE ÖĞRENMESİ

from sklearn.model_selection import train_test_split
# yukarıda sklearn kütüphanesinden model_selection modülünden train_test_split class ını ekledik

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=0)
# elimizdeki bağımlı ve bağımsız değişkenleri x ve y 'ye atıcak şekilde train ve test kümelerine böldük.
# train kümesi makineyi eğitmek için test kümeside test için kullanılıcak

from sklearn.naive_bayes import GaussianNB
# biz bu örnekte sınıflandırma algoritmalarından GaussianNB kullancağımız için onu import ettik

gnb = GaussianNB()
# obje oluşturduk

gnb.fit(x_train, y_train)
# test ve train kümesi için ayırdığımız verilerden train için olanlardan gauss'a yöntemine göre
# aralarındaki bağlantıyı bul dedik.

gnb_pred = gnb.predict(x_test)
# burda da artık x_test'den y_testi tahmin et dedik öğrendiğine göre

from sklearn.metrics import confusion_matrix

gnb_cm = confusion_matrix(y_test, gnb_pred)

print("\n")
print("GNB-NLP CM\n")
print(gnb_cm)







 
