

## KÜTÜPHANELER

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------

## VERİ YÜKLEME

# pandasın read_csv modulü csv dosyalarını aktarmamızı sağlıyor.
veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\Churn_Modelling.csv")
print(veri)

#------------------------------------------------------

## BAĞIMLI BAĞIMSIZ DEĞİŞKEN OLUŞTURMA

X = veri.iloc[:,3:13].values 
# ilk 3 sütun tahmin algoritmamız için işe yaramayacağı için ilk 3 sütunu attık
Y = veri.iloc[:,-1:].values 
# tahmin ediceğimiz değeri buraya atadık

#------------------------------------------------------------

## KATEGORİK VERİ DÖNÜŞÜMÜ

from sklearn import preprocessing
# dönüştürme işlemini yapabilmemiz için sklearn kütüphanesindeki preprocessing modulunu import ettik.

le = preprocessing.LabelEncoder()
# preprocessing modülündeki LabelEncoder() sınıfından le isimli bir obje oluşturduk.
# LabelEncoder() sanırsam kategorik olan verileri makine dilinin anlayacağı veri tipine dönüştürmeye yarıyor ama emin değilim

le_1 = preprocessing.LabelEncoder()
# preprocessing modülündeki LabelEncoder() sınıfından le isimli bir obje oluşturduk.
# LabelEncoder() sanırsam kategorik olan verileri makine dilinin anlayacağı veri tipine dönüştürmeye yarıyor ama emin değilim


X[:,1] = le.fit_transform(X[:,1])
# yukarıda şey yaptık X 'in 1. kolonundaki kategorik verileri label encode ettik ve tekrar onu X'in 1. kolonuna eşitledik.
X[:,2] = le_1.fit_transform(X[:,2])
# gene aynısını yaptık böylece kategorik verileri sayısal verilere dönüştürmüş olduk.
# 2. bi le oluşturmamızın sebebi karışıklık olmaması için çünkü 1. 'si yukarıda fit edildi yani öğrendi
# aslında sorun olmuyormuş ama kodun okunabilirliğini arttırabilmek için yaptık.

from sklearn.preprocessing import OneHotEncoder
# her birini ayrı kolonlora bölmek için one hot encoder yapıcaz bi de

from sklearn.compose import ColumnTransformer
# bu columntransformer basitçe birden fazla kolonun aynı anda ayrı ayrı dönüştürülmesini sağlıyor

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])], remainder="passthrough")
# yukarıda bir tane obje tanımladık ColumnTransformer'dan bunu niçine one hot encoder verdik 1 kolon yap deidk ama neden öle
# anlamadım sonra "ohe" ne anlamadım passthrough anlamadım.

X = ohe.fit_transform(X)
# oluşturduğumuz objeye göre fit ve transform ettik
X = X[:,1:]
# 1. kolon dahil sonrasını aldık.


#---------------------------------------------------------

## EĞİTİM VE TEST KÜMESİ OLUŞTURMA

from sklearn.model_selection import train_test_split
# yukarıda sklearn kütüphanesinden model_selection modülünden train_test_split class ını ekledik

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33,random_state=0)
#eğitim ve test kümesi olmak üzere bağımlı ve bağımsız değişkenleri böldük 

#------------------------------------------------------------------

## ÖZNİTELİK ÖLÇEKLEME

from sklearn.preprocessing import StandardScaler
# sklearn kütüphanesinden preprocessing modülünden StandardScaler sınıfını import ettik.

sc=StandardScaler()
# sc değişkenini StandardScaler sınıfından bi obje haline getirdik.

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

# verilerimizi 0-1 aralığına soktuk hepsini 

#-------------------------------------------------------------------

## YAPAY SİNİR AĞI

import keras
# keras import ettik
from keras.models import Sequential
# keras modellerini import ettik
from keras.layers import Dense
# keras katmanları oluşturabilmek için import ettik

#------------------------------------------------------------------

## YAPAY SİNİR AĞI OLUŞTURMA

neural_network = Sequential()
# bunu yaparak ilk sinir ağımızı oluşturmuş olduk

neural_network.add(Dense(6, kernel_initializer='random_uniform', activation = "relu", input_dim = 11 ))
# yukarıda oluşturduğumuz ilk yapay sinir ağımızın içini doldurmaya başladık
# 6 nörondan oluşan gizli katman ekledik Dense(6) diyerek onu vermenin tam bir kuralı yok 
# random_uniform vermemizin sebebi ise yapay sinir ağının neye göre inisilazion ediceğimizdir.
# activation fonksiyonumuzu verdik input_dim ise yapay sinir ağımızın girişinde kaç tane nöron girişi olduğu
# 11 tane bağımsız değişkenimiz olduğu için 11 verdik giriş sayısını

neural_network.add(Dense(6, kernel_initializer='random_uniform', activation = "relu"))
# ikinci bir gizli katman ekledik. Girişimizi verdiğimiz için girişe ihtiyaç duymadık
 
neural_network.add(Dense(1, kernel_initializer='random_uniform', activation = "sigmoid"))
# son olarak çıkış katmanını ekledik çıkış da sadece 1 şeyimiz olduğu için 1 verdik ve activation fonksiyonuna sigmoid verdik.

neural_network.compile(optimizer ='adam',loss = "binary_crossentropy", metrics = ["accuracy"])
# yukarıda oluşturduğumuz yapay sinir ağlarını tek bir şeyde birleştirmek için compile fonksiyonunu kullandık içine parametre olarak
# optimizer = 'adam' verdik optimizer yapay sinir ağlarını nasıl optimize edeceğimizi söylüyor. loss olarak verdiğimiz şey ise fonksyion 
# çıktı alacağımız verilerimiz 1 ve 0 lardan oluştuğu için binary verdik metrics ise metric ne bilmiom ama accuracy verdik

#------------------------------------------------------------------------

## YAPAY SİNİR AĞININ ÖĞRENMESİ

neural_network.fit(X_train, y_train, epochs=50)
# x_train'den y_train öğren dedik. epochs=50 dediğimiz şey ise
# kaç defa çalıştırıp öğreneceği yani her seferinde kendini düzeltiyordu bu yapay sinir ağı 
# işte bunu kaç defa yapacağını verdik. X_train'i büyük olan verdik yani şey olmuş olan standartize edilmiş olan verdik 
# y 'yi ise standartize etmeye gerek yoktu.

neural_pred = neural_network.predict(X_test)
# X_test'den y_testi tahmin et dedik.
# tahmin değerlerini 0 ile 1 arasında olasılık olarak döndürdü

neural_pred = (neural_pred > 0.5)
# yukarıda şey yaptık bu tahmin değerlerini 1 ile 0 arasında tahmin olarak olasılık olarak döndürdüğü için
# biz şey dedik 0.5 'den büyük ise true dönüyor buda neural_pred'deki ilgili yere 1 atıyor küçük ise false döner 0 atar
# bu sayede olasılık deil de direk bırakıp bırakmıyacağını görmüş oluruz.

from sklearn.metrics import confusion_matrix
neural_cm = confusion_matrix(y_test, neural_pred)
# confussion matrix oluştur dedik y_test ile neural_pred arasında

print("\n")
print("neural CM")
print(neural_cm)









