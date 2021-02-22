

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

from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

# xgboost 'da bir sınıflandırma algoritması hızlı ve iyi çalışan bir algoritma

from sklearn.metrics import confusion_matrix

xgb_cm = confusion_matrix(y_test, xgb_pred)
# confusion matrix oluşturduk

print("XGB CM")
print(xgb_cm)






