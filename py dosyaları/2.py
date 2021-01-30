


##KÜTÜPHANELER

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#----------------------------------------------------

#VERİYİ YÜKLEME

veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\satislar.csv")
print(veri)

#------------------------------------------------------

## VERİ ÖNİŞLEME

aylar = veri[["Aylar"]]
print(aylar)

satislar = veri[["Satislar"]]
print(satislar)

#------------------------------------------------------

## EĞİTİM VE TEST KÜMESİ OLUŞTURMA

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

#--------------------------------------------------------

## ÖZNİTELİK ÖLÇEKLEME

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

#-------------------------------------------------------

## DOĞRUSAL REGRESYON

#NOT:Doğrusal regresyon bir bağımlı değişkenin bir bağımsız değişkene göre durumudur.

Y_train =sc.fit_transform(y_train)
Y_test =sc.fit_transform(y_test)
#bu verileride standartize ederek aynı dünyaya indirgedik.

#----------------------------------------------------------

## MODEL OLUŞTURMA

from sklearn.linear_model import LinearRegression
# doğrusal regresyon için sklearn kütüphanesinden linear_model modülünden LinearRegression sınıfını import ettik

lr = LinearRegression()
#LinearRegression() sınıfından bir lr objesi oluşturduk.
"""
lr.fit(X_test,Y_test)
# bu yaptığımızla X_train den Y_train i tahmin etmek için model oluşturma yaptık
# burda bir nevi neyden neyi tahmin ediceğini söyledik.

#NOT: EĞER Bİ ŞEY HAKKINDA BİLGİ ALMAK İSTİYORSAN FAREYİ TIKLA VE CTRL+I YAP

prediction = lr.predict(X_test)
#burda lr.predict() ile X_train den Y_traini tahmin ettirdik.
"""
## COMMAND SATIRI İÇİNDE OLAN YERİ BAŞKA Bİ ŞEKİLDE YAPTIK 

lr.fit(x_test,y_test)
# bu sefer stadartize etmeden yapıcaz önce makineye neyden neyi tahmin etmesi için modellemesi gerektiğini söylüyoruz

prediction = lr.predict(x_test)
# şimdide x_test den y_testi tahmint et diyoruz.

#------------------------------------------------

## VERİLERİ GÖRSELLEŞTİRME

x_train = x_train.sort_index()
y_train = y_train.sort_index()
# yukarıda yaptığımız şey verileri index numaralarına göre sıraya sokmak oldu.

plt.plot(x_train,y_train)
# burda yaptığımız şey ise matplotlib kütüphanesinden pyplot modülünü plt olarak kaydettik
# ve sonra bu plt olarak kaydettiğimiz şeyin plot sınıfını kullandık burda iki değişken verdik(x,y) 
# ve x,y grafiği oluşturdu. 
#NOT: matplotlib genel olarak görselleştirmek için kullanılır.

plt.plot(x_train,lr.predict(x_train))
# burda ise x_train'den tahmin edilmiş yani y_train halinin grafik hali 
# ama yukarda yaptığımız gerçek verilerin grafiğide aynı grafikte.

plt.title("aylara göre satış")
# bu method grafiğin başına başlık ekler.

plt.xlabel("aylar")
plt.ylabel("satışlar")
# bu methodlar sayesinde x eksenine "aylar" y eksenine "satışlar" başlığını koyduk  

#---------------------------------------------------




















