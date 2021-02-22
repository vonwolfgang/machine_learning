

#--------------------------------------------------------

## KÜTÜPHANELER

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------

## VERİ YÜKLEME

# pandasın read_csv modulü csv dosyalarını aktarmamızı sağlıyor.
veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\Social_Network_Ads.csv")
print(veri)

#------------------------------------------------------

## VERİ ÖNİŞLEME

X = veri.iloc[:,[2,3]].values 
# bağımsız değişkenler id felan attık.
y = veri.iloc[:,4].values # bağımlı değişkenler  

#----------------------------------------------------------------

## EĞİTİM VE TEST KÜMESİ OLUŞTURMA

from sklearn.model_selection import train_test_split
# yukarıda sklearn kütüphanesinden model_selection modülünden train_test_split class ını ekledik

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)
# elimizdeki bağımlı ve bağımsız değişkenleri x ve y 'ye atıcak şekilde train ve test kümelerine böldük.
# train kümesi makineyi eğitmek için test kümeside test için kullanılıcak

#------------------------------------------------------------------

## ÖZNİTELİK ÖLÇEKLEME

from sklearn.preprocessing import StandardScaler
# sklearn kütüphanesinden preprocessing sınıfından StandartScaler metodunu import ettik.

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
# fit içine koyduğumuz veriden öğren demek transform ise öğrendiğin şeyi kullan demek.

X_test=sc.transform(x_test)
# burda ise sadece transform ettik şöyle dedik yeniden öğrenme sadece kullan dedik. Yukarıda öğrendiğini 
# burda kullan dedik.

#----------------------------------------------------------------

## SVC (SUPPORT VECTOR CLASSİFE)

from sklearn.svm import SVC

svc = SVC(kernel="rbf", random_state=0)
# burda import ettiğimiz şeyden bir obje oluşturduk oluşturduk objemizden kernel fonksiyon olarak 
# linear seçtik başka şeylerde seçilebilir. Bu işte farklı modeller felan oluşturuyor.

svc.fit(X_train, y_train)
# X_train ile y_train arasındaki bağlantıyı öğren dedik.

svc_pred = svc.predict(X_test)
# öğrendiğin şekilde X_test tahmin et dedik.

from sklearn.metrics import confusion_matrix

svc_cm = confusion_matrix(y_test, svc_pred)
# confusion matrix objesi tanımladık, y_test ile svc_pred arasında oluşacak. 
print("SVC CM \n")
print(svc_cm,"\n")
# o objeyi bastırdık


#-----------------------------------------------------------------

## K_FOLD VALİDATİON

from sklearn.model_selection import cross_val_score

cvs = cross_val_score(estimator = svc, X = X_train, y= y_train, cv = 4)
# yukarıda bi tane cross val tanımladık parametre olarka estimator'e svc yani svm objesini verdik nelerden fit edildiğini verdik bi de
# kaç kat yapacağını verdik biz 4 kat yap dedik.

print(cvs.mean())
# ortalamasını aldırdık
# bu kullandığımız şey modelin başarısını ölçüyor.


















