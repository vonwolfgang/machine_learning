## KÜTÜPHANELER

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------

## VERİ YÜKLEME

# pandasın read_csv modulü csv dosyalarını aktarmamızı sağlıyor.
veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\Wine.csv")
print(veri)

#--------------------------------------------------  

## BAĞIMLI BAĞIMSIZ DEĞİŞKEN OLUŞTURMA

X = veri.iloc[:,0:13].values 
# bağımsız değişkenleri aldık
Y = veri.iloc[:,-1:].values 
# tahmin ediceğimiz değeri buraya atadık

#------------------------------------------------------------

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

## PCA İLE BOYUT İNDİRGEME ALGORİTMASI

from sklearn.decomposition import PCA
# yukarıda PCA sınıfını import ettik. Bu PCA sınıfı 
# şu işe yarıyor örneğin şuan elimizde 13 tane bağımsız değişken kolonu var
# bu 13 kolonu belli bi sayıdaki kolona indiricek ve onlarla tahmin algoritması yapıcak

pca = PCA(n_components=2)
# 2 boyutlu bir obje oluşturduk.

X_train_pca = pca.fit_transform(X_train)
# yukarıda pca objesi fit ile hem öğrenicek sonra da transform ile de uygulıcak
X_test_pca = pca.transform(X_test)
# bu sefer fit ettirmedik çünkü yukarıda öğrendiği fit değerlerini kullan dedik 
# yeniden fit ederse başka şekillerde öğrenme yapabilir.

#-----------------------------------------------------------------

## PCA İLE BOYUTU İNDİRGENEN VERİYİ KULLANMA TAHMİN ALGORİTMASINDA

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# PCA dönüşümünden önce

logr = LogisticRegression(random_state= 0)
# obje tanımladık her seferinde farklı bir logistic regression kullanmasın diye random state'ini 0 verdik.
logr.fit(X_train,y_train)
# pca'den öncekine göre öğren dedik.
logr_pred = logr.predict(X_test)
# pca uygulanmamış şekilde öğrendiğine göre tahmin yapıcak
logr_cm = confusion_matrix(y_test, logr_pred)
# pca'den öncesine göre yapılan tahminin confusion matrix'ini oluşturduk.
print("PCA Öncesi")
print(logr_cm)
print("\n")


# PCA dönüşümünden sonra

logr_pca = LogisticRegression(random_state= 0)
# yeni bir obje tanımladık çünkü farklı verilerle eğiticez bu sefer, her seferinde farklı bir logistic regression kullanmasın diye random state'ini 0 verdik.
logr_pca.fit(X_train_pca,y_train)
# pca'den sonrakine göre öğren dedik. 
logr_pred_pca = logr_pca.predict(X_test_pca)
# pca uygulanmış şekilde öğrendiğine göre tahmin yapıcak
logr_cm_pca = confusion_matrix(y_test, logr_pred_pca)
# pca'den sonrasına göre yapılan tahminin confusion matrix'ini oluşturduk.
print("PCA Sonrası")
print(logr_cm_pca)
print("\n")


logr_cm_2 = confusion_matrix(logr_pred, logr_pred_pca)
# buda pca öncesi ve sonrası olan tahminler arasında confusion matrix oluşturuyor.
print("PCA Öncesi ve Sonrası arasında")
print(logr_cm_2)
print("\n")

#-------------------------------------------------------------------

## LDA İLE BOYUT İNDİRGEME ALGORİTMASI

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
# obje oluşturduk 2 kolona indirgicek şekilde oluşturduk.

X_train_lda = lda.fit_transform(X_train, y_train)
# lda algoritması çalışırken sınıflara bölmesi gerekiyor bu yüzden biz lda'yi eğitirken hem X_train hem y_train verdik öğrensin sınıflara bölsün ona göre kolonlora bölsün diye
X_test_lda = lda.transform(X_test)
# yeniden fit etmedik saten yeni sınıflar oluşturdu çünkü direk transform ettirdik.

#---------------------------------------------------------------------

## LDA İLE BOYUTU İNDİRGENEN VERİYİ KULLANMA

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# LDA dönüşümünden önce

logr_1 = LogisticRegression(random_state= 0)
# obje tanımladık her seferinde farklı bir logistic regression kullanmasın diye random state'ini 0 verdik.
logr_1.fit(X_train,y_train)
# LDA'den öncekine göre öğren dedik.
logr_pred_1 = logr_1.predict(X_test)
# LDA uygulanmamış şekilde öğrendiğine göre tahmin yapıcak
logr_cm_1 = confusion_matrix(y_test,logr_pred_1)
# LDA'den öncesine göre yapılan tahminin confusion matrix'ini oluşturduk.
print("LDA Öncesi")
print(logr_cm_1)
print("\n")


# LDA dönüşümünden sonra

logr_lda = LogisticRegression(random_state= 0)
# yeni bir obje tanımladık çünkü farklı verilerle eğiticez bu sefer, her seferinde farklı bir logistic regression kullanmasın diye random state'ini 0 verdik.
logr_lda.fit(X_train_lda,y_train)
# LDA'den sonrakine göre öğren dedik. 
logr_pred_lda = logr_lda.predict(X_test_lda)
# LDA uygulanmış şekilde öğrendiğine göre tahmin yapıcak
logr_cm_lda = confusion_matrix(y_test,logr_pred_lda)
# LDA sonrasına göre yapılan tahminin confusion matrix'ini oluşturduk.
print("LDA Sonrası")
print(logr_cm_lda)
print("\n")
























