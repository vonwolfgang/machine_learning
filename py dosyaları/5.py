

#--------------------------------------------------------

## KÜTÜPHANELER

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------

## VERİ YÜKLEME

# pandasın read_csv modulü csv dosyalarını aktarmamızı sağlıyor.
veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\data.csv")
print(veri)

#------------------------------------------------------

## VERİ ÖNİŞLEME

x = veri.iloc[:,1:4].values # bağımsız değişkenler
y = veri.iloc[:,-1:].values # bağımlı değişkenler  

# biz bu uygulamada boy kilo ve yaştan cinsiyet tahmini yapacağımız için x'e boy kilo ve yaş kolonlarını atadık
# y'ye ise cinsiyet kolonunu atadık. Bunları atarken values şeklinde atadık.

#----------------------------------------------------------------

## EĞİTİM VE TEST KÜMESİ OLUŞTURMA

from sklearn.model_selection import train_test_split
# yukarıda sklearn kütüphanesinden model_selection modülünden train_test_split class ını ekledik

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)
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

## SINIFLANDIRMA ALGORİTMALARI BAŞLANGIÇ

#----------------------------------------------------------------

## LOGİSTİC REGRESSİON 

from sklearn.linear_model import LogisticRegression
# sklearn kütüphanesinden linear_model şeysinden LogisticRegression şeysini import ettik

logr = LogisticRegression(random_state=0)
# LogisticRegression sınıfından bir tane logr objesi oluşturduk.

logr.fit(X_train, y_train)
# biz burda logr objesi yöntemiyle X_train ve y_train arasındaki bağlantıyı öğren dedik.

logr_pred = logr.predict(X_test)
# burda biz X_test 'den y_test'i tahmin et dedik.
print(logr_pred,"\n")
print(y_test,"\n")

#----------------------------------------------------------------

## CONFUSİON MATRİX

from sklearn.metrics import confusion_matrix
# sklearn kütüphanesindeki metrics şeysinden confusion_matrix sınıfını import ettik.

cm = confusion_matrix(y_test, logr_pred)
# confusion matrix bir verinin gerçek hali ile tahmin edilen hali arasında oluşturulur
# bu yüzden yukarıda confusion_matrix sınıfından bir obje tanımlarken hangi veriler arasında
# confusion matrix oluşturacağını parametre olarak girdik.

print(cm,"\n")
# oluşturulan confusion matrix i bastırdık koşegenlerdeki değerler 
# bize doğru sınıflandırılan sayısı verilir.

#----------------------------------------------------------------

## K-NN 

from sklearn.neighbors import KNeighborsClassifier
# sklearn'den bişeler import ettik

knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")
# bu sınıftan bir obje oluşturduk n_neighbors kaç komşuya bakılacağı metric ise onu tam olarak bilmiyorum

knn.fit(X_train,y_train)
# knn algoritmasını kullanarak öğren dedik.
  
knn_pred = knn.predict(X_test)
# burda X_test' den y_test'i tahmin et dedik.

cm_1 = confusion_matrix(y_test,knn_pred)
# burda confusion_matrix oluştur dedik yeni algoritmamıza göre
print("K-NN CM \n")
print(cm_1,"\n")
# o confusion matrixi yazdırdık.

##NOT: biz burda komşu sayısını 5 belirledik ama komşu sayısını yüksek tutmamız illa başarı elde
## edeceğimiz anlamına gelmez örneğin bu veride komşu sayısını düşürünce başarı oranı artıyor.
## yani doğru veride doğru algoritma kullanılmalı

#-------------------------------------------------------------------

## SVC (SUPPORT VECTOR CLASSİFE)

from sklearn.svm import SVC

svc = SVC(kernel="linear")
# burda import ettiğimiz şeyden bir obje oluşturduk oluşturduk objemizden kernel fonksiyon olarak 
# linear seçtik başka şeylerde seçilebilir. Bu işte farklı modeller felan oluşturuyor.

svc.fit(X_train, y_train)
# X_train ile y_train arasındaki bağlantıyı öğren dedik.

svc_pred = svc.predict(X_test)
# öğrendiğin şekilde X_test tahmin et dedik.

svc_cm = confusion_matrix(y_test, svc_pred)
# confusion matrix objesi tanımladık, y_test ile svc_pred arasında oluşacak. 
print("SVC CM \n")
print(svc_cm,"\n")
# o objeyi bastırdık

#-----------------------------------------------------------------

## NAİVE BAYES

from sklearn.naive_bayes import GaussianNB
# sklearn kütüphanesinden naive bayes modülünden GaussianNB metodunu import ettik
# başka metodlar da kullanılabilir naive bayes için

gnb = GaussianNB()
# import ettiğimiz şeyden bir obje oluşturduk

gnb.fit(X_train, y_train)
# oluşturduğumuz objeye öğren dedik bu ikisi arasındaki bağı

gnb_pred = gnb.predict(X_test)
# öğrendiğine göre tahmin et dedik

gnb_cm = confusion_matrix(y_test, gnb_pred)
# confusion matrix objesi tanımladık, y_test ile gnb_pred arasında oluşacak. 
print("Naive bayes GNB CM \n")
print(gnb_cm,"\n")
# o objeyi bastırdık

#-------------------------------------------------------------------

## DECİSİON TREE CLASSİFİER

from sklearn.tree import DecisionTreeClassifier
# sınıf import ettik

dtc = DecisionTreeClassifier(criterion = "entropy")
# obje tanımladık entropy parametresiyle

dtc.fit(X_train, y_train)
# oluşturduğumuz objeye öğren dedik bu ikisi arasındaki bağı

dtc_pred = dtc.predict(X_test)
# öğrendiğine göre tahmin et dedik

dtc_cm = confusion_matrix(y_test, dtc_pred)
# confusion matrix objesi tanımladık, y_test ile dtc_pred arasında oluşacak. 
print("Decision Tree Classifier CM \n")
print(dtc_cm,"\n")
# o objeyi bastırdık

#-----------------------------------------------------------------

## RANDOM FOREST CLASSİFİER

from sklearn.ensemble import RandomForestClassifier
# sklearn küütphanesinden bu sınıfı import ettik

rfc = RandomForestClassifier(n_estimators=10, criterion="entropy")
# obje tanımladık o objeye 10 veriyi 10 parçaya böl dedik ve yöntem olarak entropy yap dedik

rfc.fit(X_train, y_train)
# öğren dedik algoritmaya

rfc_pred = rfc.predict(X_test)
# tahmin ettirdik

rfc_cm = confusion_matrix(y_test, rfc_pred)
# confusion matrix oluşturduk.

#-------------------------------------------

## PROBA OLUŞTURMA RANDOM FOREST CLASSİFİER İÇİN

rfc_proba = rfc.predict_proba(X_test)
# burda yaptığımız şey random forest classifierin verileri sınıflandırırken 
# önce o verinin dağılım olasılığını gösterir sonrasında bu veriyi olasılığı yüksek olana göre sınıflandırdığı için 
# olasılıklar önemlidir bu metot sayesinde ise biz X_test verisinin her bir elemanının olasılığını görüyoruz.
# örneğin bağımlı değişkenimizde kadın ve erkek olma sınıfı olsun biz elimizdeki verilere göre o kişinin 
# kadın mı erkek mi olduğunu tahmin edicek olalım işte o kişinin kadın mı erkek mi olduğunun olasılığını proba veriyor
# mesela 1. kişinin atıyorum %10 kadın %90 erkek olma olasılığını verdi tamamen salladım böyle veriyor.

print("RANDOM FOREST CLASSİFİER CM\n")
print(rfc_cm,"\n")

print("RANDOM FOREST CLASSİFİER PROPA\n")
print(rfc_proba,"\n")

#-------------------------------------------

## ROC CURVEY (TPR, FPR) OLUŞTURMA RANDOM FOREST CLASSİFİER İÇİN

from sklearn import metrics
# kütüphaneden sınıfı import ettik.

fpr, tpr, thold = metrics.roc_curve(y_test,rfc_proba[:,0],pos_label="e")
# yukarıda yaptığımız şey ilk şeye test için kullandığımız verileri verdik y_test yani kümelere bölmüştük test kümesini koyduk
# ikinci yere random foresta classifiere göre yaptığımız için random forest proba yı aldık proba olasılıkları veriyordu 
# 3. yere de pozitif seçeceğimiz değerlerin neler olduğunu yazdık pozitif olarak erkekleri seçtik 
# bu arada proba alınırken tek bir kolon almamız gerekti iki kolondan oluştuğu için tek kolon aldık. 
# yukarıdaki fpr ise false pozitive rate tpr ise true pozitive rate ve thold ise trash hold verileri 
# bunları da ayrı ayrı atadık.

print("fpr")
print(fpr,"\n")

print("tpr")
print(tpr,"\n")

# bu proba oluşturma ve roc curvey işlemlerini random forest classifier için yaptık.

#----------------------------------------------------------------











