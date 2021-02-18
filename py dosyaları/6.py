


##  KÜTÜPHANELER 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-----------------------------------------------------

veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\musteriler.csv")

#-----------------------------------------------------

## K-Means algoritması

X = veri.iloc[:,3:].values
# son iki kolonu çektik çünkü şuan benim öğrenmem için yapıyoruz
# ve bu yüzden çok veriyle uğraşmamak için iki kolonu seçtik.

from sklearn.cluster import KMeans
# sklearn den cluster sınıfından KMeans altsınıfını import ettik

kmeans = KMeans(n_clusters = 3, init = "k-means++")
# KMeans sınıfından bir obje oluşturduk bu objeyi oluştururken kaç küme olacağını girdik ve hangi iyileştirme
# yöntemini kullanacağını girdik.

kmeans.fit(X)
# verileri kmeans algoritmasının içine verdik

print(kmeans.cluster_centers_,"\n")
# k-means le ilgili özellikleri alabiliriz örneğin yukarıda
# merkezlerini nerede oluşturduğunu aldık.

#---------------------------------------------------------

## K-Means WCSS İLE DOĞRU KÜME SAYISI BULMA

results = []
# sonuçlar için bir liste tanımladık

for i in range(1,11): 
    
    kmeans = KMeans(n_clusters=i, init = "k-means++", random_state=123)
    # doğru sınıfı bulsun diye her seferinde objeyi yeniden tanımlattık random_state bilerek aynı şey verdik 
    # her yeni tanımlamada aynı değere göre random etsin diye
    
    kmeans.fit(X)
    # verileri verdik
    
    results.append(kmeans.inertia_)
    # çıkan sonuçları listeye ekledik kmeans.inertia_ ile wcss hesapları yaptırdık.

plt.plot(range(1,11),results)
# grafiğini çizdirdik 1'den 10'a kadar olan değerleri al dedik
# ve results 'ı çizzdir dedik
# grafiğe göre 2, 3 veya 4 alınabilir.
plt.show()
print("\n")

#-------------------------------------------------------

## KMEANS TAHMİN ETME VE GÖRSELLEŞTİRME

kmeans_1 = KMeans(n_clusters=4, init = "k-means++", random_state=123)
# burda bulduğumuz uygun dirsek değerine göre tekrar obje oluşturduk   
kmeans_predict = kmeans_1.fit_predict(X)
# yukarıda inşa et ve tahmin et dedik.
print(kmeans_predict,"\n")

plt.scatter(X[kmeans_predict == 0,0],X[kmeans_predict==0,1],s=100, c="red")
plt.scatter(X[kmeans_predict == 1,0],X[kmeans_predict==1,1],s=100, c="blue")
plt.scatter(X[kmeans_predict == 2,0],X[kmeans_predict==2,1],s=100, c="green")
plt.scatter(X[kmeans_predict == 3,0],X[kmeans_predict==3,1],s=100, c="black")
plt.title("KMeans")
plt.show()
print("\n")
# yukarıda gene yaptığımız şeye göre çizdirdik ama bu şeylerin ne olduğuyla ilgili
# pek bir fikrim yok 

#-------------------------------------------------------------------

## HİYERARŞİK BÖLÜTLEME

from sklearn.cluster import AgglomerativeClustering
# gerekli olan kütüphaneyi import ettik

ac = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
# parametrelerle obje tanımladık

ac_predict = ac.fit_predict(X)
# yukarıda hem sistemi inşa et hem tahmin et dedik

print(ac_predict,"\n")
# tahmin sonuçlarını bastırdık.

plt.scatter(X[ac_predict == 0,0],X[ac_predict==0,1],s=100, c="red")
plt.scatter(X[ac_predict == 1,0],X[ac_predict==1,1],s=100, c="blue")
plt.scatter(X[ac_predict == 2,0],X[ac_predict==2,1],s=100, c="green")
plt.title("Hiyerarşik Bölütleme")
plt.show()
print("\n")
# biz 3 sınıf oluştur dediğimiz için 3 tane şeyi çizdirdik

#---------------------------------------------------------------

## DENDROGRAM

import scipy.cluster.hierarchy as sch
# dendogram oluşturmak için bu kütüphaneden bişeyler import ettik

dendrogram = sch.dendrogram(sch.linkage(X,method="ward"))
# obje tanımladık
plt.show()


















