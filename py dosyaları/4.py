
## KÜTÜPHANELER

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
# sklearn kütüphanesinden metrics modülünden r2_score sınıfını import ettik.

#------------------------------------------------

## VERİ YÜKLEME

# pandasın read_csv modulü csv dosyalarını aktarmamızı sağlıyor.
veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\maaslar.csv")

#-----------------------------------------------

## DATA FRAME DİLİMLEMEK

x=veri.iloc[:,1:2]
y=veri.iloc[:,-1:]
# kullanıcağımız verileri çektik 
# bu veriler polinomal regresyon ile model oluşturulur.

#-----------------------------------------------

## DİLİMLENMİŞLERİ NUMPY ARRAY(DİZİ) DÖNÜŞTÜRMEK

X=x.values
Y=y.values
# data frame olan verileri bi de values olarak tutturduk

#-----------------------------------------------

## LİNEER REGRESYON MODELİ OLUŞTURMA

from sklearn.linear_model import LinearRegression
# sklearn kütüphanesinden linear_model adlı modülden LinearRegression adlı sınıfı import ettik

lin_reg= LinearRegression()
# o sınıftan bir obje oluşturduk

lin_reg.fit(X,Y)
# x den y ' ye öğren dedik.

#----------------------------------------------

## LİNEER REGRESYON MODELİ GÖRSELLEŞTİRME

plt.scatter(X,Y,color="black") # color="black" diyerek rengini black yaptık.
# görselleştirmek için yaptık
plt.plot(X,lin_reg.predict(X),color="red") 
plt.show() # şuanki görseli ekrana vercek
print("\n")
# colour="red" diyerek rengini red yaptık.
# x ' den y'yi tahmin ettirdik ve görselleştirdik

#------------------------------------------------

## LİNEER REGRESYON GRAFİĞİ ÜZERİNDEN TAHMİNLER

print(lin_reg.predict([[11]]),"\n") 
print(lin_reg.predict([[6.6]]),"\n")
# sanırsam burda olan şey bizim lineer regresyon modeline göre çizdiğimiz grafikte
# predict([[11]]) grafikte ki 11. şeye gelen şeyi tahmin ediyor ama tam anlamadım

print("lineer regresyon R^2")
print(r2_score(Y,lin_reg.predict(X)),"\n")
# normalde polinomal yapılması gereken ama bizim lineer regresyon yaptığımız modelin R^2 hesaplaması
#------------------------------------------------

##  POLİNOMAL REGRESYON MODEL OLUŞTURMA

from sklearn.preprocessing import PolynomialFeatures
# sklearn kütüphanesinden preprocessing modülünden PolynomialFeatures sınıfını import ettik

poly_reg=PolynomialFeatures(degree=2)
# o sınıftan 2. dereceden bir obje oluşturduk

x_poly=poly_reg.fit_transform(X)
print(x_poly,"\n")
# burda X 'i polinom olarak fit transform ettik
# yani polinomal bi şekle dönüştürdük
# burda sabit değeri her zaman bir aldı 
# x'in birinci derecesini normal aldı 
# x'in ikinci derecesini ise karesini aldı 

from sklearn.linear_model import LinearRegression
# sklearn kütüphanesinden linear_model adlı modülden LinearRegression adlı sınıfı import ettik

lin_reg_2=LinearRegression()
# LinearRegression sınıfından bir tane obje oluşturduk

lin_reg_2.fit(x_poly,y)
# polinomal hale getirdiğimiz x_poly 'nin katsayılarını y 'den öğren dedik

#-------------------------------------------------

## POLİNOMAL REGRESYON MODELİNİN GÖRSELLEŞTİRİLMESİ

plt.scatter(X,Y,color="red")
# asıl değerler olan değerleri lineer şekilde görselleştiricek bu

plt.plot(X,lin_reg_2.predict(x_poly),color="black")
plt.show()# bu şimdiki görseli göstericek
print("\n")
# 1. haneye koyduğumuz X değerini neden koyduk anlamadım
# ikinci kısma ise tahmin et dedik polinomal hale geliş x 'e atanan değerlerden

# çıkan grafikte tahminin rengi siyah gerçek değerin resmi kırmızı

# burda yaptığımız örnekte en büyük x'in derecesini 2 olarak belirledik
# bu dereceyi değiştirebiliriz.

#------------------------------------------------

## POLİNOMAL REGRESYON GRAFİĞİ ÜZERİNDEN TAHMİNLER

print(lin_reg_2.predict(poly_reg.fit_transform([[6.6]])),"\n")
print(lin_reg_2.predict(poly_reg.fit_transform([[11]])),"\n")
# sanırsam burda olan şey bizim polinomal regresyon modeline göre çizdiğimiz grafikte
# predict([[11]]) grafikte ki 11. şeye gelen şeyi tahmin ediyor ama tam anlamadım

print("polinomial regression R^2")
print(r2_score(Y,lin_reg_2.predict(x_poly)),"\n")
# polinomal regresyon modelinin R^2 hesaplaması

#----------------------------------------------

## DESTEK VEKTÖR 
##NOT: destek vektör kullanırken verileri ölceklemen gerekiyor.
#------------------------------------------------------------------

## ÖZNİTELİK ÖLÇEKLEME

from sklearn.preprocessing import StandardScaler
# sklearn kütüphanesinden preprocessing modülünden StandardScaler sınıfını import ettik.

sc1=StandardScaler()
# sc değişkenini StandardScaler sınıfından bi obje haline getirdik.
sc2=StandardScaler()

X_olcek=sc1.fit_transform(x)
Y_olcek=sc2.fit_transform(y)
# tam emin değilim ama sanırsam bu StandardScaler bu elimizdeki verileri aynı dünyanın verisi yapmaya yarıyor.
# sanırsam hepsini birbirine benzer yapıyor birbirine göre ölçekliyor, optimize ediyor.

#----------------------------------------------------------
## SVR TAHMİN İÇİN KULLANMA 

from sklearn.svm import SVR
# sklearn kütüphanesinden svm modülünden SVR sınıfını import ettik.

svr_reg= SVR(kernel="rbf")  
# tek kernel fonksiyon rbf deil başka kernel fonksiyonlar var
#kernel = rbf gauss metodu gibi bişe bu tam anlamadım ama 
svr_reg.fit(X_olcek,Y_olcek)
#burda ise ölceklendirdiğimiz verileri oluşturduğumuz obje üzerinden svr oluşturuyoz.

plt.scatter(X_olcek,Y_olcek,color="black")
#bu x ve y boyutlarını oluşturucak scatter edicek.
plt.plot(X_olcek,svr_reg.predict(X_olcek),color="red")
# her bir X_olcek için svr_reg deki sonuclarını predict et ve göster dedik.

# dolayısıyla yukarıdaki Y_olcek ile bizim tahmin ettirdiğimiz predict(X_olcek)
# karşılaştırılmış olcak.
 
print(svr_reg.predict([[11]]),"\n")
print(svr_reg.predict([[6.6]]),"\n")
# yukarıda biryerdeki 6.6 noktasını ve 11 noktasını tahmin et dedik.
# ama bu noktalar nerde onu anlamadım.

print("SVR R^2 R^2")
print(r2_score(Y_olcek,svr_reg.predict(X_olcek)),"\n")

#--------------------------------------------------------

## DECİSİON TREE TAHMİN YÖNTEMİ 

from sklearn.tree import DecisionTreeRegressor
# burda sklearn kütüphanesinden tree modülünden DecisionTreeRegressor sınıfını import ettik

# NOT: decision tree ile tahmin felan yapmak için verileri standartize etmeye gerek yok 

r_dt= DecisionTreeRegressor(random_state=0)
# yukarıda bir obje yarattık random_state=0 ne bilmiom

r_dt.fit(X,Y)
# X ile Y arasındaki ilişkiyi öğren dedik.

Z = X + 0.5
# burda sanırsam X değerlerini 0.5 sağa kaydırdık.

K = X - 0.4
# burda da X değerlerini 0.4 sola kaydırdık.

plt.scatter(X,Y,color="blue")
# X değerini uzayda çiz dedik.
plt.plot(X,r_dt.predict(X),color="black")
# X değerini çiz ve decisiontree'nin tahmin ettiği her bir X değeri için tahmin edilen Y değerini çiz dedik.
plt.plot(X,r_dt.predict(Z),color="green")
# burda yeni belirlediğimiz X değerlerine göre tahmin et dedik
plt.plot(X,r_dt.predict(K),color="red")
# burda yeni belirlediğimiz X değerlerine göre tahmin et dedik
plt.show()
# bu tabloyu direk gösterip yeni tabloya geçiyor üst üste bindirmiyor.
print("\n")

print(r_dt.predict([[11]]),"\n")
print(r_dt.predict([[6.6]]),"\n")
# bu 11 ve 6.6 'nın ne olduğunu hala anlamadım ama burda bu sefer decision tree 'ye 
# tahmin ettirdik.

print("decision tree R^2")
print(r2_score(Y,r_dt.predict(X)),"\n")
#decision tree R^2 hesaplaması
# decision tree R^2 sonucu 1 çıktı ama decision tree en iyi tahmin yöntemi deil 
# sadece bir sonuca bakarak yapamazsın.

#------------------------------------------------------

## RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor
# sklearn kütüphanesinden ensemble modülünden RandomForestRegressor sınıfını import ettik

rf_reg= RandomForestRegressor(n_estimators=10, random_state=0)
# obje oluşturduk verdiğimiz değerler arsında ise n_estimators=10 demek 10 farklı decision tree 
# çiz demek verinin 10 farklı parçasını kullanarak.

rf_reg.fit(X,Y.ravel())
# X ile Y arasındaki bağlantıyı öğren dedik .ravel() ne işe yarıyor bilmiom.

print(rf_reg.predict([[6.6]]),"\n")
# burda gene 6.6 yı random forest yöntemi ile tahmin ettirdik ama 6.6 ne anlamadım.

plt.scatter(X,Y,color="red")
# asıl verileri görselleştir dedik.
plt.plot(X,rf_reg.predict(X),color="blue")
# X değerlerine karşılık gelen Y değerlerini tahmin et dedik ne ye göre X'e göre tahmin et dedik.
plt.plot(X,rf_reg.predict(Z),color="green")
# burda yeni belirlediğimiz X değerlerine göre tahmin et dedik
plt.plot(X,rf_reg.predict(K),color="yellow")
# burda yeni belirlediğimiz X değerlerine göre tahmin et dedik
plt.show()
# grafiği çiz dedik.
print("\n")

#---------------------------------------------------------------------

## RANDOM FOREST R^2 DEĞERİ HESAPLAMA 

from sklearn.metrics import r2_score
# sklearn kütüphanesinden metrics modülünden r2_score sınıfını import ettik.

print("random forest R^2")
print(r2_score(Y,rf_reg.predict(X)),"\n")
# yukarıda random forest sisteminde gerçek Y nin alacağı değerler ile bizim sistemimizin tahmin ettiği Y değerleri girdik
# bunun sonucunda R^2 değerini hesaplattık.

#-------------------------------------------------------------------



















