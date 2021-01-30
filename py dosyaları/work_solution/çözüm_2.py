## KÜTÜPHANELER

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import statsmodels.api as sm
#------------------------------------------------

## VERİ YÜKLEME

# pandasın read_csv modulü csv dosyalarını aktarmamızı sağlıyor.
veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\maaslar_yeni.csv")

#--------------------------------------------

# bizim verimizdeki unvan değişkeni zaten unvanseviyesinde de verildiği için
# unvan değişkenini almamıza gerek yok. Ve ayrıca çalışan ID ise sistemimimizin
# ezbere gitmesine sebep olurmuş o yüzden onuda almıcaz 
# maas bağımlı değişken almıcaklarımız ve maas dışındakiler ise bağımsız değişken


x = veri.iloc[:,2:5]
y = veri.iloc[:,-1:]
X=x.values
Y=y.values

#---------------------------------------------

## LİNEER REGRESYON MODELİ OLUŞTURMA

x_l = veri.iloc[:,2:3]
y_l = veri.iloc[:,-1:]
X_l=x_l.values
Y_l=y_l.values

# ilk baştaki değişkenler ile R2 ve adj R2 değerleri sırasıyla böyleydi 0,903  0,892 
# p değerleri yüksek olan değişkenleri attım ve atınca daha iyi bir R2 ve adj R2 sonucu
# onlar da sırasıyla 0.942 ve 0.940 bu yüzden p value'su yüksek olan değişkenlerinin 
# atılmış halini kullanmaya karar verdim. 


from sklearn.linear_model import LinearRegression
# sklearn kütüphanesinden linear_model adlı modülden LinearRegression adlı sınıfı import ettik

lin_reg= LinearRegression()
# o sınıftan bir obje oluşturduk

lin_reg.fit(X_l,Y_l)
# x den y ' ye öğren dedik.

## P VALUE HESAPLAMA

print("lineer regression OLS")
model = sm.OLS(lin_reg.predict(X_l),X_l)
# burda sm den OLS adlı fonksiyonu çağırdık ve ona X değerlerini
# lin_reg e göre tahmin et ve gerçek X değerleri ile karşılaştırarak p value ölc dedik.
print(model.fit().summary(),"\n")

## R^2 HESAPLAMA 

print("lineer regression R^2")
print(r2_score(Y_l,lin_reg.predict(X_l)),"\n")
print("\n")



"""
p value değeri sonrası R^2 ve adj R^2 değerlerini not alıyoruz ve p değerlerine bakıyoruz
baktığımız değişkenlerden yüksek olanları atıyoruz ve yeniden p value bakıyoruz
ve R^2 ve adj R^2 değerlerini karşılaştırıyoruz.
"""

#------------------------------------------------

##  POLİNOMAL REGRESYON MODEL OLUŞTURMA

from sklearn.preprocessing import PolynomialFeatures
# sklearn kütüphanesinden preprocessing modülünden PolynomialFeatures sınıfını import ettik

poly_reg=PolynomialFeatures(degree=2)
# o sınıftan 2. dereceden bir obje oluşturduk

x_poly=poly_reg.fit_transform(X_l)
# burda X 'i polinom olarak fit transform ettik
# yani polinomal bi şekle dönüştürdük
# burda sabit değeri her zaman bir aldı 
# x'in birinci derecesini normal aldı 
# x'in ikinci derecesini ise karesini aldı 

from sklearn.linear_model import LinearRegression
# sklearn kütüphanesinden linear_model adlı modülden LinearRegression adlı sınıfı import ettik

lin_reg_2=LinearRegression()
# LinearRegression sınıfından bir tane obje oluşturduk

lin_reg_2.fit(x_poly,y_l)
# polinomal hale getirdiğimiz x_poly 'nin katsayılarını y 'den öğren dedik


## P VALUE HESAPLAMA

print("Polynomiol OLS")
model_1 = sm.OLS(lin_reg_2.predict(poly_reg.fit_transform(X_l)),X_l)
# polynomial regression'da p value felan hesaplarkende hep polynomial hale dönüştürülmüş halini kullanıyoruz bağımsız değişkenlerin
print(model_1.fit().summary(),"\n")

## R2 HESAPLAMA

print(" Polinomiol R^2")
print(r2_score(Y,lin_reg_2.predict(poly_reg.fit_transform(X_l))),"\n")
print("\n")

"""

 R-squared 0.729, Adj. R-squared 0.698 bunlar eski değerler 
 
 R-squared 0.810, Adj. R-squared 0.803 bunlar ise p valuesi yüksek olan atıldıktan 
 
 sonraki değerler. 
 
"""
#-------------------------------------------------

## DESTEK VEKTÖR 
##NOT: destek vektör kullanırken verileri ölceklemen gerekiyor.

## ÖZNİTELİK ÖLÇEKLEME

from sklearn.preprocessing import StandardScaler
# sklearn kütüphanesinden preprocessing modülünden StandardScaler sınıfını import ettik.

sc1=StandardScaler()
# sc değişkenini StandardScaler sınıfından bi obje haline getirdik.
sc2=StandardScaler()

X_olcek=sc1.fit_transform(X_l)
Y_olcek=sc2.fit_transform(Y_l)
# tam emin değilim ama sanırsam bu StandardScaler bu elimizdeki verileri aynı dünyanın verisi yapmaya yarıyor.
# sanırsam hepsini birbirine benzer yapıyor birbirine göre ölçekliyor, optimize ediyor.


## SVR TAHMİN İÇİN KULLANMA 

from sklearn.svm import SVR
# sklearn kütüphanesinden svm modülünden SVR sınıfını import ettik.

svr_reg= SVR(kernel="rbf")  
# tek kernel fonksiyon rbf deil başka kernel fonksiyonlar var
#kernel = rbf gauss metodu gibi bişe bu tam anlamadım ama 
svr_reg.fit(X_olcek,Y_olcek)
#burda ise ölceklendirdiğimiz verileri oluşturduğumuz obje üzerinden svr oluşturuyoz.


## P VALUE HESAPLAMA

print("SVR OLS")
model_2 = sm.OLS(svr_reg.predict(sc1.fit_transform(X_l)),X_olcek)
# polynomial regression'da p value felan hesaplarkende hep polynomial hale dönüştürülmüş halini kullanıyoruz bağımsız değişkenlerin
print(model_2.fit().summary(),"\n")

## R2 HESAPLAMA

print("SVR R^2")
print(r2_score(Y_olcek,svr_reg.predict(X_olcek)),"\n")
print("\n")

#--------------------------------------------------------


## DECİSİON TREE TAHMİN YÖNTEMİ 

from sklearn.tree import DecisionTreeRegressor
# burda sklearn kütüphanesinden tree modülünden DecisionTreeRegressor sınıfını import ettik

# NOT: decision tree ile tahmin felan yapmak için verileri standartize etmeye gerek yok 

r_dt= DecisionTreeRegressor(random_state=0)
# yukarıda bir obje yarattık random_state=0 ne bilmiom

r_dt.fit(X_l,Y_l)
# X ile Y arasındaki ilişkiyi öğren dedik.


## P VALUE HESAPLAMA

print("decision tree OLS")
model_3 = sm.OLS(r_dt.predict(X_l),X_l)
# polynomial regression'da p value felan hesaplarkende hep polynomial hale dönüştürülmüş halini kullanıyoruz bağımsız değişkenlerin
print(model_3.fit().summary(),"\n")

## R2 HESAPLAMA

print("decision tree R^2")
print(r2_score(Y_l,r_dt.predict(X_l)),"\n")
print("\n")
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

rf_reg.fit(X_l,Y_l.ravel())
# X ile Y arasındaki bağlantıyı öğren dedik .ravel() ne işe yarıyor bilmiom.


## P VALUE HESAPLAMA

print("random forest OLS")
model_4 = sm.OLS(rf_reg.predict(X_l),X_l)
# polynomial regression'da p value felan hesaplarkende hep polynomial hale dönüştürülmüş halini kullanıyoruz bağımsız değişkenlerin
print(model_4.fit().summary(),"\n")

## R^2 HESAPLAMA 

print("random forest R^2")
print(r2_score(Y_l,rf_reg.predict(X_l)),"\n")
print("\n")
# yukarıda random forest sisteminde gerçek Y nin alacağı değerler ile bizim sistemimizin tahmin ettiği Y değerleri girdik
# bunun sonucunda R^2 değerini hesaplattık.

#-------------------------------------------------------------------
 
"""
## ÖZET BİLGİ

3 DEĞİŞKENLİ DURUMDA

LİNEAR
R2    0,903

POLY
R2    0,680

SVR
R2    0,782

DT
R2    0,679

RF
R2    0,713
--------------------------------------

1 DEĞİŞKENLİ

LİNEAR
R2    0,942

POLY
R2    0,759

SVR
R2    0,770

DT
R2    0,751

RF
R2    0,719
---------------------------------------

BAZI DURUMLARDA P VALUESİ YÜKSEK DE OLSA O DEĞİŞKENİ
ATMAK TAHMİN ALGORİTMASINI KÖTÜ YÖNDE ETKİLEYEBİLİR.

"""
#---------------------------------------------------

print(veri.corr())
# bu fonksiyon bizim yukarıda yaptığımız hangi veriyi alırsak işlemi
# daha doğru bir tahmin yapıcağımızı bulmak için bize verilerin 
# birbirleriyle ilişkisini gösteren bir matriks oluşturuyor.
# bu fonksiyon pandas kütüphanesine ait.






















