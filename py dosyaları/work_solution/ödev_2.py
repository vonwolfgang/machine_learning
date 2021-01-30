

## KÜTÜPHANELER

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

#------------------------------------------------

## VERİ YÜKLEME

# pandasın read_csv modulü csv dosyalarını aktarmamızı sağlıyor.
veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\maaslar_yeni.csv")
print(veri,"\n")

## tahmin edilicek veri 10 yıl tecrübeli ve 100 puan almış bir CEO
## ve 10 yıl tecrübeli ve 100 puan almış bir müdür.
#---------------------------------------------------

## gereksiz veri calısan ID
## unvan seviyesi de gereksiz veri



#-----------------------------------------------------

#lineer regression

unvan = veri.iloc[:,1:2].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

unvan[:,0] = le.fit_transform(veri.iloc[:,1])

one_hot_encoder = preprocessing.OneHotEncoder()
unvan = one_hot_encoder.fit_transform(unvan).toarray()

unvan = pd.DataFrame(data=unvan, index=range(30), columns=['c-level','ceo','cayci','direktor','mudur','pro_yön','sef','sekreter','uzman','uzman_yar'])



x =pd.concat([unvan,veri.iloc[:,3:5]],axis=1)
y = veri.iloc[:,-1:]



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)




from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_train =sc.fit_transform(y_train)
Y_test =sc.fit_transform(y_test)




from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,Y_train)

prediction = lr.predict(X_test)


X = x.values
Y = y.values

print("lineer regresyon R^2")
print(r2_score(Y,lr.predict(X)),"\n")

#------------------------------------------------

## Polynomial regression

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=2)

x_poly=poly_reg.fit_transform(X)

from sklearn.linear_model import LinearRegression
# sklearn kütüphanesinden linear_model adlı modülden LinearRegression adlı sınıfı import ettik

lin_reg=LinearRegression()
# LinearRegression sınıfından bir tane obje oluşturduk

lin_reg.fit(x_poly,Y)
# polinomal hale getirdiğimiz x_poly 'nin katsayılarını y 'den öğren dedik
prediction_1 = lin_reg.predict(x_poly)

print("polinomial regression R^2")
print(r2_score(Y,lin_reg.predict(x_poly)),"\n")

#---------------------------------------------------------


## DESTEK VEKTÖR 
##NOT: destek vektör kullanırken verileri ölceklemen gerekiyor.
#------------------------------------------------------------------

## ÖZNİTELİK ÖLÇEKLEME

from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
sc2=StandardScaler()

X_olcek=sc1.fit_transform(x)
Y_olcek=sc2.fit_transform(y)

#----------------------------------------------------------
## SVR TAHMİN İÇİN KULLANMA 

from sklearn.svm import SVR
# sklearn kütüphanesinden svm modülünden SVR sınıfını import ettik.

svr_reg= SVR(kernel="rbf")  
svr_reg.fit(X_olcek,Y_olcek)
prediction_2 = svr_reg.predict(X_olcek)

print("SVR R^2")
print(r2_score(Y_olcek,svr_reg.predict(X_olcek)),"\n")

#-------------------------------------------------------

## DECİSİON TREE TAHMİN YÖNTEMİ 

from sklearn.tree import DecisionTreeRegressor
# burda sklearn kütüphanesinden tree modülünden DecisionTreeRegressor sınıfını import ettik

# NOT: decision tree ile tahmin felan yapmak için verileri standartize etmeye gerek yok 

r_dt= DecisionTreeRegressor(random_state=0)
# yukarıda bir obje yarattık random_state=0 ne bilmiom

r_dt.fit(X,Y)
# X ile Y arasındaki ilişkiyi öğren dedik.
prediction_3 = r_dt.predict(X)

print("decision tree R^2")
print(r2_score(Y,r_dt.predict(X)),"\n")

#------------------------------------------------------

## RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor
# sklearn kütüphanesinden ensemble modülünden RandomForestRegressor sınıfını import ettik

rf_reg= RandomForestRegressor(n_estimators=10, random_state=0)
# obje oluşturduk verdiğimiz değerler arsında ise n_estimators=10 demek 10 farklı decision tree 
# çiz demek verinin 10 farklı parçasını kullanarak.

rf_reg.fit(X,Y.ravel())
# X ile Y arasındaki bağlantıyı öğren dedik .ravel() ne işe yarıyor bilmiom.

prediction_4 = rf_reg.predict(X)

print("random forest R^2")
print(r2_score(Y,rf_reg.predict(X)),"\n")
# yukarıda random forest sisteminde gerçek Y nin alacağı değerler ile bizim sistemimizin tahmin ettiği Y değerleri girdik
# bunun sonucunda R^2 değerini hesaplattık.















