
## KÜTÜPHANELER

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------------------

## VERİ YÜKLEME

veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\odev_tenis.csv")
print(veri,"\n")

#-----------------------------------------------

## KATEGORİK VERİ DÖNÜŞÜMÜ

outlook = veri.iloc[:,0:1].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

outlook[:,0] = le.fit_transform(veri.iloc[:,0])

one_hot_encoder = preprocessing.OneHotEncoder()

outlook = one_hot_encoder.fit_transform(outlook).toarray()
print(outlook,"\n")

# outlook da ise 2 den çok olasılık vardı o yüzden hem label hem onehot yaptık

windy = veri.iloc[:,3:4].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

windy = le.fit_transform(veri.iloc[:,3:4])
print(windy,"\n")

# windy ve play bunları sadece labelencoder yaptık yani 1 ve 0 şeklinde ddönüştürdük
# ama bunlarda iki olasılık vardı çünkü o yüzden onehotencoder kullanmaya gerek kalmadı

play = veri.iloc[:,-1:].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

play = le.fit_transform(veri.iloc[:,-1])

print(play,"\n")

#---------------------------------------------------------

## VERİLERİN BİRLEŞTİRİLMESİ


temp_hum = veri[["temperature","humidity"]]

sonuc = pd.DataFrame(data=outlook, index=range(14), columns=['overcast','rainy','sunny'])
print(sonuc,"\n")

sonuc_1 = pd.DataFrame(data=windy, index=range(14), columns=['windy'])
print(sonuc_1,"\n")

sonuc_2 = pd.DataFrame(data=play, index=range(14), columns=['play'])
print(sonuc_2,"\n")

s=pd.concat([sonuc,sonuc_1],axis=1)
print(s,"\n")

s_1=pd.concat([s,temp_hum],axis=1)
print(s_1,"\n")

s_2=pd.concat([s_1,sonuc_2],axis=1)
print(s_2,"\n")

#--------------------------------------------------------

## EĞİTİM VE TEST KÜMESİ OLUŞTURMA

from sklearn.model_selection import train_test_split
# yukarıda sklearn kütüphanesinden model_selection modülünden train_test_split class ını ekledik
x_train, x_test, y_train, y_test = train_test_split(s_1,sonuc_2,test_size=0.33,random_state=0)

#------------------------------------------------------------------

## ÖZNİTELİK ÖLÇEKLEME

from sklearn.preprocessing import StandardScaler
# sklearn kütüphanesinden preprocessing modülünden StandardScaler sınıfını import ettik.
sc=StandardScaler()
# sc değişkenini StandardScaler sınıfından bi obje haline getirdik.

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

Y_train =sc.fit_transform(y_train)
Y_test =sc.fit_transform(y_test)

#--------------------------------------------------------------------
## ÇOKLU DEĞİŞKEN LİNEER MODEL OLUŞTURMA 

from sklearn.linear_model import LinearRegression
# sklearn kütüphanesinden linear_model modülünden LinearRegression fonksiyonunu import ettik

regressor = LinearRegression()
# LinearRegression() fonksiyonundan bir regressor objesi oluşturduk

regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test)

#------------------------------------------------------------

## GERİYE ELEME YÖNTEMİ İLE HANGİ DEĞİŞKENLERİN ALINACAĞINI BELİRLEME


#-----------------------------------------------------------

# SABİT DEĞERİ OLUŞTURMA

import statsmodels.api as sm
# statsmodels kütüphanesinden api modulunu aldık 

X = np.append(arr=np.ones((14,1)).astype(int), values=s_1, axis=1)

#----------------------------------------------------------

# ELEMANLARIN P DEĞERLERİ BULUNMASI

X_list = s_1.iloc[:,[0,1,2,3,4,5]].values

X_list= np.array(X_list,dtype=float)

model=sm.OLS(sonuc_2,X_list).fit()
print(model.summary())
 

