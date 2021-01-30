
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

from sklearn import preprocessing

veri_1=veri.apply(preprocessing.LabelEncoder().fit_transform)
# bu işlem sayesinde uzun uzun parçalayarak yaptığımız o labelencode etme işlemini
# bu modül içinde yazmışlar biz ise sadece o modülü kullanarak kısaca tüm hepsini label encode ettik
# ama burda temperatue gibi label encode edilmemesi gerekenler de encode edilmiş!!!

outlook=veri_1.iloc[:,:1]
# yukarıda hepsini labelencode ettik ama outlook onehot encode da edilmesi gerekiyor o yüzden 
# outlook u çektik
one_hot_encoder = preprocessing.OneHotEncoder()
# preprocessing modülünden OneHotEncoder() sınıfından bir obje oluşturduk
outlook=one_hot_encoder.fit_transform(outlook).toarray()
# outlook u one hot encode ettik

#---------------------------------------------------------

## VERİLERİN BİRLEŞTİRİLMESİ

sonuc = pd.DataFrame(data=outlook, index=range(14), columns=['overcast','rainy','sunny'])
print(sonuc,"\n")

s=pd.concat([sonuc, veri.iloc[:,1:3]],axis=1)
print(s,"\n")
# burda veri den aldık concat ettik çünkü burda humidity ve temperature aldık yani label encode edilmemesi gerekenleri

s_1=pd.concat([veri_1.iloc[:,3:],s],axis=1)
print(s_1,"\n")
# burda ise veri_1 den aldık çünkü burdan aldıklarımı windy ve play labelencode edilmesi gerekiyordu.
#--------------------------------------------------------

## EĞİTİM VE TEST KÜMESİ OLUŞTURMA

from sklearn.model_selection import train_test_split
# yukarıda sklearn kütüphanesinden model_selection modülünden train_test_split class ını ekledik
x_train, x_test, y_train, y_test = train_test_split(s_1.iloc[:,0:6],s_1.iloc[:,6:],test_size=0.33,random_state=0)

#------------------------------------------------------------------

## ÖZNİTELİK ÖLÇEKLEME

# ÖZNİTELİK ÖLÇEKLENDİRME YAPMADIK NEDEN BİLMİYORUM.

#--------------------------------------------------------------------
## TAHMİN ETTİRME

# biz bu uygulamada humidity'i tahmin ettirdik.

from sklearn.linear_model import LinearRegression
# sklearn kütüphanesinden linear_model modülünden LinearRegression fonksiyonunu import ettik

regressor = LinearRegression()
# LinearRegression() fonksiyonundan bir regressor objesi oluşturduk

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#------------------------------------------------------------

## GERİYE ELEME YÖNTEMİ İLE HANGİ DEĞİŞKENLERİN ALINACAĞINI BELİRLEME


#-----------------------------------------------------------

# SABİT DEĞERİ OLUŞTURMA

import statsmodels.api as sm
# statsmodels kütüphanesinden api modulunu aldık 

X = np.append(arr=np.ones((14,1)).astype(int), values=s_1.iloc[:,:-1], axis=1)

#----------------------------------------------------------

# ELEMANLARIN P DEĞERLERİ BULUNMASI

X_list = s_1.iloc[:,[0,1,2,3,4,5]].values
# bunu yaparken bağımlı değişkeni almıyoruz.

X_list= np.array(X_list,dtype=float)

model=sm.OLS(s_1.iloc[:,-1:],X_list).fit()
print(model.summary())
# p değerleri çok yüksek o yüzden en yükseğinden başlayarak tek tek atıyoruz.



s_2 = s_1.iloc[:,1:]
X_list = s_2.iloc[:,[0,1,2,3,4]].values
# bunu yaparken bağımlı değişkeni almıyoruz.

X_list= np.array(X_list,dtype=float)

model=sm.OLS(s_2.iloc[:,-1:],X_list).fit()
print(model.summary())
# birinci sütunu atmıştık şimdi de yüksek bu sefer de en yüksek p değerine sahip olan
# x4 'ü atıyorum



s_3 = s_2.iloc[:,:4]
X_list = s_3.iloc[:,[0,1,2,3]].values
# bunu yaparken bağımlı değişkeni almıyoruz.

X_list= np.array(X_list,dtype=float)

model=sm.OLS(s_2.iloc[:,-1:],X_list).fit()
print(model.summary())
# windy ve temperature sütunuda gitti şimdi de yüksek olan x1 'i atıom



s_4 = s_3.iloc[:,1:]
X_list = s_4.iloc[:,[0,1,2]].values
# bunu yaparken bağımlı değişkeni almıyoruz.

X_list= np.array(X_list,dtype=float)

model=sm.OLS(s_2.iloc[:,-1:],X_list).fit()
print(model.summary())
# sonunda p value düştü şimdi de tekrar tahmin edelim bakalım 


#--------------------------------------------------------

## TEKRARDAN EĞİTİM VE TEST KÜMESİ OLUŞTURMA

from sklearn.model_selection import train_test_split
# yukarıda sklearn kütüphanesinden model_selection modülünden train_test_split class ını ekledik
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(s_4,s_1.iloc[:,6:],test_size=0.33,random_state=0)

#--------------------------------------------------------------------

## TEKRARDAN TAHMİN ETTİRME

# biz bu uygulamada humidity'i tahmin ettirdik.

from sklearn.linear_model import LinearRegression
# sklearn kütüphanesinden linear_model modülünden LinearRegression fonksiyonunu import ettik

regressor = LinearRegression()
# LinearRegression() fonksiyonundan bir regressor objesi oluşturduk

regressor.fit(x_train_1,y_train_1)
y_pred_1 = regressor.predict(x_test_1)

#------------------------------------------------------------



















