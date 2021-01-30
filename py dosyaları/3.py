
## ÇOK DEĞİŞKENLİ REGRESYON İÇİN VERİ HAZIRLAMA

#------------------------------------------------------

## KÜTÜPHANELER

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#---------------------------------

## VERİ YÜKLEME

veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\data.csv")
print(veri,"\n")

#------------------------------------------------------

## KATEGORİK VERİ DÖNÜŞÜMÜ

# ulke kolonu için

ulke = veri.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veri.iloc[:,0])

one_hot_encoder = preprocessing.OneHotEncoder()

ulke = one_hot_encoder.fit_transform(ulke).toarray()
print(ulke,"\n")

#-----------------------------------------------------

# cinsiyet kolonu için

cınsıyet = veri.iloc[:,-1:].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

cınsıyet[:,-1] = le.fit_transform(veri.iloc[:,-1])

one_hot_encoder = preprocessing.OneHotEncoder()

cınsıyet = one_hot_encoder.fit_transform(cınsıyet).toarray()
print(cınsıyet,"\n")

#---------------------------------------------------------

## VERİLERİN BİRLEŞTİRİLMESİ

#----------------------------------------
boy_kilo_yas = veri[["boy","kilo","yas"]]
print(boy_kilo_yas)
# ben boy kilo ve yas kolonlarını önceden seçmediğim için şimdi burda aldım 
#---------------------------------------

sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])

sonuc_1 = pd.DataFrame(data=boy_kilo_yas, index=range(22), columns=['boy','kilo','yas'])

sonuc_2 = pd.DataFrame(data=cınsıyet[:,:1], index=range(22), columns=['cinsiyet'])
# Biz burda numeric hale dönüştürdüğümüz 2 cinsiyet kolonundan birini kullandık 
# çünkü ikisini birden kullanırsak "dumy trap" denen sorunu yaşayabilirdik.

s=pd.concat([sonuc,sonuc_1],axis=1)
#burda ulke ve boy kilo yas verilerini birleştirdik

s_1=pd.concat([s,sonuc_2],axis=1)
print(s_1,"\n")
# burda artık tüm verileri birleştirdik

#-----------------------------------------------------------

## EĞİTİM VE TEST KÜMESİ OLUŞTURMA

from sklearn.model_selection import train_test_split
# yukarıda sklearn kütüphanesinden model_selection modülünden train_test_split class ını ekledik

x_train, x_test, y_train, y_test = train_test_split(s,sonuc_2,test_size=0.33,random_state=0)
# ne işe yaradığıyla ya da ne yaptığımızla ilgili en ufak bir fikrim yok sor bunu
# ama sanırsam şöyle bişey olabilir bu yaptığımız ile seçtiğimiz verileri 4 e böldü test_size=0.33 dediğimiz için %33'ü test bölümüne atandı
# geri kalanı ise train olarak atandı ve bu atama random_state = 0 dediğimiz için atamayı rastgele yaptı.
 
#----------------------------------------------------------

## ÇOKLU DEĞİŞKEN LİNEER MODEL OLUŞTURMA 

from sklearn.linear_model import LinearRegression
# sklearn kütüphanesinden linear_model modülünden LinearRegression fonksiyonunu import ettik

regressor = LinearRegression()
# LinearRegression() fonksiyonundan bir regressor objesi oluşturduk

regressor.fit(x_train,y_train)
# burda oluşturduğumuz objenin fit şeyini kullandık fit ise şundan şunu öğren demeye yarıyor
# yani biz burda x_train den y_train'i öğren dedik.
# öğrenmekten kastı ise x_train ile y_train arasında bir bağlantı model oluştur demek.

y_pred = regressor.predict(x_test)
# burda ise yukarda kullandığı bağlantı veya model sayesinde
# x_test ' den y_test' i tahmin et ve tahminini y_pred içine ata dedik.

# biz yukarıda cinsiyet değerini tahmin ettirdik makinaya

#---------------------------------------------------------

## BOY DEĞERİNİ TAHMİN ETTİRME

#--------------------------------------------------------

## BU AŞAĞIDAKİ BOY TAHMİN İŞLEMİNİ BEN KENDİM YAPTIM  

"""
boy = s_1.iloc[:,3:4].values
# hazırlanmış verimizden boy değişkenini çektik

sonuc_3 = pd.DataFrame(data=boy, index=range(22), columns=['boy'])
# çektiğimiz boy kolonunu düzenledik.


kilo_yas = boy_kilo_yas[["kilo","yas"]]
# boy_kilo_yas şeyinden sadece kilo ve yası çektim.

s_3=pd.concat([sonuc,kilo_yas],axis=1)
# kilo ve yası encode edilmiş ulke kolonlarına ekledim.

s_4=pd.concat([s_3,sonuc_2],axis=1)
# bi önceki adımda birleştirdiğim verilere encode edilmiş ve dummy trap 'den temizlenmiş cinsiyet kolonunu ekledim.


x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(s_4,sonuc_3,test_size=0.33,random_state=0)
# makinaya s_4 ' deki verileri ve sonuc_3 'deki verileri train ve test olarak rastgele bir şekil de bölmesini söyledik.


regressor_1 = LinearRegression()
# LinearRegression() fonksiyonundan bir regressor_1 objesi oluşturduk

regressor.fit(x_train_1,y_train_1)
# burda oluşturduğumuz objenin fit şeyini kullandık fit ise şundan şunu öğren demeye yarıyor
# yani biz burda x_train_1 den y_train_1'i öğren dedik.
# öğrenmekten kastı ise x_train_1 ile y_train_1 arasında bir bağlantı model oluştur demek.

y_pred_1 = regressor.predict(x_test_1)
# burda ise yukarda kullandığı bağlantı veya model sayesinde
# x_test_1 ' den y_test_1' i tahmin et ve tahminini y_pred_1 içine ata dedik.
"""
#------------------------------------------------------------------

## HOCANIN YAPTIĞI BOY TAHMİN İŞLEMİ

boy_1 = s_1.iloc[:,3:4].values
# hazırlanmış verimizden boy değişkenini çektik


sol=s_1.iloc[:,:3] 
# s_1 tablosundaki 3. kolona kadar olan hepsini aldı
sağ=s_1.iloc[:,4:] 
# s_1 tablosundaki 4. kolondan başlayıp kalan kolonları aldı


s_5=pd.concat([sol,sağ],axis=1)
# demin çektiğimiz verileri tek bir tabloda birleştirdi.


x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(s_5,boy_1,test_size=0.33,random_state=0)
# makinaya s_5 ' deki verileri ve boy_1 'deki verileri train ve test olarak rastgele bir şekil de bölmesini söyledik.


regressor.fit(x_train_2,y_train_2)
# burda oluşturduğumuz objenin fit şeyini kullandık fit ise şundan şunu öğren demeye yarıyor
# yani biz burda x_train_2 den y_train_2'i öğren dedik.
# öğrenmekten kastı ise x_train_2 ile y_train_2 arasında bir bağlantı model oluştur demek.

y_pred_2 = regressor.predict(x_test_2)
# burda ise yukarda kullandığı bağlantı veya model sayesinde
# x_test_2 ' den y_test_2' i tahmin et ve tahminini y_pred_2 içine ata dedik.

#--------------------------------------------------------------------

## GERİYE ELEME YÖNTEMİ İLE HANGİ DEĞİŞKENLERİN ALINACAĞINI BELİRLEME

import statsmodels.api as sm
# statsmodels kütüphanesinden api modulunu aldık 

X = np.append(arr=np.ones((22,1)).astype(int), values=s_5, axis=1)
# numpy kütüphanesinin özelliğini kullanarak dizi oluşturduk yani array oluşturduk
# bu arrayin boyutu matriks de diyebiliriz 22 satır 1 sütun şeklinde astype olarak
# int belirledik ve bu oluşturacağı değerleri s_5 datasından alarak yap dedik. axis 1 ise kolon olarak ekle demek. 
# aslında hani çok değişkenli lineer model de bi sabit değer var dı işte o sabit değeri eklemiş olduk.


X_list = s_5.iloc[:,[0,1,2,3,4,5]].values
# biz şimdi verilerin p değerlerinihesaplamak için tüm kolonları aldık
X_list= np.array(X_list,dtype=float)
# tüm aldığımız kolonları türü float olan bir array yaptık
model=sm.OLS(boy_1,X_list).fit()
# burda bağımlı değişken olan boya bağımsız değişkenlerin etkisini tek tek 
# analizini yapıyor burda.
print(model.summary())
# bu print ettirdiğimiz şey çok önemli burda bağımsız değişkenlerin
# p değerlerini felan görüyoruz ona göre doğru tahmin için  
# hangi değişkeni alıp hangi değişkeni almamamız gerektiğini görüyoruz.
# geriye eleme yönteminde p değeri en yüksek olandan elemeye başlıyoruz.
# şimdi biz geriye eleme yönteminde genellikle p değeri 0.05 in altındaysa kabul edilebilir olarak
# alırız yani elde ettiğimiz modele göre 4. indexdeki sütunu yani x5'yi 
# 5. indexdeki sütunu yani x6 'yı elememiz gerekir çünkü p değerleri 0.05 'den büyük
 

X_list_1 = s_5.iloc[:,[0,1,2,3]].values
X_list_1= np.array(X_list_1,dtype=float)
model_1=sm.OLS(boy_1,X_list_1).fit()
print(model_1.summary())
# biz burda bir önceki modelde gördüğümüz boy için tahmini kötüleştiren değişkenleri attık
# şimdi atılmış hali ile yeni bir model oluşturuyoruz.kalan değerlerin p değerleri 0.05'in altında
# olduğu için değişkenleri değiştirmiyorum bidaha.
















