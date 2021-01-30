

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

boy = veri[["boy"]]
print(boy)
# yukarıdaki şey sayesinde sadece boy kolonunu alabiliyoruz

boykilo = veri[["boy","kilo"]]
print(boykilo)
# iki kolonu almak içinde yukarıdaki gibi yaptık

#-----------------------------------------------------

## PYTHON NESNE YÖNELİMLİ PROGRAMLAMA

class insan:
    boy = 180
    def kosmak(self,b):
        return b+10
# biz yukarıda bir insan sınıfı tanımladık 

ali = insan()
# bu yaptığımız sayesinde ali'yi insan sınıfından bi obje yaptık

print(ali.kosmak(10))
print(ali.boy)
# aliyi insan sınıfından bi obje yaptığımız için insan sınıfındaki metodu(fonksiyon) ve değişkeni  alide kullanabiliyoz.

#-----------------------------------------------------------

## EKSİK VERİLERİ DÜZELTME

eks_data = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\eksikveriler.csv")
# eksik olan dataları tamamlamamız gerekiz çünkü çalışmıyor diğer türlü 

# YÖNTEM

# eğer verimiz sayısalsa burda olduğunu gibi eksik verilerin bulunduğu kolonun ortalamasını alırız ve eksik yerlere bu ortalamayı atarız.

from sklearn.impute import SimpleImputer
# yukarıdaki atama işlemini yapabilmemiz için sklearn kütüphanesindeki impute modülündeki SimpleImputer sınıfını import ettik

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# SimpleImputerde kayıp verilerin ne olduğunu belirttik ve nasıl doldurması gerektiğini belirttik
# mean ortalama demek ortalamayla doldur dedik.

Yas = eks_data.iloc[:,1:4].values
print(Yas)
# yukarıda iloc ile yas kolonunu çekmeye çalıştık "," ün ne işe yaradığını bilmiyorum.

imputer = imputer.fit(Yas[:,1:4])
#yukarıda objeleştirdiğimiz imputerdeki fit fonksiyonu sayesinde Yas dan hangi kolonu alması gerektiğini öğrettik.

Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)
# yukarıda fit ile öğrettik şimdide transform ile nan olan değerlere öğrettiğimiz fit değerini atadık.

#------------------------------------------------------------

## KATEGORİK VERİ DÖNÜŞÜMÜ

ulke = veri.iloc[:,0:1].values
print(ulke)
# yukarıda kategorik verileri makinenin anlayacağı dile dönüştürmek için sadece ulke kolonunu aldık.

from sklearn import preprocessing
# dönüştürme işlemini yapabilmemiz için sklearn kütüphanesindeki preprocessing modulunu import ettik.

le = preprocessing.LabelEncoder()
# preprocessing modülündeki LabelEncoder() sınıfından le isimli bir obje oluşturduk.
# LabelEncoder() sanırsam kategorik olan verileri makine dilinin anlayacağı veri tipine dönüştürmeye yarıyor ama emin değilim

ulke[:,0] = le.fit_transform(veri.iloc[:,0])
print(ulke)
# yukarıda le objesine encode etmesini öğrettik sonra dedikki data daki sütuna ülkedeki encode edilmiş şeyleri ata dedik

one_hot_encoder = preprocessing.OneHotEncoder()
# preprocessing modülünden OneHotEncoder() sınıfından bir obje oluşturduk

ulke = one_hot_encoder.fit_transform(ulke).toarray()
print(ulke)
# yukarıda ne yaptığımızı anlamadım.One_Hot_Encoder amacı kolon başlıklarına etiketleri taşımak ve ilgili boşluklara 1, 0 felan yazmak

#---------------------------------------------------------

## VERİLERİN BİRLEŞTİRİLMESİ

sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc)
# yukarıda yaptığımız şey encode ettiğimiz "ulke" değişkenini DataFrame sayesinde kolonlara aktardık ve sayısal olarak tablolaştırdık.

sonuc_1 = pd.DataFrame(data=Yas, index=range(22), columns=['boy','kilo','yas'])
print(sonuc_1)
# yas, boy, kilo kolonlarınıda ekleyip birleştirdik.

cinsiyet = veri.iloc[:,-1].values
print(cinsiyet)
# veri kümemizden cinsiyet kolonunu liste halinde çektik
sonuc_2 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])
print(sonuc_2)
# liste haline getirdiğimiz cinsiyet kolonunu şimdi data frame halinde tekrardan tablo olarak yazdırdık

s=pd.concat([sonuc,sonuc_1],axis=1)
print(s)
# yukarıda concat ile elde ettiğimiz kolonları birleştirdik axis=1 dememiz şu işe yaradı 0. satırdakileri yan yana getir ve öle sırala 
s_1=pd.concat([s,sonuc_2],axis=1)
print(s_1)
# concat işlemi iki dataframe arasında yapıldığı için bi concat işlemi daha yapmak gerekti bu concat işlemindede cinsiyet kolonu ve bi önceki concat işleminden elde ettiğimiz kolonlar topluluğunu birleştirdik

#----------------------------------------------------------------

## EĞİTİM VE TEST KÜMESİ OLUŞTURMA

from sklearn.model_selection import train_test_split
# yukarıda sklearn kütüphanesinden model_selection modülünden train_test_split class ını ekledik

x_train, x_test, y_train, y_test = train_test_split(s,sonuc_2,test_size=0.33,random_state=0)
# ne işe yaradığıyla ya da ne yaptığımızla ilgili en ufak bir fikrim yok sor bunu
# ama sanırsam şöyle bişey olabilir bu yaptığımız ile seçtiğimiz verileri 4 e böldü test_size=0.33 dediğimiz için %33'ü test bölümüne atandı
# geri kalanı ise train olarak atandı ve bu atama random_state = 0 dediğimiz için atamayı rastgele yaptı.
 
#------------------------------------------------------------------

## ÖZNİTELİK ÖLÇEKLEME

from sklearn.preprocessing import StandardScaler
# sklearn kütüphanesinden preprocessing modülünden StandardScaler sınıfını import ettik.

sc=StandardScaler()
# sc değişkenini StandardScaler sınıfından bi obje haline getirdik.

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
# tam emin değilim ama sanırsam bu StandardScaler bu elimizdeki verileri aynı dünyanın verisi yapmaya yarıyor.
# yani boy ve kilo farklı veri türleri birbiri arasında kıyaslama yapmak olmuyo birbiri arasında kıyaslama yapmak için
# sanırsam hepsini birbirine benzer yapıyor birbirine göre ölçekliyor, optimize ediyor.









