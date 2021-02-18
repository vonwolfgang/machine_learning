import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
# random sayı üretmemiz gerekiyormuş o yüzden bunu da import ettik

veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\Ads_CTR_Optimisation.csv")
# veriyi import ettik



#--------------------------------------------------

## RANDOM SELECTİON ALGORİTHMS
"""
N = 10000
# 10k satır olduğu için böle bişe yaptık 

d = 10

summary = 0

choosen = []

for n in range(0,N):
    ad = random.randrange(d)
    # burda ise rastgele bir sayı tanımla dedik 
    # d= 10 olduğu için 10 a kadar olan rakamlardan tanımlıcak
    
    choosen.append(ad)
    # burda seçilen değerleri choosen adlı listemize atadık.    
    
    award = veri.values[n,ad]
    # burda da sanırsam 10k satırı dönücek olan n 
    # eğer ve seçilen değerleri alan ad var bu değer seçilmişse
    # bunu award içine atıyor
    
    summary = summary + award
    # burdada seçilen değerleri topladık ama neden
    # yaptık gerçekten anlamadım.

plt.hist(choosen)
plt.show

# biz yukarıdaki algoritmada random selection yaptık yani 10k 'dan fazla
# gösterim var bunların kaçına tıklanılmış olduğunu gördük her tıklama 1
# her tıklamama 0 olduğu içinse her tıklamada ödül kazanmışız gibi 
# hanemize 1 ekledik ve sonuca göre 1300 ödül kazandık 
# 10k şeyden.
"""
#---------------------------------------------------------

### !!! REİNFORCED ALGORİTMALARI !!! 

#--------------------------------------------------------

# UPPER CONFİDİENCE BOUND

import math

# algoritma önceki yaptığı seçimlerden ders alarak 
# ilerliyor yani tahmini seçimler yapmıyor

N = 10000
# verimizde 10000 tane tıklama var diye atadık
d = 10
# verimizde 10 tane ilan var
awards_ucb = [0] * d # = Ri(n)
# burda ödülleri sıfır olarak dizi olarak tanımladık 10 tane elemanı olan ama her elemanı 0 olan bir dizi tanımladık
sum_ucb = 0 
# toplam olarak bişe tanımladık.
clicks_ucb = [0] * d # = Ni(n)
# aynı şekilde tıklama dizisi oluşturduk her bir elemanı 0 olan 
choosen_ucb = []
# boş bir liste oluşturduk

for n in range(0,N):
    # 10k tane tıklama olduğu için herbir tıklamayı kontrol etmemiz gerek
    ad_ucb = 0 
    # bu bizim seçmek istediğimiz yani aslında
    # en çok tıklama alan ilan bunu bulmaya çalışıyoruz
    max_ucb = 0 
    # max_ucb'yi 0 olarak tanımladık
    
    for i in range(0,d):
        # burdada tıklamalardan 10 tane ilan'a bakıyoruz
        
        if(clicks_ucb[i]>0):
            
            avarages_ucb = awards_ucb[i] / clicks_ucb[i]
            # ortalamayı hesapladık
            delta_ucb = math.sqrt(3/2*math.log(n)/clicks_ucb[i])
            # burda ise o ikinci formül gibi şeyi yaptık
            ucb = delta_ucb + avarages_ucb
            # burda ucb hesaplattık
        else:
            
            ucb = N*10
            
        
        if max_ucb < ucb:
            # max_ucb belirledik bundan büyük bir ucb var ise
            max_ucb = ucb
            # max dan büyük olan yeni max oluyor
            ad_ucb = i
            # dolayısıyla ordaki ilanda en iyi ilan olmuş oluyor.
    
    clicks_ucb[ad_ucb] += 1            
    choosen_ucb.append(ad_ucb)
    # seçilen reklamı listenin içine atadık               
    award_ucb = veri.values[n,ad_ucb] 
    awards_ucb[ad_ucb] += award_ucb
    sum_ucb += award_ucb

print("UCB toplam ödül")
print(sum_ucb,"\n")

print("UCB choosen grafik")
plt.hist(choosen_ucb)
plt.show

#------------------------------------------------------------------------

## THOMPSON ALGORİTMASI

import random

# algoritma önceki yaptığı seçimlerden ders alarak 
# ilerliyor yani tahmini seçimler yapmıyor

N = 10000
# verimizde 10000 tane tıklama var diye atadık

d = 10
# verimizde 10 tane ilan var

sum_thp = 0 
# toplam olarak bişe tanımladık.

choosen_thp = []
# boş bir liste oluşturduk

birler_thp = [0]*d
# bir gelenleri tutsun diye değeri sıfır olan bir dizi oluşturduk

sifirlar_thp = [0]*d
# sıfır gelenleri tutsun diye değeri sıfır olan bir dizi oluşturuk

for n in range(1,N):
    # 10k tane tıklama olduğu için herbir tıklamayı kontrol etmemiz gerek
    
    ad_thp = 0 
    # ilk başta seçilen ilanı 0 olarak atadık
    
    max_thp = 0 
    # max_thp'yi 0 olarak tanımladık
    
    for i in range(0,d):
        # burdada tıklamalardan 10 tane ilan'a bakıyoruz
        
        ras_beta = random.betavariate(birler_thp[i]+1, sifirlar_thp[i]+1)
        # rastgele beta değerleri oluşturmak için böyle bişe yaptık ama anlayamadım parametrelerini
        
        if(ras_beta>max_thp):
            # en büyük beta değerini bulmak için böyle bişe yapıyoruz.
            
            max_thp = ras_beta
            # ras_beta max_thp 'den büyükse yeni max_thp'miz ras_beta oluyor.
            
            ad_thp = i
            # bunun kaçıncı ilan olduğunu da ad_thp'ye atıyoruz.
            
         
    choosen_thp.append(ad_thp)
    # seçilen reklamı listenin içine atadık               
    
    award_thp = veri.values[n,ad_thp] 
    # burda verilerin içindeki bir ve sıfırları felan çektik
    
    if (award_thp == 1):
        
        birler_thp[ad_thp] += 1
        # eğer bu ilanın değeri 1'se bu ilanın birler değerini 1 artır dedik
    
    else:
        
        sifirlar_thp[ad_thp] += 1
        # şayet 1 değilse 0 'dır o zaman ve o zaman da bu ilanın sıfırlar değerini 1 artır
        
    
    sum_thp += award_thp

print("Thompsen toplam ödül")
print(sum_thp,"\n")

print("Thompsen choosen grafik")
plt.hist(choosen_thp)
plt.show

#--------------------------------------------------------------------





