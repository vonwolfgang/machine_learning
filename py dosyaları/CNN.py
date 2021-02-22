
## KÜTÜPHANELER

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

#----------------------------------------------------

sql = Sequential()
# obje tanımladık

# ilkleme

# istediğimiz kadar convolution + pooling katmanları ekleyebiliriz.

# Convolution katmanı

sql.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
# okuduğumuz resimleri 64 'e 64 piksel boyutlarına indirgiyoruz bu yüzden 64 64 yazdır 3 'ün sebebide
# 3 tane katman vardı renk olarak red green blue ve 3 tane matrix oluşturuluyordu bu yüzden 3 yazdık. 

# Pooling katmanı

sql.add(MaxPooling2D(pool_size = (2,2)))

# Convolution katmanı

sql.add(Convolution2D(32, 3, 3, activation = "relu"))

# Pooling katmanı

sql.add(MaxPooling2D(pool_size=(2,2)))

#---------------------------------------------------

# Flattening

sql.add(Flatten())
# düzleştirme yaptık

#---------------------------------------------------


# YSA

sql.add(Dense(128, activation = "relu"))
# giriş fonksiyonu 128'lik output vericek avtivation fonksyionu relu
sql.add(Dense(1, activation = "sigmoid"))
# giriş ve çıkış fonksiyonları çıkış fonksiyonu sigmoid giriş relu

#----------------------------------------------------

# CNN

sql.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
# optimize fonksiyonu bu optimizasyon yapıyo accuracy'nin en iyi olduğu duruma göre optimize edicek

#-----------------------------------------------------

# CNN ve Resimler

from keras.preprocessing.image import ImageDataGenerator
# ImageDataGenerator resim okumaya yarar sırayla okuyor bu kütüphane.

train_datagen = ImageDataGenerator(rescale=1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale=1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
# 64'e 64 piksellik resimler çıkıyor. yukarıda test ve train kümelerindeki resimlerin hangi filtreler ile işleneceğini girdik

training_set = train_datagen.flow_from_directory("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\cinsiyet_resim\\veriler\\training_set", target_size=(64,64),batch_size=1,class_mode="binary")
test_set = train_datagen.flow_from_directory("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\cinsiyet_resim\\veriler\\test_set",target_size=(64,64),batch_size=1,class_mode="binary")
# yukarıda ise filtreleri belirlediğimiz train ve test şeylerine datanın adreslerini vererek import felan ettik

sql.fit_generator(training_set, steps_per_epoch = 8000, epochs=1, validation_data=test_set, validation_steps=2000)
# burda ise verimizi artık eğitiyoruz. Belli parametreler kullanıyoruz.

import numpy as np
import pandas as pd


test_set.reset()
pred = sql.predict_generator(test_set, verbose=1)
# yukarıda fonksiyonumuz da test_set için tahmin ettirdik liste şeklinde bişey döndürdü.

pred[pred>.5] = 1
pred[pred<=.5] = 0
# sonuç 1 ile 0 arasında bir değer döneceği için eğer 0.5'den büyük ise 1 
# 0.5'den küçük ise 0 döndür dedik.


test_labels = []

for i in range(0, int(203)):
    test_labels.extend(np.array(test_set[i][1]))
    
print("test_labels")
print(test_labels)

#---------------------------------------------------

folder_names = test_set.filenames
# dosya isimlerini aldık test_set verisinden

result = pd.DataFrame()
result["folder names"] = folder_names 
result["predicts"] = pred
result["test"] = test_labels
# yukarıda tek bir tablo üzerinde dosya isimlerini 
# sonra bizim o dosyayı ne olarak tahmin ettiğimiz ve gerçekte
# o dosyanın ne olduğunu gösteren bi dataframe yaptık

#-----------------------------------------------------

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, pred)
# gerçek değerler ile tahmin ettiğimiz değerler arasında confusion matrix oluşturduk

print("CM")
print(cm)






















