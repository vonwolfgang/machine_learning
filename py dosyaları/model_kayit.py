
import pandas as pd

url = "https://bilkav.com/satislar.csv"

veri = pd.read_csv(url)

X = veri.iloc[:,0:1]
Y = veri.iloc[:,-1:]

bolme = 0.33

from sklearn import model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(X,Y,test_size = bolme)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)
# şimdi şu yukarıda yaptığımız fit etme işleminden sonra makine bir öğrenme modeli oluşturuyor
# bu öğrenme modelini kaydedicez.

import pickle
# pickle kütüphanesini import ettik

dosya = "model.kayit"
# dosya ismi oluşturduk

pickle.dump(lr,open(dosya,"wb"))
# oluşturduğumuz modeli pickle.dump ile kaydettik
# write and binary modunda kaydettik

uploded = pickle.load(open(dosya,"rb"))
# read and binary modunda uploded'a eşitledik
print(uploded.predict(x_test))
# artık lr.fit demeden direk bu değişkeni kullanarak predict ettirebiliriz.




