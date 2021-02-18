import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veri = pd.read_csv("C:\\Users\\Computer\\Desktop\\python-machine_learning\\Veriler\\sepet.csv", header=None)

t = []
for i in range(0,7051):
    t.append([str(veri.values[i,j]) for j in range(0,20)])
    
# yukarıda garip bir ikili for döngüsü yaptık. Bu for döngüleri sayesinde
# elimizdeki liste halinde felan olmayan verileri listeledik
    
#------------------------------------------------------------

## APRİORİ ALGORİTMASI 

from apyori import apriori
# github'dan indirdiğimiz kütüphaneyi import ettik
rule = apriori(t, min_support=0.01, min_confidence=0.2, min_lift=3, min_lenght=2)
print(list(rule))

# NOT BU ALGORİTMAYI ANLAYAMADIM.

#--------------------------------------------------------------


