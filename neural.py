from numpy import exp,array,random,dot,abs,mean
import numpy

girdi = array([[0,0,1],
               [1,1,1],
               [1,0,1]])

gercek_sonuc = array([[0,1,1]]).T #Transpose işlemi yapıyoruz. 3x3'ü 3x1 ile çarpop 3x1 elde edicez. Yani bu matrisi

agirlik = array([[1.0,1.0,1.0]]).T #Başlangıç rastgele ağırlıklarını verdik.

for tekrar in range(10000):
    #z = a*wa+b*wb
    z = dot(girdi,agirlik) #Vektör çarpımını yaptık
    tahmin = 1/(1+exp(-z)) #Sigmoid işlemini burda tanımladık
    agirlik +=dot(girdi.T,((gercek_sonuc-tahmin)*tahmin*(1-tahmin))) #Girdi x Hata x Sigmoid Türevi. Ne not aldıysan onu uyguluyoruz.
    print(str(mean(abs(gercek_sonuc-tahmin)))) #ortalamalarını göreceğiz hata değerimizin.

deneme =  1/(1+exp(-(dot(array([0,0,0]),agirlik)))) #bunun sonucunu görmek istiyoruz

print(deneme)