import funcnn
import pandas as pd


data1 = pd.read_csv('iris.csv')
x1 = data1.iloc[:,0:4]
y1 = data1.iloc[:,4:5].values

w0,w1,w2,w3=funcnn.nn(x1,y1,10000,5,errorviz=True)

predict = [6.0,3.4,4.5,1.6] #1
predict1 = [5.1,3.7,1.5,0.4] #0
funcnn.tahmin(predict,w0,w1,w2,w3)
funcnn.tahmin(predict1,w0,w1,w2,w3)
