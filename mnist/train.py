from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense


(data_train,label_train),(data_test,label_test) = mnist.load_data()
print(len(data_train),len(data_test))
