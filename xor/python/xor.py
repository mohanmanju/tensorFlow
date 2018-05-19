from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

model = Sequential()
model.add(Dense(units=50,activation='sigmoid',input_dim=2))
model.add(Dense(units=10,activation='sigmoid'))
model.add(Dense(units=1))

model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

data = (np.random.randint(0,2,(10000,2)))
#data.append(list(np.random.randint(0,2,10)))
print(data)
data1 = np.transpose(data)
labels = data1[0]^data1[1]

model.fit(data,labels,epochs = 10)
model.save('xor.h5')



