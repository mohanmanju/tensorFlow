from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_json
import numpy as np
import os

model = Sequential()
model.add(Dense(32,activation='relu',input_dim=2))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#train on 100000 data samples
data = np.random.binomial(1,0.5,(100000,2))

#to match the dimension of the input sepecified
result = np.transpose(data)
#true xor value of the data
labels = result[0]^result[1]

#trains for 5 iteration on the same data batch size is 1
model.fit(data,labels,epochs=5)

#test on 1000 samples
dataTest = np.random.binomial(1,0.5,(1000,2))

result = np.transpose(dataTest)
labelsTest = result[0]^result[1]


score = model.evaluate(dataTest,labelsTest)
#print the amount of loss on test data
print(score)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
