from keras.models import Sequential
from keras.layers import Dense

from keras.models import model_from_json
import numpy as np
import os


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
dataTest = np.random.binomial(1,0.5,(1000,2))

result = np.transpose(dataTest)
labelsTest = result[0]^result[1]
score = loaded_model.evaluate(dataTest, labelsTest, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
