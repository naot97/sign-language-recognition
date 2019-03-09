from keras.utils import np_utils
import numpy as np
def load_mod():
	return load_model('model.h5')
def predict(model, img):
	a = model.predict(img,1)
	return np.argmax(a),np.max(a)*100 